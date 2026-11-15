"""Grid data ingestion for VoltEdge.

``GridClient`` returns hourly locational marginal pricing (LMP, USD/MWh) and
carbon intensity (gCO2/kWh) for a region. It reads from a pluggable
``GridDataProvider``. The only provider shipped here is ``SimulatedGridProvider``,
which generates realistic diurnal curves so the platform runs with no API key.

To connect a live market feed, implement ``GridDataProvider.fetch`` and pass the
provider to ``GridClient``. If the live provider raises, the client falls back
to the simulation automatically.

All simulation parameters below are illustrative. They are not measured market
data for the named regions.
"""

from __future__ import annotations

import logging
import math
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger("voltedge.grid")

HORIZON_HOURS = 24
STRESS_PRICE_WEIGHT = 0.6
STRESS_CARBON_WEIGHT = 0.4
STRESS_THRESHOLD = 0.5


class UnknownRegionError(ValueError):
    """Raised when a caller requests a region that is not configured."""


@dataclass(frozen=True)
class RegionProfile:
    """Shape parameters for one simulated grid region."""

    code: str
    name: str
    timezone: str
    base_price: float
    diurnal_price_amplitude: float
    evening_price_spike: float
    base_carbon: float
    diurnal_carbon_amplitude: float
    evening_carbon_spike: float
    midday_solar_carbon_dip: float


REGIONS: dict[str, RegionProfile] = {
    "CAISO": RegionProfile(
        code="CAISO",
        name="California ISO",
        timezone="America/Los_Angeles",
        base_price=48.0,
        diurnal_price_amplitude=12.0,
        evening_price_spike=70.0,
        base_carbon=260.0,
        diurnal_carbon_amplitude=40.0,
        evening_carbon_spike=150.0,
        midday_solar_carbon_dip=35.0,
    ),
    "ERCOT": RegionProfile(
        code="ERCOT",
        name="Electric Reliability Council of Texas",
        timezone="America/Chicago",
        base_price=42.0,
        diurnal_price_amplitude=14.0,
        evening_price_spike=85.0,
        base_carbon=400.0,
        diurnal_carbon_amplitude=55.0,
        evening_carbon_spike=140.0,
        midday_solar_carbon_dip=25.0,
    ),
    "PJM": RegionProfile(
        code="PJM",
        name="PJM Interconnection",
        timezone="America/New_York",
        base_price=45.0,
        diurnal_price_amplitude=11.0,
        evening_price_spike=50.0,
        base_carbon=430.0,
        diurnal_carbon_amplitude=45.0,
        evening_carbon_spike=110.0,
        midday_solar_carbon_dip=10.0,
    ),
    "NYISO": RegionProfile(
        code="NYISO",
        name="New York ISO",
        timezone="America/New_York",
        base_price=50.0,
        diurnal_price_amplitude=12.0,
        evening_price_spike=55.0,
        base_carbon=300.0,
        diurnal_carbon_amplitude=35.0,
        evening_carbon_spike=95.0,
        midday_solar_carbon_dip=8.0,
    ),
}

DEFAULT_REGION = "CAISO"


@dataclass(frozen=True)
class RawGridPoint:
    """One hourly observation or forecast as delivered by a provider."""

    timestamp: datetime
    price_usd_mwh: float
    carbon_g_kwh: float


@dataclass(frozen=True)
class GridPoint:
    """One hourly point enriched with a grid stress score."""

    timestamp: datetime
    local_timestamp: datetime
    hour_of_day: int
    price_usd_mwh: float
    carbon_g_kwh: float
    stress_score: float
    is_stressed: bool


@dataclass(frozen=True)
class GridForecast:
    """A contiguous hourly forecast window. Index 0 is the current hour."""

    region: str
    region_name: str
    timezone: str
    source: str
    generated_at: datetime
    points: list[GridPoint]

    @property
    def prices(self) -> list[float]:
        return [point.price_usd_mwh for point in self.points]

    @property
    def carbon(self) -> list[float]:
        return [point.carbon_g_kwh for point in self.points]

    @property
    def stressed(self) -> list[bool]:
        return [point.is_stressed for point in self.points]


class GridDataProvider(Protocol):
    """Contract for any grid data source, live or simulated."""

    source_name: str

    def fetch(self, profile: RegionProfile, start: datetime, hours: int) -> list[RawGridPoint]:
        """Return exactly ``hours`` hourly points beginning at ``start`` (UTC, top of hour)."""


def _gaussian(x: float, center: float, width: float) -> float:
    return math.exp(-0.5 * ((x - center) / width) ** 2)


class SimulatedGridProvider:
    """Generates diurnal sinusoidal grid curves.

    Each curve dips overnight (minimum near 03:00 local time) and spikes between
    16:00 and 20:00 local time (peak near 18:00). Noise is seeded per region and
    per hour, so a given hour always produces the same values. That keeps the
    forecast stable while the 24-hour window rolls forward.
    """

    source_name = "simulation"

    def __init__(self, seed: int = 2024, price_noise: float = 2.0, carbon_noise: float = 6.0) -> None:
        self.seed = seed
        self.price_noise = price_noise
        self.carbon_noise = carbon_noise

    def _noise(self, region: str, timestamp: datetime) -> tuple[float, float]:
        key = f"{self.seed}:{region}:{timestamp.isoformat()}".encode("utf-8")
        rng = np.random.default_rng(zlib.crc32(key))
        return (
            float(rng.normal(0.0, self.price_noise)),
            float(rng.normal(0.0, self.carbon_noise)),
        )

    def _point(self, profile: RegionProfile, timestamp: datetime) -> RawGridPoint:
        local = timestamp.astimezone(ZoneInfo(profile.timezone))
        hour = local.hour + local.minute / 60.0

        # A 24-hour cosine with its trough at 03:00 and its crest at 15:00.
        diurnal = math.cos(2.0 * math.pi * (hour - 15.0) / 24.0)
        evening_spike = _gaussian(hour, 18.0, 1.7)
        morning_ramp = _gaussian(hour, 8.0, 1.5)
        midday_solar = _gaussian(hour, 12.0, 2.0)
        price_noise, carbon_noise = self._noise(profile.code, timestamp)

        price = (
            profile.base_price
            + profile.diurnal_price_amplitude * diurnal
            + profile.evening_price_spike * evening_spike
            + 0.12 * profile.evening_price_spike * morning_ramp
            + price_noise
        )
        carbon = (
            profile.base_carbon
            + profile.diurnal_carbon_amplitude * diurnal
            + profile.evening_carbon_spike * evening_spike
            - profile.midday_solar_carbon_dip * midday_solar
            + carbon_noise
        )
        return RawGridPoint(
            timestamp=timestamp,
            price_usd_mwh=round(max(price, 1.0), 2),
            carbon_g_kwh=round(max(carbon, 10.0), 1),
        )

    def fetch(self, profile: RegionProfile, start: datetime, hours: int) -> list[RawGridPoint]:
        return [self._point(profile, start + timedelta(hours=offset)) for offset in range(hours)]


def _normalize(values: list[float]) -> list[float]:
    """Scale values to the 0..1 range. A flat series maps to all zeros."""
    low = min(values)
    high = max(values)
    span = high - low
    if span <= 1e-9:
        return [0.0 for _ in values]
    return [(value - low) / span for value in values]


def compute_stress_scores(prices: list[float], carbon: list[float]) -> list[float]:
    """Blend normalized price and carbon into a 0..1 grid stress score per hour."""
    if len(prices) != len(carbon):
        raise ValueError("Price and carbon series must have the same length.")
    if not prices:
        return []
    price_norm = _normalize(prices)
    carbon_norm = _normalize(carbon)
    return [
        round(STRESS_PRICE_WEIGHT * p + STRESS_CARBON_WEIGHT * c, 4)
        for p, c in zip(price_norm, carbon_norm)
    ]


def classify_stress(score: float) -> str:
    """Translate a stress score into a label for dashboards and API consumers."""
    if score >= 0.75:
        return "critical"
    if score >= STRESS_THRESHOLD:
        return "high"
    if score >= 0.25:
        return "moderate"
    return "low"


def resolve_region(region: str) -> RegionProfile:
    """Look up a region profile by code, case-insensitively."""
    code = (region or "").strip().upper()
    if code not in REGIONS:
        supported = ", ".join(sorted(REGIONS))
        raise UnknownRegionError(f"Unknown region '{region}'. Supported regions: {supported}.")
    return REGIONS[code]


class GridClient:
    """Fetches an hourly grid forecast, with automatic fallback to simulation."""

    def __init__(self, provider: GridDataProvider | None = None) -> None:
        self._simulator = SimulatedGridProvider()
        self._provider: GridDataProvider = provider if provider is not None else self._simulator

    @staticmethod
    def _validate(points: list[RawGridPoint], hours: int) -> None:
        if len(points) != hours:
            raise ValueError(f"Provider returned {len(points)} points, expected {hours}.")
        for point in points:
            if not (math.isfinite(point.price_usd_mwh) and math.isfinite(point.carbon_g_kwh)):
                raise ValueError("Provider returned a non-finite price or carbon value.")

    def get_forecast(
        self,
        region: str = DEFAULT_REGION,
        hours: int = HORIZON_HOURS,
        start: datetime | None = None,
    ) -> GridForecast:
        """Return ``hours`` hourly points starting at the current hour (or ``start``)."""
        if hours < 1:
            raise ValueError("The forecast horizon must cover at least one hour.")
        profile = resolve_region(region)

        now = datetime.now(timezone.utc)
        reference = start if start is not None else now
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        window_start = reference.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)

        source = self._provider.source_name
        try:
            raw_points = self._provider.fetch(profile, window_start, hours)
            self._validate(raw_points, hours)
        except Exception as exc:
            if self._provider is self._simulator:
                raise
            logger.warning(
                "Grid provider '%s' failed (%s). Falling back to simulation.", source, exc
            )
            raw_points = self._simulator.fetch(profile, window_start, hours)
            source = self._simulator.source_name

        prices = [point.price_usd_mwh for point in raw_points]
        carbon = [point.carbon_g_kwh for point in raw_points]
        scores = compute_stress_scores(prices, carbon)
        zone = ZoneInfo(profile.timezone)

        points: list[GridPoint] = []
        for raw, score in zip(raw_points, scores):
            local = raw.timestamp.astimezone(zone)
            points.append(
                GridPoint(
                    timestamp=raw.timestamp,
                    local_timestamp=local,
                    hour_of_day=local.hour,
                    price_usd_mwh=raw.price_usd_mwh,
                    carbon_g_kwh=raw.carbon_g_kwh,
                    stress_score=score,
                    is_stressed=score >= STRESS_THRESHOLD,
                )
            )

        return GridForecast(
            region=profile.code,
            region_name=profile.name,
            timezone=profile.timezone,
            source=source,
            generated_at=now,
            points=points,
        )
