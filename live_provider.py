"""Live grid data provider for CAISO.

Prices
    CAISO day-ahead hourly LMP for one trading hub, read from CAISO OASIS
    through the open-source ``gridstatus`` library. No API key is required.
    The day-ahead market publishes cleared prices for every hour of the next
    day, so most hours in the 24-hour window carry real cleared prices. Hours
    that CAISO has not published yet reuse the price of the same hour on the
    most recent published day and carry ``price_estimated=True``.

Carbon intensity
    EIA Open Data API v2, hourly net generation by fuel type for the CISO
    balancing authority. A free API key is required (``EIA_API_KEY``). EIA
    publishes this data after the fact, with a lag of several hours, and offers
    no forecast. Each hour in the window therefore reuses the fuel mix of the
    same hour of day from the most recent published day and carries
    ``carbon_estimated=True`` unless EIA has already published that exact hour.

Carbon intensity is generation-based: it reflects power plants inside the
balancing authority and ignores imports. The emission factors below are
approximate direct-combustion values. Review them before relying on the
absolute numbers.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests

from grid_client import RawGridPoint, RegionProfile

logger = logging.getLogger("voltedge.live")

DEFAULT_CAISO_NODE = "TH_SP15_GEN-APND"
EIA_FUEL_MIX_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
EIA_RESPONDENT = "CISO"
EIA_HISTORY_HOURS = 120
EIA_FUEL_TYPE_COUNT = 9
MAX_LOOKBACK_DAYS = 4
CACHE_TTL_SECONDS = 900
REQUEST_TIMEOUT_SECONDS = 30

# Approximate direct CO2 emissions per unit of net generation, in gCO2/kWh,
# keyed by EIA fuel type code. Fuel type "OTH" (other) has an unknown
# composition, so the calculation leaves it out of both the emissions and the
# generation totals.
EMISSION_FACTORS_G_PER_KWH: dict[str, float] = {
    "COL": 1040.0,
    "NG": 440.0,
    "OIL": 1080.0,
    "NUC": 0.0,
    "SUN": 0.0,
    "WND": 0.0,
    "WAT": 0.0,
    "GEO": 0.0,
}

PriceFetcher = Callable[[datetime, datetime], dict[datetime, float]]
FuelMixFetcher = Callable[[], dict[datetime, dict[str, float]]]


class LiveDataError(RuntimeError):
    """Raised when a live feed is unreachable, misconfigured, or incomplete."""


def carbon_intensity_from_mix(mix: dict[str, float]) -> float | None:
    """Convert an hourly fuel mix (MWh by EIA fuel code) into gCO2/kWh.

    Negative values (for example storage charging) count as zero. Return None
    when the mix holds no usable generation.
    """
    generation = 0.0
    emissions = 0.0
    for fuel, factor in EMISSION_FACTORS_G_PER_KWH.items():
        value = mix.get(fuel)
        if value is None or not math.isfinite(value) or value <= 0:
            continue
        generation += value
        emissions += value * factor
    if generation <= 0:
        return None
    return emissions / generation


def fetch_caiso_day_ahead_prices(node: str, start: datetime, end: datetime) -> dict[datetime, float]:
    """Return day-ahead hourly LMP (USD/MWh) keyed by UTC interval start."""
    try:
        import gridstatus
        import pandas as pd
    except ImportError as exc:
        raise LiveDataError("The 'gridstatus' package is not installed.") from exc

    logging.getLogger("gridstatus").setLevel(logging.WARNING)
    try:
        frame = gridstatus.CAISO().get_lmp(
            date=pd.Timestamp(start),
            end=pd.Timestamp(end),
            market="DAY_AHEAD_HOURLY",
            locations=[node],
            verbose=False,
        )
    except Exception as exc:
        raise LiveDataError(f"CAISO day-ahead price request failed: {exc}") from exc

    prices: dict[datetime, float] = {}
    for interval_start, lmp in zip(frame["Interval Start"], frame["LMP"]):
        value = float(lmp)
        if math.isfinite(value):
            prices[interval_start.tz_convert("UTC").to_pydatetime()] = value
    if not prices:
        raise LiveDataError("CAISO returned no day-ahead prices.")
    return prices


def fetch_eia_fuel_mix(api_key: str, respondent: str = EIA_RESPONDENT) -> dict[datetime, dict[str, float]]:
    """Return recent hourly net generation (MWh) by fuel type, keyed by UTC hour."""
    params = [
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[respondent][]", respondent),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", str(EIA_HISTORY_HOURS * EIA_FUEL_TYPE_COUNT)),
    ]
    try:
        # The key travels in a header so it never appears in URLs or logs.
        response = requests.get(
            EIA_FUEL_MIX_URL,
            params=params,
            headers={"X-Api-Key": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise LiveDataError(f"EIA fuel mix request failed: {exc}") from exc
    if response.status_code in (401, 403):
        raise LiveDataError("EIA rejected the API key. Check EIA_API_KEY.")
    if response.status_code >= 400:
        raise LiveDataError(f"EIA fuel mix request returned HTTP {response.status_code}.")

    try:
        rows = response.json()["response"]["data"]
    except (ValueError, KeyError, TypeError) as exc:
        raise LiveDataError("EIA returned a response in an unexpected format.") from exc

    mix: dict[datetime, dict[str, float]] = {}
    for row in rows:
        try:
            period = datetime.strptime(row["period"], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            value = float(row["value"])
            fuel = str(row["fueltype"])
        except (KeyError, TypeError, ValueError):
            continue
        mix.setdefault(period, {})[fuel] = value
    if not mix:
        raise LiveDataError("EIA returned no fuel mix data.")
    return mix


def fill_by_hour_of_day(
    series: dict[datetime, float], target: datetime, max_lookback_days: int = MAX_LOOKBACK_DAYS
) -> tuple[float, bool]:
    """Return (value, estimated) for ``target``.

    Use the exact hour when the series holds it. Otherwise step back one day at
    a time and reuse the same hour of day from the most recent day available.
    """
    if target in series:
        return series[target], False
    for days_back in range(1, max_lookback_days + 1):
        candidate = target - timedelta(days=days_back)
        if candidate in series:
            return series[candidate], True
    raise LiveDataError(
        f"No data within {max_lookback_days} days for the hour starting {target.isoformat()}."
    )


class CaisoLiveProvider:
    """Combines CAISO day-ahead prices with EIA-derived carbon intensity."""

    def __init__(
        self,
        eia_api_key: str,
        node: str = DEFAULT_CAISO_NODE,
        price_fetcher: PriceFetcher | None = None,
        fuel_mix_fetcher: FuelMixFetcher | None = None,
        cache_ttl_seconds: float = CACHE_TTL_SECONDS,
    ) -> None:
        if not eia_api_key:
            raise LiveDataError("EIA_API_KEY is not set.")
        self.node = node
        self.source_name = f"live (CAISO day-ahead LMP at {node}, EIA fuel mix)"
        self._price_fetcher: PriceFetcher = price_fetcher or (
            lambda start, end: fetch_caiso_day_ahead_prices(node, start, end)
        )
        self._fuel_mix_fetcher: FuelMixFetcher = fuel_mix_fetcher or (
            lambda: fetch_eia_fuel_mix(eia_api_key)
        )
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[tuple[datetime, int], tuple[float, list[RawGridPoint]]] = {}
        self._lock = threading.Lock()

    def fetch(self, profile: RegionProfile, start: datetime, hours: int) -> list[RawGridPoint]:
        if profile.code != "CAISO":
            raise LiveDataError(f"Live data covers CAISO only. No live feed exists for {profile.code}.")

        key = (start, hours)
        # The lock also stops concurrent requests from querying the upstream
        # feeds at the same time.
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and time.monotonic() - cached[0] < self._cache_ttl_seconds:
                return cached[1]
            points = self._build(start, hours)
            self._cache = {key: (time.monotonic(), points)}
            return points

    def _build(self, start: datetime, hours: int) -> list[RawGridPoint]:
        window_end = start + timedelta(hours=hours)
        prices = self._price_fetcher(start - timedelta(days=MAX_LOOKBACK_DAYS), window_end)
        fuel_mix = self._fuel_mix_fetcher()

        carbon: dict[datetime, float] = {}
        for period, mix in fuel_mix.items():
            intensity = carbon_intensity_from_mix(mix)
            if intensity is not None:
                carbon[period] = intensity
        if not carbon:
            raise LiveDataError("EIA fuel mix data holds no usable generation values.")

        points: list[RawGridPoint] = []
        for offset in range(hours):
            timestamp = start + timedelta(hours=offset)
            price, price_estimated = fill_by_hour_of_day(prices, timestamp)
            intensity, carbon_estimated = fill_by_hour_of_day(carbon, timestamp)
            points.append(
                RawGridPoint(
                    timestamp=timestamp,
                    price_usd_mwh=round(price, 2),
                    carbon_g_kwh=round(intensity, 1),
                    price_estimated=price_estimated,
                    carbon_estimated=carbon_estimated,
                )
            )
        logger.info(
            "Loaded live CAISO data: %d of %d prices and %d of %d carbon values are estimates.",
            sum(point.price_estimated for point in points),
            hours,
            sum(point.carbon_estimated for point in points),
            hours,
        )
        return points


def build_live_provider() -> CaisoLiveProvider:
    """Create the live provider from environment variables."""
    return CaisoLiveProvider(
        eia_api_key=os.getenv("EIA_API_KEY", "").strip(),
        node=os.getenv("VOLTEDGE_CAISO_NODE", DEFAULT_CAISO_NODE).strip() or DEFAULT_CAISO_NODE,
    )
