"""VoltEdge API: grid metrics and workload schedule optimization.

Run locally with:

    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from database import DatabaseError, get_database
from grid_client import (
    DEFAULT_REGION,
    HORIZON_HOURS,
    REGIONS,
    STRESS_THRESHOLD,
    GridClient,
    GridForecast,
    UnknownRegionError,
    classify_stress,
)
from solver import InfeasibleScheduleError, ScheduleInputError, optimize_schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voltedge.api")

app = FastAPI(
    title="VoltEdge",
    description="Intelligent Workload Scheduler and Grid Arbitrage Engine",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

grid_client = GridClient()


class RegionInfo(BaseModel):
    code: str
    name: str
    timezone: str


class GridHour(BaseModel):
    timestamp: datetime
    local_timestamp: datetime
    hour_of_day: int
    price_usd_mwh: float
    carbon_g_kwh: float
    stress_score: float
    stress_level: str
    is_stressed: bool


class GridSummary(BaseModel):
    mean_price_usd_mwh: float
    min_price_usd_mwh: float
    max_price_usd_mwh: float
    peak_price_local_timestamp: datetime
    mean_carbon_g_kwh: float
    min_carbon_g_kwh: float
    max_carbon_g_kwh: float
    peak_carbon_local_timestamp: datetime
    stressed_hours: int
    stress_threshold: float


class GridMetricsResponse(BaseModel):
    region: str
    region_name: str
    timezone: str
    source: str
    generated_at: datetime
    current: GridHour
    forecast: list[GridHour]
    summary: GridSummary
    persisted: bool
    storage_backend: str


class OptimizeRequest(BaseModel):
    required_hours: int = Field(..., ge=1, description="Exact compute hours the job needs.")
    power_kw: float = Field(..., gt=0, le=1_000_000, description="GPU cluster power draw in kW.")
    region: str = Field(DEFAULT_REGION, description="Grid region code.")
    deadline_hours: int = Field(
        HORIZON_HOURS, ge=1, le=HORIZON_HOURS, description="Hours from now by which the job must finish."
    )
    carbon_price_usd_per_tonne: float = Field(
        0.0, ge=0, le=10_000, description="Optional internal carbon price added to the objective."
    )
    persist: bool = Field(True, description="Store the schedule in the database.")


class ScheduleHour(BaseModel):
    timestamp: datetime
    local_timestamp: datetime
    hour_of_day: int
    price_usd_mwh: float
    carbon_g_kwh: float
    stress_score: float
    is_stressed: bool
    run: int
    baseline_run: int


class ScheduleMetrics(BaseModel):
    energy_kwh: float
    optimized_cost_usd: float
    baseline_cost_usd: float
    cost_savings_usd: float
    cost_savings_pct: float
    optimized_carbon_kg: float
    baseline_carbon_kg: float
    carbon_avoided_kg: float
    optimized_stressed_hours: int
    baseline_stressed_hours: int
    stressed_hours_avoided: int


class OptimizeResponse(BaseModel):
    region: str
    region_name: str
    timezone: str
    source: str
    generated_at: datetime
    horizon_start: datetime
    required_hours: int
    deadline_hours: int
    power_kw: float
    carbon_price_usd_per_tonne: float
    execution_mask: list[int]
    baseline_mask: list[int]
    hours: list[ScheduleHour]
    metrics: ScheduleMetrics
    solver_status: str
    persisted: bool
    storage_backend: str


def _load_forecast(region: str) -> GridForecast:
    try:
        return grid_client.get_forecast(region=region, hours=HORIZON_HOURS)
    except UnknownRegionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Grid forecast failed for region '%s'.", region)
        raise HTTPException(status_code=502, detail=f"Grid data is unavailable: {exc}") from exc


def _persist_grid_logs(forecast: GridForecast) -> bool:
    rows = [
        {
            "region": forecast.region,
            "ts": point.timestamp.isoformat(),
            "price_usd_mwh": point.price_usd_mwh,
            "carbon_g_kwh": point.carbon_g_kwh,
            "stress_score": point.stress_score,
            "is_stressed": point.is_stressed,
            "source": forecast.source,
        }
        for point in forecast.points
    ]
    try:
        get_database().upsert_grid_logs(rows)
    except DatabaseError as exc:
        logger.warning("Grid logs were not stored: %s", exc)
        return False
    return True


def _grid_hour(point: Any) -> GridHour:
    return GridHour(
        timestamp=point.timestamp,
        local_timestamp=point.local_timestamp,
        hour_of_day=point.hour_of_day,
        price_usd_mwh=point.price_usd_mwh,
        carbon_g_kwh=point.carbon_g_kwh,
        stress_score=point.stress_score,
        stress_level=classify_stress(point.stress_score),
        is_stressed=point.is_stressed,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
def health() -> dict[str, Any]:
    database = get_database()
    return {
        "status": "ok",
        "storage_backend": database.backend_name,
        "storage_reachable": database.ping(),
        "grid_source": grid_client.get_forecast(DEFAULT_REGION, hours=1).source,
    }


@app.get("/regions", response_model=list[RegionInfo])
def list_regions() -> list[RegionInfo]:
    return [
        RegionInfo(code=profile.code, name=profile.name, timezone=profile.timezone)
        for profile in REGIONS.values()
    ]


@app.get("/grid/metrics", response_model=GridMetricsResponse)
def grid_metrics(
    region: str = Query(DEFAULT_REGION, description="Grid region code."),
) -> GridMetricsResponse:
    """Return the current hour plus a 24-hour forecast of price, carbon, and grid stress."""
    forecast = _load_forecast(region)
    persisted = _persist_grid_logs(forecast)

    hours = [_grid_hour(point) for point in forecast.points]
    prices = forecast.prices
    carbon = forecast.carbon
    peak_price_index = max(range(len(prices)), key=lambda index: prices[index])
    peak_carbon_index = max(range(len(carbon)), key=lambda index: carbon[index])

    summary = GridSummary(
        mean_price_usd_mwh=round(sum(prices) / len(prices), 2),
        min_price_usd_mwh=min(prices),
        max_price_usd_mwh=max(prices),
        peak_price_local_timestamp=forecast.points[peak_price_index].local_timestamp,
        mean_carbon_g_kwh=round(sum(carbon) / len(carbon), 1),
        min_carbon_g_kwh=min(carbon),
        max_carbon_g_kwh=max(carbon),
        peak_carbon_local_timestamp=forecast.points[peak_carbon_index].local_timestamp,
        stressed_hours=sum(forecast.stressed),
        stress_threshold=STRESS_THRESHOLD,
    )
    return GridMetricsResponse(
        region=forecast.region,
        region_name=forecast.region_name,
        timezone=forecast.timezone,
        source=forecast.source,
        generated_at=forecast.generated_at,
        current=hours[0],
        forecast=hours,
        summary=summary,
        persisted=persisted,
        storage_backend=get_database().backend_name,
    )


@app.post("/schedule/optimize", response_model=OptimizeResponse)
def schedule_optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Return the binary execution mask that minimizes the cost of the training job."""
    forecast = _load_forecast(request.region)

    try:
        result = optimize_schedule(
            prices=forecast.prices,
            carbon=forecast.carbon,
            required_hours=request.required_hours,
            power_kw=request.power_kw,
            deadline_hours=request.deadline_hours,
            carbon_price_usd_per_tonne=request.carbon_price_usd_per_tonne,
        )
    except (InfeasibleScheduleError, ScheduleInputError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Schedule optimization failed.")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc

    stressed = forecast.stressed
    optimized_stressed = sum(1 for flag, hot in zip(result.mask, stressed) if flag and hot)
    baseline_stressed = sum(1 for flag, hot in zip(result.baseline_mask, stressed) if flag and hot)

    metrics = ScheduleMetrics(
        energy_kwh=round(result.optimized.energy_kwh, 3),
        optimized_cost_usd=round(result.optimized.cost_usd, 2),
        baseline_cost_usd=round(result.baseline.cost_usd, 2),
        cost_savings_usd=round(result.cost_savings_usd, 2),
        cost_savings_pct=round(result.cost_savings_pct, 2),
        optimized_carbon_kg=round(result.optimized.carbon_kg, 2),
        baseline_carbon_kg=round(result.baseline.carbon_kg, 2),
        carbon_avoided_kg=round(result.carbon_avoided_kg, 2),
        optimized_stressed_hours=optimized_stressed,
        baseline_stressed_hours=baseline_stressed,
        stressed_hours_avoided=baseline_stressed - optimized_stressed,
    )

    horizon_start = forecast.points[0].timestamp
    persisted = False
    if request.persist:
        _persist_grid_logs(forecast)
        try:
            get_database().insert_schedule(
                {
                    "region": forecast.region,
                    "horizon_start": horizon_start.isoformat(),
                    "required_hours": request.required_hours,
                    "deadline_hours": request.deadline_hours,
                    "power_kw": request.power_kw,
                    "carbon_price_usd_per_tonne": request.carbon_price_usd_per_tonne,
                    "execution_mask": result.mask,
                    "baseline_mask": result.baseline_mask,
                    "optimized_cost_usd": metrics.optimized_cost_usd,
                    "baseline_cost_usd": metrics.baseline_cost_usd,
                    "cost_savings_pct": metrics.cost_savings_pct,
                    "carbon_avoided_kg": metrics.carbon_avoided_kg,
                    "stressed_hours_avoided": metrics.stressed_hours_avoided,
                    "solver_status": result.solver_status,
                    "data_source": forecast.source,
                }
            )
            persisted = True
        except DatabaseError as exc:
            logger.warning("Schedule was not stored: %s", exc)

    hours = [
        ScheduleHour(
            timestamp=point.timestamp,
            local_timestamp=point.local_timestamp,
            hour_of_day=point.hour_of_day,
            price_usd_mwh=point.price_usd_mwh,
            carbon_g_kwh=point.carbon_g_kwh,
            stress_score=point.stress_score,
            is_stressed=point.is_stressed,
            run=run,
            baseline_run=baseline_run,
        )
        for point, run, baseline_run in zip(forecast.points, result.mask, result.baseline_mask)
    ]

    return OptimizeResponse(
        region=forecast.region,
        region_name=forecast.region_name,
        timezone=forecast.timezone,
        source=forecast.source,
        generated_at=forecast.generated_at,
        horizon_start=horizon_start,
        required_hours=request.required_hours,
        deadline_hours=request.deadline_hours,
        power_kw=request.power_kw,
        carbon_price_usd_per_tonne=request.carbon_price_usd_per_tonne,
        execution_mask=result.mask,
        baseline_mask=result.baseline_mask,
        hours=hours,
        metrics=metrics,
        solver_status=result.solver_status,
        persisted=persisted,
        storage_backend=get_database().backend_name,
    )


@app.get("/schedule/history")
def schedule_history(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """Return the most recently stored schedules."""
    database = get_database()
    try:
        schedules = database.fetch_recent_schedules(limit=limit)
    except DatabaseError as exc:
        raise HTTPException(status_code=503, detail=f"Schedule history is unavailable: {exc}") from exc
    return {"storage_backend": database.backend_name, "count": len(schedules), "schedules": schedules}
