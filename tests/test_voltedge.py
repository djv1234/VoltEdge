"""Tests for the solver, the grid simulation, the database fallback, and the API."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database  # noqa: E402
import solver  # noqa: E402
from grid_client import GridClient, RawGridPoint, UnknownRegionError  # noqa: E402


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setenv("VOLTEDGE_SQLITE_PATH", str(tmp_path / "test.db"))
    database.reset_database()
    import main

    yield TestClient(main.app)
    database.reset_database()


def test_solver_picks_cheapest_hours():
    prices = [50.0] * 24
    for hour in (3, 7, 11, 20):
        prices[hour] = 10.0
    carbon = [300.0] * 24
    result = solver.optimize_schedule(prices, carbon, required_hours=4, power_kw=1000.0)
    assert [hour for hour, flag in enumerate(result.mask) if flag] == [3, 7, 11, 20]
    assert result.solver_status == "optimal"
    assert result.optimized.cost_usd == pytest.approx(40.0)
    assert result.baseline.cost_usd == pytest.approx(160.0)
    assert result.cost_savings_pct == pytest.approx(75.0)


def test_solver_matches_greedy_on_simulated_data():
    forecast = GridClient().get_forecast("ERCOT")
    for required in range(1, 25):
        result = solver.optimize_schedule(forecast.prices, forecast.carbon, required, 750.0)
        expected = sum(sorted(forecast.prices)[:required]) * 0.75
        assert sum(result.mask) == required
        assert set(result.mask) <= {0, 1}
        assert result.optimized.cost_usd == pytest.approx(expected)
        assert result.optimized.cost_usd <= result.baseline.cost_usd + 1e-9


def test_solver_respects_deadline():
    prices = [100.0] * 12 + [1.0] * 12
    result = solver.optimize_schedule(prices, [200.0] * 24, 5, 100.0, deadline_hours=12)
    assert sum(result.mask[:12]) == 5
    assert sum(result.mask[12:]) == 0


def test_solver_carbon_price_shifts_schedule():
    prices = [20.0, 21.0] + [90.0] * 22
    carbon = [900.0, 100.0] + [500.0] * 22
    cost_only = solver.optimize_schedule(prices, carbon, 1, 1000.0)
    carbon_aware = solver.optimize_schedule(prices, carbon, 1, 1000.0, carbon_price_usd_per_tonne=100.0)
    assert cost_only.mask[0] == 1
    assert carbon_aware.mask[1] == 1


def test_solver_rejects_infeasible_and_malformed_inputs():
    prices = [30.0] * 24
    carbon = [300.0] * 24
    with pytest.raises(solver.InfeasibleScheduleError):
        solver.optimize_schedule(prices, carbon, 25, 100.0)
    with pytest.raises(solver.InfeasibleScheduleError):
        solver.optimize_schedule(prices, carbon, 8, 100.0, deadline_hours=6)
    with pytest.raises(solver.ScheduleInputError):
        solver.optimize_schedule(prices, carbon, 0, 100.0)
    with pytest.raises(solver.ScheduleInputError):
        solver.optimize_schedule(prices, carbon, 4, -5.0)
    with pytest.raises(solver.ScheduleInputError):
        solver.optimize_schedule(prices, carbon[:-1], 4, 100.0)
    with pytest.raises(solver.ScheduleInputError):
        solver.optimize_schedule([float("nan")] * 24, carbon, 4, 100.0)


def test_solver_falls_back_to_greedy_when_cbc_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("solver binary missing")

    monkeypatch.setattr(solver, "_solve_milp", broken)
    prices = [float(hour) for hour in range(24, 0, -1)]
    result = solver.optimize_schedule(prices, [300.0] * 24, 3, 100.0)
    assert result.solver_status == "optimal_greedy_fallback"
    assert result.mask[-3:] == [1, 1, 1]


def test_simulation_dips_overnight_and_spikes_in_the_evening():
    start = datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)  # local midnight in CAISO
    forecast = GridClient().get_forecast("CAISO", start=start)
    assert len(forecast.points) == 24
    by_hour = {point.hour_of_day: point for point in forecast.points}
    peak_hour = max(by_hour, key=lambda hour: by_hour[hour].price_usd_mwh)
    trough_hour = min(by_hour, key=lambda hour: by_hour[hour].price_usd_mwh)
    assert 16 <= peak_hour <= 20
    assert 0 <= trough_hour <= 6
    carbon_peak_hour = max(by_hour, key=lambda hour: by_hour[hour].carbon_g_kwh)
    assert 16 <= carbon_peak_hour <= 20
    stressed_hours = {hour for hour, point in by_hour.items() if point.is_stressed}
    assert {17, 18, 19} <= stressed_hours
    assert stressed_hours <= set(range(15, 22))
    again = GridClient().get_forecast("CAISO", start=start)
    assert again.prices == forecast.prices


def test_grid_client_falls_back_when_live_provider_fails():
    class BrokenProvider:
        source_name = "live"

        def fetch(self, profile, start, hours) -> list[RawGridPoint]:
            raise ConnectionError("upstream down")

    forecast = GridClient(provider=BrokenProvider()).get_forecast("PJM")
    assert forecast.source == "simulation"
    assert len(forecast.points) == 24
    with pytest.raises(UnknownRegionError):
        GridClient().get_forecast("ATLANTIS")


def test_database_falls_back_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://invalid.invalid")
    monkeypatch.setenv("SUPABASE_KEY", "not-a-real-key")
    monkeypatch.setenv("VOLTEDGE_SQLITE_PATH", str(tmp_path / "fallback.db"))
    db = database.build_database()
    assert db.backend_name == "sqlite"
    assert db.ping()

    row = {
        "region": "CAISO",
        "ts": "2026-01-01T00:00:00+00:00",
        "price_usd_mwh": 40.0,
        "carbon_g_kwh": 250.0,
        "stress_score": 0.2,
        "is_stressed": False,
        "source": "simulation",
    }
    assert db.upsert_grid_logs([row]) == 1
    assert db.upsert_grid_logs([{**row, "price_usd_mwh": 45.0}]) == 1
    logs = db.fetch_grid_logs("CAISO")
    assert len(logs) == 1
    assert logs[0]["price_usd_mwh"] == 45.0

    monkeypatch.setenv("VOLTEDGE_SQLITE_PATH", str(tmp_path / "missing_dir" / "x.db"))
    monkeypatch.delenv("SUPABASE_URL")
    assert database.build_database().backend_name == "sqlite-memory"


def test_api_end_to_end(api_client):
    health = api_client.get("/health")
    assert health.status_code == 200
    assert health.json()["storage_backend"] == "sqlite"

    grid = api_client.get("/grid/metrics", params={"region": "caiso"})
    assert grid.status_code == 200
    body = grid.json()
    assert len(body["forecast"]) == 24
    assert body["current"] == body["forecast"][0]
    assert body["persisted"] is True

    response = api_client.post("/schedule/optimize", json={"required_hours": 6, "power_kw": 500})
    assert response.status_code == 200
    schedule = response.json()
    assert len(schedule["execution_mask"]) == 24
    assert sum(schedule["execution_mask"]) == 6
    assert schedule["persisted"] is True
    assert [hour["price_usd_mwh"] for hour in schedule["hours"]] == [
        hour["price_usd_mwh"] for hour in body["forecast"]
    ]

    history = api_client.get("/schedule/history").json()
    assert history["count"] == 1
    assert history["schedules"][0]["execution_mask"] == schedule["execution_mask"]


def test_api_rejects_bad_requests(api_client):
    too_long = api_client.post("/schedule/optimize", json={"required_hours": 30, "power_kw": 500})
    assert too_long.status_code == 422
    assert "30 compute hours" in too_long.json()["detail"]

    past_deadline = api_client.post(
        "/schedule/optimize", json={"required_hours": 8, "power_kw": 500, "deadline_hours": 4}
    )
    assert past_deadline.status_code == 422

    assert api_client.post("/schedule/optimize", json={"required_hours": 0, "power_kw": 500}).status_code == 422
    assert api_client.post("/schedule/optimize", json={"required_hours": 4, "power_kw": -1}).status_code == 422
    assert api_client.post("/schedule/optimize", json={"required_hours": 4}).status_code == 422
    assert (
        api_client.post(
            "/schedule/optimize", json={"required_hours": 4, "power_kw": 100, "region": "ATLANTIS"}
        ).status_code
        == 404
    )
    assert api_client.get("/grid/metrics", params={"region": "ATLANTIS"}).status_code == 404
