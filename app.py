"""VoltEdge dashboard. Talks to the FastAPI backend defined in main.py.

Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("VOLTEDGE_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 15

PRICE_COLOR = "#2a78d6"
CARBON_COLOR = "#eb6834"
SCHEDULE_COLOR = "#1baf7a"
BASELINE_COLOR = "#8a8984"
STRESS_COLOR = "#d03b3b"

FALLBACK_REGIONS = [{"code": "CAISO", "name": "California ISO", "timezone": "America/Los_Angeles"}]


class BackendError(RuntimeError):
    """Raised when the API is unreachable or rejects a request."""


def _error_detail(response: requests.Response) -> str:
    """Extract a readable message from a FastAPI error payload."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text or f"HTTP {response.status_code}"
    if isinstance(detail, list):
        messages = []
        for item in detail:
            if isinstance(item, dict):
                location = ".".join(str(part) for part in item.get("loc", []) if part != "body")
                messages.append(f"{location}: {item.get('msg', 'invalid value')}")
            else:
                messages.append(str(item))
        return "; ".join(messages)
    return str(detail) if detail else f"HTTP {response.status_code}"


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_URL}{path}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.ConnectionError as exc:
        raise BackendError(
            f"Cannot reach the VoltEdge API at {API_URL}. "
            "Start it with: uvicorn main:app --port 8000"
        ) from exc
    except requests.Timeout as exc:
        raise BackendError(f"The VoltEdge API at {API_URL} timed out.") from exc
    except requests.RequestException as exc:
        raise BackendError(f"Request to the VoltEdge API failed: {exc}") from exc
    if response.status_code >= 400:
        raise BackendError(_error_detail(response))
    try:
        return response.json()
    except ValueError as exc:
        raise BackendError("The VoltEdge API returned a response that is not valid JSON.") from exc


@st.cache_data(ttl=600, show_spinner=False)
def fetch_regions() -> list[dict[str, str]]:
    return _request("GET", "/regions")


@st.cache_data(ttl=120, show_spinner=False)
def fetch_grid_metrics(region: str, horizon_key: str) -> dict[str, Any]:
    # horizon_key changes every hour, so the cache never serves a stale window.
    return _request("GET", "/grid/metrics", params={"region": region})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schedule(
    region: str,
    required_hours: int,
    power_kw: float,
    deadline_hours: int,
    carbon_price: float,
    horizon_key: str,
) -> dict[str, Any]:
    # Caching on the full parameter set also prevents duplicate rows in the
    # database when Streamlit reruns the script without a parameter change.
    payload = {
        "region": region,
        "required_hours": required_hours,
        "power_kw": power_kw,
        "deadline_hours": deadline_hours,
        "carbon_price_usd_per_tonne": carbon_price,
        "persist": True,
    }
    return _request("POST", "/schedule/optimize", json=payload)


def fetch_history(limit: int = 10) -> dict[str, Any]:
    return _request("GET", "/schedule/history", params={"limit": limit})


def build_hourly_frame(schedule: dict[str, Any]) -> pd.DataFrame:
    """Flatten the API response into one row per forecast hour."""
    rows = []
    for index, hour in enumerate(schedule["hours"]):
        local = datetime.fromisoformat(hour["local_timestamp"]).replace(tzinfo=None)
        rows.append(
            {
                "index": index,
                "local_time": local,
                "label": local.strftime("%a %H:%M"),
                "tick": local.strftime("%H:%M"),
                "price": float(hour["price_usd_mwh"]),
                "carbon": float(hour["carbon_g_kwh"]),
                "stress_score": float(hour["stress_score"]),
                "is_stressed": bool(hour["is_stressed"]),
                "run": int(hour["run"]),
                "baseline_run": int(hour["baseline_run"]),
            }
        )
    frame = pd.DataFrame(rows)
    # A daylight-saving fall-back repeats one local hour. Keep category labels unique.
    if frame["label"].duplicated().any():
        frame["label"] = [f"{label} (+{index}h)" for label, index in zip(frame["label"], frame["index"])]
    return frame


def build_chart(frame: pd.DataFrame, deadline_hours: int) -> go.Figure:
    """Overlay schedule bars on the price curve and the carbon intensity line."""
    labels = frame["label"].tolist()
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=labels,
            y=frame["run"].tolist(),
            name="Optimized training hours",
            yaxis="y3",
            marker={"color": SCHEDULE_COLOR, "opacity": 0.30, "line": {"width": 0}},
            customdata=["GPUs running" if flag else "GPUs paused" for flag in frame["run"]],
            hovertemplate="%{customdata}<extra>Optimized schedule</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=frame["baseline_run"].tolist(),
            name="Baseline (run immediately)",
            yaxis="y3",
            visible="legendonly",
            marker={
                "color": BASELINE_COLOR,
                "opacity": 0.30,
                "line": {"width": 0},
                "pattern": {"shape": "/", "fgcolor": BASELINE_COLOR, "fgopacity": 0.8},
            },
            customdata=["GPUs running" if flag else "GPUs idle" for flag in frame["baseline_run"]],
            hovertemplate="%{customdata}<extra>Baseline schedule</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[0.025 if flag else 0.0 for flag in frame["is_stressed"]],
            name="Stressed grid hour",
            yaxis="y3",
            marker={"color": STRESS_COLOR, "opacity": 0.95, "line": {"width": 0}},
            customdata=["Yes" if flag else "No" for flag in frame["is_stressed"]],
            hovertemplate="%{customdata}<extra>Grid stressed</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=frame["price"].tolist(),
            name="Electricity price (USD/MWh)",
            yaxis="y",
            mode="lines+markers",
            line={"color": PRICE_COLOR, "width": 2},
            marker={"size": 6, "symbol": "circle"},
            hovertemplate="$%{y:.2f}/MWh<extra>Price</extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=frame["carbon"].tolist(),
            name="Carbon intensity (gCO2/kWh)",
            yaxis="y2",
            mode="lines+markers",
            line={"color": CARBON_COLOR, "width": 2, "dash": "dash"},
            marker={"size": 6, "symbol": "diamond"},
            hovertemplate="%{y:.0f} gCO2/kWh<extra>Carbon</extra>",
        )
    )

    if deadline_hours < len(labels):
        # Category axes place label i at position i, so the boundary between the
        # last allowed hour and the first forbidden hour sits at deadline - 0.5.
        figure.add_vline(
            x=deadline_hours - 0.5,
            line={"color": BASELINE_COLOR, "width": 1.5, "dash": "dot"},
            annotation_text="Deadline",
            annotation_position="top",
        )

    tick_positions = labels[::2]
    tick_text = frame["tick"].tolist()[::2]
    figure.update_layout(
        barmode="overlay",
        bargap=0.06,
        height=520,
        hovermode="x unified",
        margin={"l": 60, "r": 70, "t": 70, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.03, "xanchor": "left", "x": 0.0},
        xaxis={
            "title": {"text": "Local time in grid region (hour starting)"},
            "type": "category",
            "categoryorder": "array",
            "categoryarray": labels,
            "tickmode": "array",
            "tickvals": tick_positions,
            "ticktext": tick_text,
            "showgrid": False,
        },
        yaxis={
            "title": {"text": "Electricity price (USD/MWh)"},
            "rangemode": "tozero",
            "showgrid": True,
            "gridcolor": "rgba(128,128,128,0.18)",
            "zeroline": False,
        },
        yaxis2={
            "title": {"text": "Carbon intensity (gCO2/kWh)"},
            "overlaying": "y",
            "side": "right",
            "rangemode": "tozero",
            "tickformat": ".0f",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis3={
            "overlaying": "y",
            "range": [0, 1],
            "visible": False,
            "fixedrange": True,
        },
    )
    return figure


def render_sidebar(regions: list[dict[str, str]]) -> dict[str, Any]:
    st.sidebar.header("Job configuration")
    codes = [region["code"] for region in regions]
    names = {region["code"]: region["name"] for region in regions}
    region = st.sidebar.selectbox(
        "Grid region",
        options=codes,
        format_func=lambda code: f"{code} ({names[code]})",
    )
    required_hours = st.sidebar.slider(
        "Required AI Training Hours", min_value=1, max_value=12, value=6, step=1
    )
    power_kw = st.sidebar.number_input(
        "GPU Cluster Power Rating (kW)",
        min_value=1.0,
        max_value=100_000.0,
        value=500.0,
        step=50.0,
        help="Electrical draw of the cluster while the training job runs.",
    )
    with st.sidebar.expander("Advanced"):
        deadline_hours = st.slider(
            "Deadline (hours from now)",
            min_value=1,
            max_value=24,
            value=24,
            step=1,
            help="The job must finish within this window.",
        )
        carbon_price = st.number_input(
            "Internal carbon price (USD per tonne CO2)",
            min_value=0.0,
            max_value=1_000.0,
            value=0.0,
            step=10.0,
            help="Zero optimizes purely on electricity cost. A positive value also penalizes emissions.",
        )
    st.sidebar.caption(f"API endpoint: {API_URL}")
    return {
        "region": region,
        "required_hours": int(required_hours),
        "power_kw": float(power_kw),
        "deadline_hours": int(deadline_hours),
        "carbon_price": float(carbon_price),
    }


def render_kpis(schedule: dict[str, Any]) -> None:
    metrics = schedule["metrics"]
    required_hours = schedule["required_hours"]
    baseline_carbon = metrics["baseline_carbon_kg"]
    carbon_pct = 100.0 * metrics["carbon_avoided_kg"] / baseline_carbon if baseline_carbon > 0 else 0.0

    first, second, third = st.columns(3)
    first.metric(
        "Total Cost Savings",
        f"{metrics['cost_savings_pct']:.1f}%",
        delta=f"${metrics['cost_savings_usd']:,.2f} vs. running now",
        border=True,
    )
    second.metric(
        "Carbon Avoided",
        f"{metrics['carbon_avoided_kg']:,.1f} kg CO2",
        delta=f"{carbon_pct:.1f}% vs. running now",
        border=True,
    )
    third.metric(
        "Stressed Grid Hours Avoided",
        f"{metrics['stressed_hours_avoided']} h",
        delta=(
            f"{metrics['optimized_stressed_hours']} of {required_hours} scheduled hours "
            "on a stressed grid"
        ),
        delta_color="off",
        delta_arrow="off",
        border=True,
    )

    fourth, fifth, sixth = st.columns(3)
    fourth.metric("Optimized job cost", f"${metrics['optimized_cost_usd']:,.2f}")
    fifth.metric("Baseline job cost", f"${metrics['baseline_cost_usd']:,.2f}")
    sixth.metric("Energy for the job", f"{metrics['energy_kwh']:,.0f} kWh")


def render_live_status(grid: dict[str, Any]) -> None:
    current = grid["current"]
    summary = grid["summary"]
    peak_price_time = datetime.fromisoformat(summary["peak_price_local_timestamp"]).strftime("%a %H:%M")
    st.subheader("Live grid status")
    first, second, third, fourth = st.columns(4)
    first.metric("Current price", f"${current['price_usd_mwh']:.2f}/MWh")
    second.metric("Current carbon intensity", f"{current['carbon_g_kwh']:.0f} gCO2/kWh")
    third.metric("Current grid stress", current["stress_level"].capitalize())
    fourth.metric("Forecast price peak", f"${summary['max_price_usd_mwh']:.2f}", delta=peak_price_time, delta_color="off", delta_arrow="off")


def render_tables(frame: pd.DataFrame) -> None:
    with st.expander("Hourly schedule table"):
        table = pd.DataFrame(
            {
                "Local time": frame["label"],
                "Price (USD/MWh)": frame["price"],
                "Carbon (gCO2/kWh)": frame["carbon"],
                "Stress score": frame["stress_score"],
                "Grid stressed": frame["is_stressed"],
                "Optimized: run": frame["run"].astype(bool),
                "Baseline: run": frame["baseline_run"].astype(bool),
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Recent stored schedules"):
        try:
            history = fetch_history(limit=10)
        except BackendError as exc:
            st.warning(f"Schedule history is unavailable: {exc}")
            return
        if not history["schedules"]:
            st.write("No schedules stored yet.")
            return
        records = pd.DataFrame(history["schedules"])
        columns = [
            "created_at",
            "region",
            "required_hours",
            "power_kw",
            "optimized_cost_usd",
            "baseline_cost_usd",
            "cost_savings_pct",
            "carbon_avoided_kg",
            "stressed_hours_avoided",
            "solver_status",
        ]
        st.dataframe(records[[column for column in columns if column in records.columns]], hide_index=True, width="stretch")
        st.caption(f"Storage backend: {history['storage_backend']}")


def main() -> None:
    st.set_page_config(page_title="VoltEdge", page_icon=":zap:", layout="wide")
    st.title("VoltEdge")
    st.caption("Intelligent Workload Scheduler and Grid Arbitrage Engine")

    try:
        regions = fetch_regions()
    except BackendError as exc:
        st.error(str(exc))
        st.stop()
    if not regions:
        regions = FALLBACK_REGIONS

    config = render_sidebar(regions)
    horizon_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")

    if config["required_hours"] > config["deadline_hours"]:
        st.error(
            f"The job needs {config['required_hours']} compute hours but the deadline allows only "
            f"{config['deadline_hours']}. Extend the deadline or reduce the training hours."
        )
        st.stop()

    try:
        with st.spinner("Optimizing schedule"):
            schedule = fetch_schedule(
                config["region"],
                config["required_hours"],
                config["power_kw"],
                config["deadline_hours"],
                config["carbon_price"],
                horizon_key,
            )
            grid = fetch_grid_metrics(config["region"], horizon_key)
    except BackendError as exc:
        st.error(str(exc))
        st.stop()

    if schedule["source"] == "simulation":
        st.info(
            "Grid data source: built-in simulation. Prices and carbon intensity are synthetic, "
            "not live market data."
        )

    render_kpis(schedule)
    st.caption(
        "Baseline: the job starts now and runs without pausing. "
        "Savings compare the optimized schedule against that baseline."
    )

    frame = build_hourly_frame(schedule)
    st.subheader(f"24-hour schedule for {schedule['region_name']}")
    st.plotly_chart(build_chart(frame, schedule["deadline_hours"]), width="stretch")
    st.caption(
        "Green bars mark the hours when the GPUs run. Red ticks along the bottom mark stressed grid hours. "
        "Price uses the left axis and carbon intensity uses the right axis. "
        "Select the baseline in the legend to compare."
    )

    render_live_status(grid)
    render_tables(frame)

    st.caption(
        f"Solver status: {schedule['solver_status']}. "
        f"Storage backend: {schedule['storage_backend']}. "
        f"Schedule stored: {'yes' if schedule['persisted'] else 'no'}."
    )


main()
