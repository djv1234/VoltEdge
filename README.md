# VoltEdge

Intelligent Workload Scheduler and Grid Arbitrage Engine. VoltEdge reads a
24-hour forecast of electricity prices (LMP) and carbon intensity for a grid
region, then uses mixed-integer programming to pick the cheapest hours in which
to run an AI training job.

## Modules

| File | Purpose |
|------|---------|
| `database.py` | Supabase client with automatic fallback to local SQLite |
| `grid_client.py` | Grid data client and the built-in diurnal simulation engine |
| `live_provider.py` | Live CAISO prices and EIA-based carbon intensity |
| `solver.py` | PuLP (CBC) binary program, with an exact greedy fallback |
| `main.py` | FastAPI backend |
| `app.py` | Streamlit dashboard |
| `supabase_schema.sql` | Tables for Supabase |
| `tests/test_voltedge.py` | Test suite |

## Quick start

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Start the API in one terminal:

```bash
uvicorn main:app --port 8000
```

Start the dashboard in a second terminal:

```bash
streamlit run app.py
```

Open http://localhost:8501. With no configuration, the app uses simulated grid
data and stores records in `voltedge_local.db` (SQLite).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /grid/metrics?region=CAISO` | Current hour plus 24-hour forecast of price, carbon, and stress |
| `POST /schedule/optimize` | Returns the binary execution mask and savings metrics |
| `GET /schedule/history` | Recently stored schedules |
| `GET /regions` | Supported region codes |
| `GET /health` | Storage backend and grid data source in use |

Example request:

```bash
curl -X POST http://localhost:8000/schedule/optimize -H "content-type: application/json" -d '{"required_hours": 6, "power_kw": 500}'
```

Interactive API docs: http://localhost:8000/docs

## Optimization model

For each hour `t` in the 24-hour horizon, the binary variable `x_t` states
whether the GPUs run.

* Objective: minimize `sum(x_t * power_MW * price_t)`. An optional internal
  carbon price adds `carbon_price * power_kW * carbon_t / 1e6` per hour.
* Constraint: `sum(x_t) == required_hours`.
* Constraint: `x_t == 0` for every hour at or after the deadline.

A stressed grid hour is one of the 5 hours in the 24-hour window with the
highest stress score, where the score blends normalized price (60%) and
normalized carbon intensity (40%). `VOLTEDGE_STRESSED_HOURS` accepts 4, 5, or 6.

The savings metrics compare the optimized schedule against a baseline that
starts the job immediately and runs it without pausing.

## Supabase

1. Create a Supabase project.
2. Run `supabase_schema.sql` in the SQL Editor.
3. Copy `.env.example` to `.env`, then set `SUPABASE_URL` and `SUPABASE_KEY`
   (the secret, server-side key).
4. Restart the API. `GET /health` reports `"storage_backend": "supabase"`.

If Supabase is unreachable or the tables are missing, the API logs a warning
and uses SQLite.

## Grid data

VoltEdge has two data sources and picks one at startup.

**Simulation (default).** Synthetic diurnal curves for CAISO, ERCOT, PJM, and
NYISO. The region parameters are illustrative, not measured market data.

**Live data for CAISO.** Enabled when `EIA_API_KEY` is set in `.env`.

| Signal | Source | Key |
|--------|--------|-----|
| Price | CAISO day-ahead hourly LMP at one trading hub, read through the open-source `gridstatus` library | None |
| Carbon intensity | EIA Open Data API v2, hourly generation by fuel type for the CISO balancing authority | Free key from https://www.eia.gov/opendata/register.php |

Limits of the live data:

* CAISO publishes the next day's prices in the afternoon. Until then, hours
  past local midnight reuse the price of the same hour from the latest
  published day and carry `price_estimated: true`.
* EIA publishes the fuel mix several hours after the fact and offers no
  forecast. Carbon intensity for each hour reuses the fuel mix of the same hour
  of day from the latest published day and carries `carbon_estimated: true`.
* Carbon intensity covers generation inside the balancing authority and ignores
  imports. The emission factors in `live_provider.py` are approximations.
* Other regions keep using the simulation, and the API says so in
  `source_notice`.
* If either feed fails, the API falls back to the simulation and reports the
  reason in `source_notice`. Responses are cached for 15 minutes.

Set `VOLTEDGE_GRID_SOURCE=simulation` to force the simulation. Set
`VOLTEDGE_CAISO_NODE` to price a different node (default `TH_SP15_GEN-APND`).

To add another live feed, implement the `GridDataProvider` protocol in
`grid_client.py`.

## Tests

```bash
python -m pytest -q
```
