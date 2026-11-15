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

The only data source included is the simulation in `grid_client.py`. Its region
parameters are illustrative, not measured market data. To connect a live feed,
implement the `GridDataProvider` protocol and pass the provider to `GridClient`
in `main.py`.

## Tests

```bash
python -m pytest -q
```
