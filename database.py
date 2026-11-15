"""Persistence layer for VoltEdge.

The module exposes one factory, ``get_database()``, which returns a ready-to-use
storage backend:

* ``SupabaseDatabase`` when ``SUPABASE_URL`` and ``SUPABASE_KEY`` are set and the
  project answers a connectivity probe.
* ``SQLiteDatabase`` (a local file, or an in-memory database as a last resort)
  in every other case, so the application never crashes on initialization.

Both backends implement the same interface and store two record types:
time-series grid logs and optimized schedules.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("voltedge.database")

GRID_LOGS_TABLE = "grid_logs"
SCHEDULES_TABLE = "schedules"

GRID_LOG_COLUMNS: tuple[str, ...] = (
    "region",
    "ts",
    "price_usd_mwh",
    "carbon_g_kwh",
    "stress_score",
    "is_stressed",
    "source",
)

SCHEDULE_COLUMNS: tuple[str, ...] = (
    "region",
    "horizon_start",
    "required_hours",
    "deadline_hours",
    "power_kw",
    "carbon_price_usd_per_tonne",
    "execution_mask",
    "baseline_mask",
    "optimized_cost_usd",
    "baseline_cost_usd",
    "cost_savings_pct",
    "carbon_avoided_kg",
    "stressed_hours_avoided",
    "solver_status",
    "data_source",
)


class DatabaseError(RuntimeError):
    """Raised when a storage backend cannot complete a read or write."""


class Database(ABC):
    """Common interface implemented by every storage backend."""

    backend_name: str = "unknown"

    @abstractmethod
    def upsert_grid_logs(self, rows: list[dict[str, Any]]) -> int:
        """Insert or update grid observations keyed by (region, ts). Return the row count."""

    @abstractmethod
    def insert_schedule(self, row: dict[str, Any]) -> dict[str, Any]:
        """Store one optimized schedule and return the stored record."""

    @abstractmethod
    def fetch_grid_logs(self, region: str, limit: int = 48) -> list[dict[str, Any]]:
        """Return the most recent grid logs for a region, newest first."""

    @abstractmethod
    def fetch_recent_schedules(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recently stored schedules, newest first."""

    @abstractmethod
    def ping(self) -> bool:
        """Return True when the backend answers a trivial query."""


def _project(row: dict[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    """Keep only known columns and fail loudly when a required one is missing."""
    missing = [column for column in columns if column not in row]
    if missing:
        raise DatabaseError(f"Record is missing required fields: {', '.join(missing)}")
    return {column: row[column] for column in columns}


class SupabaseDatabase(Database):
    """Supabase (PostgreSQL) backend built on the official supabase-py client."""

    backend_name = "supabase"

    def __init__(self, url: str, key: str) -> None:
        from supabase import create_client

        self._client = create_client(url, key)
        # create_client performs no network I/O, so probe both tables now. A bad
        # key, an unreachable project, or a missing table raises here and lets
        # the factory fall back to the local database.
        self._client.table(GRID_LOGS_TABLE).select("id").limit(1).execute()
        self._client.table(SCHEDULES_TABLE).select("id").limit(1).execute()

    def upsert_grid_logs(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        payload = [_project(row, GRID_LOG_COLUMNS) for row in rows]
        try:
            response = (
                self._client.table(GRID_LOGS_TABLE)
                .upsert(payload, on_conflict="region,ts")
                .execute()
            )
        except Exception as exc:
            raise DatabaseError(f"Supabase grid log upsert failed: {exc}") from exc
        return len(response.data or [])

    def insert_schedule(self, row: dict[str, Any]) -> dict[str, Any]:
        payload = _project(row, SCHEDULE_COLUMNS)
        try:
            response = self._client.table(SCHEDULES_TABLE).insert(payload).execute()
        except Exception as exc:
            raise DatabaseError(f"Supabase schedule insert failed: {exc}") from exc
        if not response.data:
            raise DatabaseError("Supabase schedule insert returned no rows.")
        return dict(response.data[0])

    def fetch_grid_logs(self, region: str, limit: int = 48) -> list[dict[str, Any]]:
        try:
            response = (
                self._client.table(GRID_LOGS_TABLE)
                .select("*")
                .eq("region", region)
                .order("ts", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception as exc:
            raise DatabaseError(f"Supabase grid log query failed: {exc}") from exc
        return [dict(item) for item in (response.data or [])]

    def fetch_recent_schedules(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            response = (
                self._client.table(SCHEDULES_TABLE)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception as exc:
            raise DatabaseError(f"Supabase schedule query failed: {exc}") from exc
        return [dict(item) for item in (response.data or [])]

    def ping(self) -> bool:
        try:
            self._client.table(SCHEDULES_TABLE).select("id").limit(1).execute()
        except Exception:
            return False
        return True


class SQLiteDatabase(Database):
    """Local fallback backend. Mirrors the Supabase schema in a SQLite database."""

    backend_name = "sqlite"

    def __init__(self, path: str = "voltedge_local.db") -> None:
        self.path = path
        self._lock = threading.Lock()
        # FastAPI runs sync endpoints in a thread pool, so share one connection
        # across threads and serialize access with the lock.
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._create_tables()
        if path == ":memory:":
            self.backend_name = "sqlite-memory"

    def _create_tables(self) -> None:
        with self._lock:
            self._connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {GRID_LOGS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    price_usd_mwh REAL NOT NULL,
                    carbon_g_kwh REAL NOT NULL,
                    stress_score REAL NOT NULL,
                    is_stressed INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (region, ts)
                );
                CREATE TABLE IF NOT EXISTS {SCHEDULES_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    region TEXT NOT NULL,
                    horizon_start TEXT NOT NULL,
                    required_hours INTEGER NOT NULL,
                    deadline_hours INTEGER NOT NULL,
                    power_kw REAL NOT NULL,
                    carbon_price_usd_per_tonne REAL NOT NULL,
                    execution_mask TEXT NOT NULL,
                    baseline_mask TEXT NOT NULL,
                    optimized_cost_usd REAL NOT NULL,
                    baseline_cost_usd REAL NOT NULL,
                    cost_savings_pct REAL NOT NULL,
                    carbon_avoided_kg REAL NOT NULL,
                    stressed_hours_avoided INTEGER NOT NULL,
                    solver_status TEXT NOT NULL,
                    data_source TEXT NOT NULL
                );
                """
            )
            self._connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _decode_schedule(record: sqlite3.Row) -> dict[str, Any]:
        row = dict(record)
        row["execution_mask"] = json.loads(row["execution_mask"])
        row["baseline_mask"] = json.loads(row["baseline_mask"])
        return row

    @staticmethod
    def _decode_grid_log(record: sqlite3.Row) -> dict[str, Any]:
        row = dict(record)
        row["is_stressed"] = bool(row["is_stressed"])
        return row

    def upsert_grid_logs(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        created_at = self._now()
        values = []
        for row in rows:
            clean = _project(row, GRID_LOG_COLUMNS)
            values.append(
                (
                    clean["region"],
                    clean["ts"],
                    float(clean["price_usd_mwh"]),
                    float(clean["carbon_g_kwh"]),
                    float(clean["stress_score"]),
                    1 if clean["is_stressed"] else 0,
                    clean["source"],
                    created_at,
                )
            )
        statement = f"""
            INSERT INTO {GRID_LOGS_TABLE}
                (region, ts, price_usd_mwh, carbon_g_kwh, stress_score, is_stressed, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (region, ts) DO UPDATE SET
                price_usd_mwh = excluded.price_usd_mwh,
                carbon_g_kwh = excluded.carbon_g_kwh,
                stress_score = excluded.stress_score,
                is_stressed = excluded.is_stressed,
                source = excluded.source
        """
        try:
            with self._lock:
                self._connection.executemany(statement, values)
                self._connection.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQLite grid log upsert failed: {exc}") from exc
        return len(values)

    def insert_schedule(self, row: dict[str, Any]) -> dict[str, Any]:
        clean = _project(row, SCHEDULE_COLUMNS)
        created_at = self._now()
        values = (
            created_at,
            clean["region"],
            clean["horizon_start"],
            int(clean["required_hours"]),
            int(clean["deadline_hours"]),
            float(clean["power_kw"]),
            float(clean["carbon_price_usd_per_tonne"]),
            json.dumps(list(clean["execution_mask"])),
            json.dumps(list(clean["baseline_mask"])),
            float(clean["optimized_cost_usd"]),
            float(clean["baseline_cost_usd"]),
            float(clean["cost_savings_pct"]),
            float(clean["carbon_avoided_kg"]),
            int(clean["stressed_hours_avoided"]),
            clean["solver_status"],
            clean["data_source"],
        )
        statement = f"""
            INSERT INTO {SCHEDULES_TABLE}
                (created_at, region, horizon_start, required_hours, deadline_hours, power_kw,
                 carbon_price_usd_per_tonne, execution_mask, baseline_mask, optimized_cost_usd,
                 baseline_cost_usd, cost_savings_pct, carbon_avoided_kg, stressed_hours_avoided,
                 solver_status, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._lock:
                cursor = self._connection.execute(statement, values)
                self._connection.commit()
                record = self._connection.execute(
                    f"SELECT * FROM {SCHEDULES_TABLE} WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQLite schedule insert failed: {exc}") from exc
        return self._decode_schedule(record)

    def fetch_grid_logs(self, region: str, limit: int = 48) -> list[dict[str, Any]]:
        try:
            with self._lock:
                records = self._connection.execute(
                    f"SELECT * FROM {GRID_LOGS_TABLE} WHERE region = ? ORDER BY ts DESC LIMIT ?",
                    (region, int(limit)),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQLite grid log query failed: {exc}") from exc
        return [self._decode_grid_log(record) for record in records]

    def fetch_recent_schedules(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            with self._lock:
                records = self._connection.execute(
                    f"SELECT * FROM {SCHEDULES_TABLE} ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQLite schedule query failed: {exc}") from exc
        return [self._decode_schedule(record) for record in records]

    def ping(self) -> bool:
        try:
            with self._lock:
                self._connection.execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
        return True


def _build_local_database() -> Database:
    """Open the SQLite file, or an in-memory database if the file is unusable."""
    path = os.getenv("VOLTEDGE_SQLITE_PATH", "voltedge_local.db")
    try:
        return SQLiteDatabase(path)
    except (sqlite3.Error, OSError) as exc:
        logger.warning("SQLite file '%s' is unusable (%s). Using an in-memory database.", path, exc)
        return SQLiteDatabase(":memory:")


def build_database() -> Database:
    """Create the best available backend. This function never raises."""
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    if url and key:
        try:
            database = SupabaseDatabase(url, key)
            logger.info("Connected to Supabase.")
            return database
        except Exception as exc:
            logger.warning(
                "Supabase is configured but unavailable (%s). Falling back to local SQLite.", exc
            )
    else:
        logger.info("SUPABASE_URL or SUPABASE_KEY is not set. Using local SQLite.")
    return _build_local_database()


_database: Database | None = None
_database_lock = threading.Lock()


def get_database() -> Database:
    """Return the process-wide database instance, creating it on first use."""
    global _database
    if _database is None:
        with _database_lock:
            if _database is None:
                _database = build_database()
    return _database


def reset_database() -> None:
    """Drop the cached instance so the next call to get_database() rebuilds it."""
    global _database
    with _database_lock:
        _database = None
