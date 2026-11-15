-- VoltEdge schema for Supabase (PostgreSQL).
-- Run this once in the Supabase dashboard: SQL Editor, New query, paste, Run.

create table if not exists public.grid_logs (
    id bigint generated always as identity primary key,
    region text not null,
    ts timestamptz not null,
    price_usd_mwh double precision not null,
    carbon_g_kwh double precision not null,
    stress_score double precision not null,
    is_stressed boolean not null,
    source text not null,
    created_at timestamptz not null default now(),
    constraint grid_logs_region_ts_key unique (region, ts)
);

create index if not exists grid_logs_region_ts_idx
    on public.grid_logs (region, ts desc);

create table if not exists public.schedules (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    region text not null,
    horizon_start timestamptz not null,
    required_hours integer not null check (required_hours between 1 and 24),
    deadline_hours integer not null check (deadline_hours between 1 and 24),
    power_kw double precision not null check (power_kw > 0),
    carbon_price_usd_per_tonne double precision not null default 0,
    execution_mask jsonb not null,
    baseline_mask jsonb not null,
    optimized_cost_usd double precision not null,
    baseline_cost_usd double precision not null,
    cost_savings_pct double precision not null,
    carbon_avoided_kg double precision not null,
    stressed_hours_avoided integer not null,
    solver_status text not null,
    data_source text not null
);

create index if not exists schedules_created_at_idx
    on public.schedules (created_at desc);

-- Enable row level security with no policies. The backend connects with the
-- service role key, which bypasses RLS. Browser clients that hold only the
-- anon key can neither read nor write these tables.
alter table public.grid_logs enable row level security;
alter table public.schedules enable row level security;
