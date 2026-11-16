"""Schedule optimization engine for VoltEdge.

Problem statement
-----------------
Given hourly electricity prices ``p_t`` (USD/MWh) and carbon intensities
``c_t`` (gCO2/kWh) for a horizon of T hours, choose binary decisions ``x_t``
(1 = GPUs run during hour t) to

    minimize    sum_t  x_t * hourly_cost_t
    subject to  sum_t  x_t == required_hours
                x_t == 0            for every t >= deadline_hours
                x_t in {0, 1}

where ``hourly_cost_t = power_mw * p_t + carbon_price * power_kw * c_t / 1e6``.
With the default carbon price of zero, the objective equals the total
electricity cost: Sum(Job_Active_t * Price_t) scaled by the cluster power.

The job can pause and resume freely, so it splits into hourly chunks.
PuLP (CBC) solves the model. If CBC is unavailable or fails, an exact greedy
selection takes over. The greedy method is optimal for this specific problem
because the only coupling constraint is the cardinality of the selection.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import pulp

logger = logging.getLogger("voltedge.solver")

MAX_HORIZON_HOURS = 24


class ScheduleInputError(ValueError):
    """Raised when the solver receives malformed inputs."""


class InfeasibleScheduleError(ValueError):
    """Raised when no schedule can satisfy the job constraints."""


@dataclass(frozen=True)
class ScheduleEvaluation:
    """Cost and emissions of running a given binary mask."""

    active_hours: int
    energy_kwh: float
    cost_usd: float
    carbon_kg: float


@dataclass(frozen=True)
class ScheduleResult:
    """Output of the optimizer, including the naive baseline for comparison."""

    mask: list[int]
    baseline_mask: list[int]
    optimized: ScheduleEvaluation
    baseline: ScheduleEvaluation
    objective_value: float
    solver_status: str

    @property
    def cost_savings_usd(self) -> float:
        return self.baseline.cost_usd - self.optimized.cost_usd

    @property
    def cost_savings_pct(self) -> float:
        if abs(self.baseline.cost_usd) < 1e-9:
            return 0.0
        return 100.0 * self.cost_savings_usd / abs(self.baseline.cost_usd)

    @property
    def carbon_avoided_kg(self) -> float:
        return self.baseline.carbon_kg - self.optimized.carbon_kg


def _validate_series(name: str, values: Sequence[float]) -> list[float]:
    try:
        series = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ScheduleInputError(f"The {name} series must contain only numbers.") from exc
    if not series:
        raise ScheduleInputError(f"The {name} series is empty.")
    if any(not math.isfinite(value) for value in series):
        raise ScheduleInputError(f"The {name} series contains NaN or infinite values.")
    return series


def _validate_inputs(
    prices: Sequence[float],
    carbon: Sequence[float],
    required_hours: int,
    power_kw: float,
    deadline_hours: int | None,
    carbon_price_usd_per_tonne: float,
) -> tuple[list[float], list[float], int, int]:
    price_series = _validate_series("price", prices)
    carbon_series = _validate_series("carbon", carbon)
    if len(price_series) != len(carbon_series):
        raise ScheduleInputError(
            f"Price series has {len(price_series)} hours but carbon series has {len(carbon_series)}."
        )
    horizon = len(price_series)
    if horizon > MAX_HORIZON_HOURS:
        raise ScheduleInputError(
            f"The horizon covers {horizon} hours. The maximum is {MAX_HORIZON_HOURS}."
        )

    if isinstance(required_hours, bool) or not isinstance(required_hours, int):
        raise ScheduleInputError("Required compute hours must be a whole number.")
    if required_hours < 1:
        raise ScheduleInputError("Required compute hours must be at least 1.")

    if not isinstance(power_kw, (int, float)) or isinstance(power_kw, bool):
        raise ScheduleInputError("Power rating must be a number.")
    if not math.isfinite(power_kw) or power_kw <= 0:
        raise ScheduleInputError("Power rating must be a positive, finite number of kW.")

    if not math.isfinite(carbon_price_usd_per_tonne) or carbon_price_usd_per_tonne < 0:
        raise ScheduleInputError("Carbon price must be zero or a positive number.")

    deadline = horizon if deadline_hours is None else deadline_hours
    if isinstance(deadline, bool) or not isinstance(deadline, int):
        raise ScheduleInputError("Deadline must be a whole number of hours.")
    if deadline < 1 or deadline > horizon:
        raise ScheduleInputError(
            f"Deadline must fall between 1 and {horizon} hours. Received {deadline}."
        )

    if required_hours > deadline:
        raise InfeasibleScheduleError(
            f"The job needs {required_hours} compute hours but only {deadline} hours remain "
            f"before the deadline. Reduce the required hours or extend the deadline "
            f"(maximum {horizon} hours)."
        )
    return price_series, carbon_series, horizon, deadline


def hourly_costs(
    prices: Sequence[float],
    carbon: Sequence[float],
    power_kw: float,
    carbon_price_usd_per_tonne: float = 0.0,
) -> list[float]:
    """Return the objective coefficient (USD) of running the cluster in each hour."""
    power_mw = power_kw / 1000.0
    costs = []
    for price, intensity in zip(prices, carbon):
        energy_cost = power_mw * price
        # kW * 1 h * g/kWh = grams. Divide by 1e6 for tonnes.
        carbon_cost = carbon_price_usd_per_tonne * power_kw * intensity / 1_000_000.0
        costs.append(energy_cost + carbon_cost)
    return costs


def evaluate_schedule(
    mask: Sequence[int],
    prices: Sequence[float],
    carbon: Sequence[float],
    power_kw: float,
) -> ScheduleEvaluation:
    """Compute energy, electricity cost, and emissions for a binary mask."""
    if not (len(mask) == len(prices) == len(carbon)):
        raise ScheduleInputError("Mask, price, and carbon series must have the same length.")
    active_hours = int(sum(mask))
    cost_usd = sum(flag * price * power_kw / 1000.0 for flag, price in zip(mask, prices))
    carbon_kg = sum(flag * intensity * power_kw / 1000.0 for flag, intensity in zip(mask, carbon))
    return ScheduleEvaluation(
        active_hours=active_hours,
        energy_kwh=active_hours * power_kw,
        cost_usd=cost_usd,
        carbon_kg=carbon_kg,
    )


def build_baseline_mask(required_hours: int, horizon: int) -> list[int]:
    """The naive schedule: start now and run without pausing until the job is done."""
    return [1 if hour < required_hours else 0 for hour in range(horizon)]


def _solve_greedy(costs: Sequence[float], required_hours: int, deadline: int) -> list[int]:
    """Pick the cheapest eligible hours. Ties resolve to the earliest hour."""
    eligible = sorted(range(deadline), key=lambda hour: (costs[hour], hour))
    chosen = set(eligible[:required_hours])
    return [1 if hour in chosen else 0 for hour in range(len(costs))]


def _solve_milp(costs: Sequence[float], required_hours: int, deadline: int) -> list[int]:
    """Solve the binary program with PuLP and CBC. Raise if no optimum is proven."""
    horizon = len(costs)
    problem = pulp.LpProblem("voltedge_training_schedule", pulp.LpMinimize)
    run = [pulp.LpVariable(f"run_{hour}", cat=pulp.LpBinary) for hour in range(horizon)]

    problem += pulp.lpSum(costs[hour] * run[hour] for hour in range(horizon)), "total_cost"
    problem += pulp.lpSum(run) == required_hours, "required_compute_hours"
    for hour in range(deadline, horizon):
        problem += run[hour] == 0, f"after_deadline_{hour}"

    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"CBC finished with status '{status}'.")

    mask = []
    for variable in run:
        value = variable.value()
        mask.append(1 if value is not None and value > 0.5 else 0)
    if sum(mask) != required_hours or any(mask[hour] for hour in range(deadline, horizon)):
        raise RuntimeError("CBC returned a mask that violates the constraints.")
    return mask


def optimize_schedule(
    prices: Sequence[float],
    carbon: Sequence[float],
    required_hours: int,
    power_kw: float,
    deadline_hours: int | None = None,
    carbon_price_usd_per_tonne: float = 0.0,
) -> ScheduleResult:
    """Find the cheapest set of hours in which to run the training job.

    Args:
        prices: Hourly LMP in USD/MWh. Index 0 is the current hour.
        carbon: Hourly carbon intensity in gCO2/kWh, aligned with ``prices``.
        required_hours: Exact number of compute hours the job needs.
        power_kw: Electrical draw of the GPU cluster while it runs.
        deadline_hours: The job must finish within this many hours from now.
            Defaults to the full horizon.
        carbon_price_usd_per_tonne: Optional internal carbon price. Zero keeps
            the objective purely on electricity cost.

    Raises:
        ScheduleInputError: The inputs are malformed.
        InfeasibleScheduleError: The job cannot fit before the deadline.
    """
    price_series, carbon_series, horizon, deadline = _validate_inputs(
        prices, carbon, required_hours, power_kw, deadline_hours, carbon_price_usd_per_tonne
    )
    costs = hourly_costs(price_series, carbon_series, power_kw, carbon_price_usd_per_tonne)

    try:
        mask = _solve_milp(costs, required_hours, deadline)
        solver_status = "optimal"
    except Exception as exc:
        logger.warning("MILP solve failed (%s). Using the exact greedy solver.", exc)
        mask = _solve_greedy(costs, required_hours, deadline)
        solver_status = "optimal_greedy_fallback"

    baseline_mask = build_baseline_mask(required_hours, horizon)
    return ScheduleResult(
        mask=mask,
        baseline_mask=baseline_mask,
        optimized=evaluate_schedule(mask, price_series, carbon_series, power_kw),
        baseline=evaluate_schedule(baseline_mask, price_series, carbon_series, power_kw),
        objective_value=sum(flag * cost for flag, cost in zip(mask, costs)),
        solver_status=solver_status,
    )
