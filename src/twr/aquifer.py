"""Aquifer storage accounting.

Antecedent storage capacity is half of the capture question: a perfectly timed
flood is worthless if there is nowhere to put the water. This module tracks a
single lumped storage bucket per site so the pipeline can report headroom
(AF still available) alongside the hydrologic forecast.

The bucket is intentionally simple. A real deployment would swap it for the
district's groundwater model, keeping the same interface: given dates, give me
storage_af and headroom_af.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Site


def _distribute_annual(weights: np.ndarray, annual_af: float, n_days: int) -> np.ndarray:
    """Spread an annual volume across days in proportion to ``weights``."""
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones(n_days)
        total = float(n_days)
    return annual_af * weights / total * (n_days / 365.25)


def seasonal_demand_af_per_day(dates: pd.DatetimeIndex, annual_af: float) -> np.ndarray:
    """Recovery (pumping) demand, peaking in late summer."""
    doy = dates.dayofyear.to_numpy().astype(float)
    shape = np.clip(1.0 + 0.85 * np.sin(2 * np.pi * (doy - 130) / 365.25), 0.15, None)
    return _distribute_annual(shape, annual_af, len(dates))


def simulate_storage(
    dates: pd.DatetimeIndex,
    site: Site,
    precip_mm: np.ndarray,
    annual_demand_af: float | None = None,
    annual_natural_recharge_af: float | None = None,
    captured_af_per_day: np.ndarray | None = None,
    leakage_fraction: float = 0.00008,
) -> pd.DataFrame:
    """Roll a lumped storage balance forward.

    Both fluxes are specified as annual volumes and then distributed across days,
    demand by a summer-peaked seasonal shape and natural recharge in proportion to
    30-day antecedent rainfall. Expressing them the same way matters: an earlier
    version scaled recharge by an arbitrary constant instead of by capacity, so
    demand outran recharge at every site, the bucket drained to empty within the
    first year and stayed there, headroom was always maximal, and the storage
    constraint could never bind. A flat line at zero is not a storage model.

    Defaults give recharge slightly above demand, so the bucket runs full-ish with
    a seasonal summer drawdown, and leakage stabilises it below capacity.

    Args:
        precip_mm: daily precipitation used as the natural-recharge driver.
        captured_af_per_day: managed recharge actually delivered. ``None`` means
            the baseline "no MAR" trajectory.
    """
    capacity = site.infrastructure.storage_capacity_af
    if capacity <= 0:
        raise ValueError("storage_capacity_af must be positive")

    n = len(dates)
    if annual_demand_af is None:
        annual_demand_af = 0.18 * capacity
    if annual_natural_recharge_af is None:
        annual_natural_recharge_af = 0.22 * capacity
    demand = seasonal_demand_af_per_day(dates, annual_demand_af)

    # Natural recharge responds to rainfall with a long memory (30-day mean).
    smoothed = pd.Series(precip_mm).rolling(30, min_periods=1).mean().to_numpy()
    natural = _distribute_annual(smoothed, annual_natural_recharge_af, n)

    managed = np.zeros(n) if captured_af_per_day is None else np.asarray(captured_af_per_day, float)
    if len(managed) != n:
        raise ValueError("captured_af_per_day must align with dates")

    storage = np.zeros(n)
    level = site.storage_initial_fraction * capacity
    for i in range(n):
        level += natural[i] + managed[i]
        level -= demand[i]
        level -= leakage_fraction * level
        level = float(np.clip(level, 0.0, capacity))
        storage[i] = level

    return pd.DataFrame(
        {
            "date": dates,
            "site_id": site.id,
            "storage_af": storage,
            "headroom_af": capacity - storage,
            "storage_fraction": storage / capacity,
            "natural_recharge_af": natural,
            "recovery_af": demand,
            "managed_recharge_af": managed,
        }
    )
