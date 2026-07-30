"""High-magnitude flow (HMF) identification.

An HMF is defined here as flow above a basin-specific high-flow threshold, taken
as a percentile of the daily record. The quantity that matters operationally is
not the peak but the *excess volume*: how much water passes above the threshold,
because only that part is a candidate for capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .units import CFS_DAY_TO_AF


def hmf_threshold(flow_cfs: np.ndarray | pd.Series, percentile: float) -> float:
    """Threshold flow for a basin. Computed on positive flows only."""
    values = np.asarray(flow_cfs, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("no finite positive flow values")
    if not 50.0 <= percentile < 100.0:
        raise ValueError("percentile should be a high-flow percentile in [50, 100)")
    return float(np.percentile(values, percentile))


def excess_volume_af(flow_cfs: np.ndarray | pd.Series, threshold_cfs: float) -> np.ndarray:
    """Daily volume above the threshold, in acre-feet."""
    flow = np.asarray(flow_cfs, dtype=float)
    excess_cfs = np.clip(flow - threshold_cfs, 0.0, None)
    return excess_cfs * CFS_DAY_TO_AF


def identify_events(
    dates: pd.Series | pd.DatetimeIndex,
    flow_cfs: np.ndarray | pd.Series,
    threshold_cfs: float,
    min_duration_days: int = 1,
    merge_gap_days: int = 2,
) -> pd.DataFrame:
    """Group consecutive above-threshold days into discrete HMF events.

    Short recessions below the threshold (``merge_gap_days``) are treated as part
    of the same event, which is how an operator would see a single multi-peak
    flood rather than three separate ones.
    """
    dates = pd.DatetimeIndex(pd.Series(dates).to_numpy())
    flow = np.asarray(flow_cfs, dtype=float)
    if len(dates) != len(flow):
        raise ValueError("dates and flow_cfs must be the same length")

    above = flow > threshold_cfs
    if not above.any():
        return pd.DataFrame(
            columns=["start", "end", "duration_days", "peak_cfs", "excess_af", "total_af"]
        )

    idx = np.flatnonzero(above)
    breaks = np.flatnonzero(np.diff(idx) > merge_gap_days + 1)
    groups = np.split(idx, breaks + 1)

    daily_excess = excess_volume_af(flow, threshold_cfs)
    rows = []
    for group in groups:
        lo, hi = int(group[0]), int(group[-1])
        span = slice(lo, hi + 1)
        duration = hi - lo + 1
        if duration < min_duration_days:
            continue
        rows.append(
            {
                "start": dates[lo],
                "end": dates[hi],
                "duration_days": duration,
                "peak_cfs": float(flow[span].max()),
                "excess_af": float(daily_excess[span].sum()),
                "total_af": float(flow[span].sum() * CFS_DAY_TO_AF),
            }
        )
    return pd.DataFrame(rows)


def event_summary(events: pd.DataFrame, years: float) -> dict[str, float]:
    """Descriptive statistics an operator would want on the retrospective record."""
    if events.empty or years <= 0:
        return {
            "n_events": 0,
            "events_per_year": 0.0,
            "median_excess_af": 0.0,
            "p90_excess_af": 0.0,
            "mean_duration_days": 0.0,
            "annual_excess_af": 0.0,
        }
    return {
        "n_events": int(len(events)),
        "events_per_year": float(len(events) / years),
        "median_excess_af": float(events["excess_af"].median()),
        "p90_excess_af": float(events["excess_af"].quantile(0.90)),
        "mean_duration_days": float(events["duration_days"].mean()),
        "annual_excess_af": float(events["excess_af"].sum() / years),
    }
