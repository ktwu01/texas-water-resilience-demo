"""Feature and target construction.

Two rules are enforced here, because breaking either one produces a model that
looks excellent and is useless:

1. **No temporal leakage.** Every feature at time t is a function of sensor data
   at times <= t only. Every target is a function of times t+1 .. t+horizon.
2. **No climatological leakage.** Percentiles and anomalies are referenced to a
   fixed historical baseline period, not to the full record. An operator in 2026
   does not know the 2027 distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hmf import excess_volume_af, hmf_threshold

FEATURE_COLUMNS = [
    "precip_1d",
    "precip_3d",
    "precip_7d",
    "precip_14d",
    "precip_30d",
    "api_decay",
    "soil_moisture",
    "soil_moisture_pct",
    "soil_moisture_delta_7d",
    "storage_deficit_index",
    "et_7d",
    "et_anomaly",
    "extent_anomaly_log",
    "flow_cfs",
    "flow_7d_mean",
    "flow_ratio_threshold",
    "doy_sin",
    "doy_cos",
]

META_COLUMNS = ["date", "basin_id", "hmf_threshold_cfs"]
TARGET_VOLUME = "y_excess_af"
TARGET_EVENT = "y_event"


@dataclass
class FeatureSpec:
    horizon_days: int = 7
    baseline_years: int = 5
    api_decay: float = 0.92
    climatology_window_days: int = 15
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if self.baseline_years < 1:
            raise ValueError("baseline_years must be >= 1")


def _antecedent_precip_index(precip: np.ndarray, decay: float) -> np.ndarray:
    """Exponentially weighted antecedent precipitation index (causal)."""
    out = np.zeros_like(precip, dtype=float)
    running = 0.0
    for i, value in enumerate(precip):
        running = running * decay + (value if np.isfinite(value) else 0.0)
        out[i] = running
    return out


def _doy_climatology(
    frame: pd.DataFrame, column: str, baseline: pd.DataFrame, window: int
) -> pd.Series:
    """Day-of-year mean of ``column`` from the baseline period, +/- window days."""
    doy = frame["date"].dt.dayofyear.to_numpy()
    base_doy = baseline["date"].dt.dayofyear.to_numpy()
    base_values = baseline[column].to_numpy(dtype=float)

    means = np.full(367, np.nan)
    for target in range(1, 367):
        # Circular distance in day-of-year space.
        delta = np.abs(base_doy - target)
        delta = np.minimum(delta, 365 - delta)
        mask = (delta <= window) & np.isfinite(base_values)
        if mask.any():
            means[target] = base_values[mask].mean()
    filled = pd.Series(means).interpolate(limit_direction="both").to_numpy()
    return pd.Series(filled[doy], index=frame.index)


def _doy_percentile(
    frame: pd.DataFrame, column: str, baseline: pd.DataFrame, window: int
) -> pd.Series:
    """Empirical percentile of ``column`` against its baseline day-of-year window."""
    doy = frame["date"].dt.dayofyear.to_numpy()
    values = frame[column].to_numpy(dtype=float)
    base_doy = baseline["date"].dt.dayofyear.to_numpy()
    base_values = baseline[column].to_numpy(dtype=float)

    out = np.full(len(frame), np.nan)
    cache: dict[int, np.ndarray] = {}
    for i, (day, value) in enumerate(zip(doy, values, strict=True)):
        if not np.isfinite(value):
            continue
        if day not in cache:
            delta = np.abs(base_doy - day)
            delta = np.minimum(delta, 365 - delta)
            sample = base_values[(delta <= window) & np.isfinite(base_values)]
            cache[day] = np.sort(sample)
        sample = cache[day]
        if sample.size:
            out[i] = float(np.searchsorted(sample, value, side="right") / len(sample))
    return pd.Series(out, index=frame.index)


def build_basin_features(
    frame: pd.DataFrame, hmf_percentile: float, spec: FeatureSpec | None = None
) -> pd.DataFrame:
    """Build features and targets for a single basin's daily record."""
    spec = spec or FeatureSpec()
    df = frame.sort_values("date").reset_index(drop=True).copy()
    if df["date"].duplicated().any():
        raise ValueError("duplicate dates in basin record")

    baseline_end = df["date"].iloc[0] + pd.DateOffset(years=spec.baseline_years)
    baseline = df[df["date"] < baseline_end]
    if len(baseline) < 365:
        raise ValueError("baseline period needs at least one year of data")

    # Threshold from the baseline period only, so it is knowable in real time.
    threshold = hmf_threshold(baseline["flow_cfs"], hmf_percentile)
    df["hmf_threshold_cfs"] = threshold

    precip = df["precip_mm"].to_numpy(dtype=float)
    df["precip_1d"] = precip
    for window in (3, 7, 14, 30):
        df[f"precip_{window}d"] = (
            df["precip_mm"].rolling(window, min_periods=1).sum().to_numpy()
        )
    df["api_decay"] = _antecedent_precip_index(precip, spec.api_decay)

    df["soil_moisture_delta_7d"] = df["soil_moisture"] - df["soil_moisture"].shift(7)
    df["soil_moisture_pct"] = _doy_percentile(
        df, "soil_moisture", baseline, spec.climatology_window_days
    )
    # Antecedent storage capacity proxy: how far below its baseline maximum the
    # near-surface store sits. High deficit means the soil column can accept
    # water; low deficit means the next storm mostly runs off.
    sm_max = float(baseline["soil_moisture"].quantile(0.98))
    sm_min = float(baseline["soil_moisture"].quantile(0.02))
    df["storage_deficit_index"] = np.clip(
        (sm_max - df["soil_moisture"]) / max(sm_max - sm_min, 1e-6), 0.0, 1.0
    )

    # ECOSTRESS has real gaps; fill causally (forward only) so no future value
    # ever leaks backwards.
    et_filled = df["et_mm"].ffill()
    df["et_7d"] = et_filled.rolling(7, min_periods=1).mean()
    et_clim = _doy_climatology(df, "et_mm", baseline, spec.climatology_window_days)
    df["et_anomaly"] = df["et_7d"] - et_clim

    extent_clim = _doy_climatology(df, "water_extent_km2", baseline, spec.climatology_window_days)
    df["extent_anomaly_log"] = np.log(
        np.clip(df["water_extent_km2"], 1e-6, None) / np.clip(extent_clim, 1e-6, None)
    )

    df["flow_7d_mean"] = df["flow_cfs"].rolling(7, min_periods=1).mean()
    df["flow_ratio_threshold"] = df["flow_cfs"] / threshold

    doy = df["date"].dt.dayofyear.to_numpy().astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # --- targets: strictly forward-looking --------------------------------
    daily_excess = excess_volume_af(df["flow_cfs"], threshold)
    excess_series = pd.Series(daily_excess, index=df.index)
    horizon = spec.horizon_days
    # Sum over t+1 .. t+horizon: reverse-roll then shift back by one day.
    forward_sum = excess_series[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]
    df[TARGET_VOLUME] = forward_sum.shift(-1)
    forward_max = df["flow_cfs"][::-1].rolling(horizon, min_periods=horizon).max()[::-1]
    df[TARGET_EVENT] = (forward_max.shift(-1) > threshold).astype("float")
    df.loc[df[TARGET_VOLUME].isna(), TARGET_EVENT] = np.nan

    df["daily_excess_af"] = daily_excess
    return df


def build_features(
    records: pd.DataFrame, basin_percentiles: dict[str, float], spec: FeatureSpec | None = None
) -> pd.DataFrame:
    """Build the modelling table for every basin in ``records``."""
    spec = spec or FeatureSpec()
    frames = []
    for basin_id, group in records.groupby("basin_id", sort=True):
        if basin_id not in basin_percentiles:
            raise KeyError(f"no HMF percentile configured for basin {basin_id!r}")
        frames.append(build_basin_features(group, basin_percentiles[basin_id], spec))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["basin_id", "date"]).reset_index(drop=True)


def training_matrix(
    table: pd.DataFrame, spec: FeatureSpec | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Drop warm-up and tail rows, then split into X, y_volume, y_event, groups."""
    spec = spec or FeatureSpec()
    needed = spec.feature_columns + [TARGET_VOLUME, TARGET_EVENT]
    clean = table.dropna(subset=needed).reset_index(drop=True)
    if clean.empty:
        raise ValueError("no complete rows after dropping NaNs")
    X = clean[spec.feature_columns].astype(float)
    y_volume = clean[TARGET_VOLUME].astype(float)
    y_event = clean[TARGET_EVENT].astype(int)
    groups = clean["basin_id"]
    return X, y_volume, y_event, groups
