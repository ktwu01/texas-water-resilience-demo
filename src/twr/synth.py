"""Synthetic stand-ins for the multi-sensor record the real project would fuse.

The demo needs data that (a) nobody can mistake for an observation and (b) has
genuine physical structure, so that a model trained on it shows real skill
rather than fitting noise. So we run a small conceptual rainfall-runoff model
and then degrade its internal states into "sensor" variables:

    precip_mm        <- IMERG / GPM proxy       (multiplicative retrieval error)
    soil_moisture    <- SMAP L3 proxy           (top-layer wetness + noise)
    et_mm            <- ECOSTRESS proxy         (actual ET + noise, gappy)
    water_extent_km2 <- SWOT proxy              (hydraulic-geometry function of Q)
    flow_cfs         <- stream gauge proxy      (near-truth, small noise)

Only the sensor columns are ever handed to the learning code. The internal
states (soil store, groundwater store) stay hidden, which is the honest version
of the real problem: the thing you want to know is not directly measured.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Basin
from .units import MM_DAY_KM2_TO_CFS

SENSOR_COLUMNS = [
    "precip_mm",
    "soil_moisture",
    "et_mm",
    "water_extent_km2",
    "flow_cfs",
]


@dataclass(frozen=True)
class SynthParams:
    """Conceptual model parameters. Not calibrated to anything."""

    soil_capacity_mm: float = 260.0
    runoff_exponent: float = 2.2
    baseflow_coefficient: float = 0.014
    percolation_fraction: float = 0.22
    quickflow_recession: float = 0.55
    pet_mean_mm: float = 3.6
    pet_amplitude_mm: float = 2.6
    imerg_log_error_sd: float = 0.22
    smap_noise: float = 0.012
    ecostress_noise_mm: float = 0.35
    ecostress_gap_fraction: float = 0.28
    gauge_log_error_sd: float = 0.05


def basin_seed(basin_id: str, seed: int) -> int:
    """Derive a per-basin RNG seed that is stable across processes.

    This uses CRC32 rather than the builtin ``hash()``. ``hash()`` on strings is
    salted per interpreter by PYTHONHASHSEED, so seeding from it made the whole
    "synthetic data is reproducible from a seed" claim false: every fresh process
    generated a different record, and results moved between runs for no visible
    reason. An in-process reproducibility test cannot catch that, because the salt
    is fixed within one process.
    """
    return (zlib.crc32(basin_id.encode("utf-8")) + int(seed)) % (2**32)


def _seasonal_storm_probability(doy: np.ndarray) -> np.ndarray:
    """Bimodal Texas wet season: a May-June peak and a weaker September peak."""
    spring = np.exp(-0.5 * ((doy - 150) / 45.0) ** 2)
    fall = np.exp(-0.5 * ((doy - 262) / 35.0) ** 2)
    return 0.06 + 0.20 * spring + 0.13 * fall


def _pet_mm(doy: np.ndarray, params: SynthParams, aridity: float) -> np.ndarray:
    seasonal = params.pet_mean_mm + params.pet_amplitude_mm * np.sin(
        2 * np.pi * (doy - 110) / 365.25
    )
    return np.clip(seasonal * aridity, 0.4, None)


def simulate_basin(
    basin: Basin,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    seed: int = 0,
    params: SynthParams | None = None,
) -> pd.DataFrame:
    """Generate one basin's daily synthetic record."""
    params = params or SynthParams()
    rng = np.random.default_rng(basin_seed(basin.id, seed))

    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy().astype(float)
    n = len(dates)

    # --- forcing -----------------------------------------------------------
    # Multi-year wet/dry regimes give the compound-extreme behaviour the
    # proposal is about: multi-year drought punctuated by flood years.
    regime = 1.0 + 0.35 * np.sin(2 * np.pi * np.arange(n) / (365.25 * 4.3) + rng.uniform(0, 6.28))

    p_storm = _seasonal_storm_probability(doy) * regime
    wet_day = rng.random(n) < np.clip(p_storm, 0.01, 0.95)
    # Heavy-tailed storm depths so that the 95th-percentile flow threshold is
    # driven by a handful of genuinely large events.
    depth = rng.gamma(shape=0.62, scale=26.0, size=n)
    precip_true = np.where(wet_day, depth, 0.0)

    # Rescale to the basin's nominal annual precipitation.
    years = n / 365.25
    scale = basin.mean_annual_precip_mm / max(precip_true.sum() / years, 1e-6)
    precip_true *= scale

    aridity = float(np.clip(900.0 / basin.mean_annual_precip_mm, 0.7, 2.0))
    pet = _pet_mm(doy, params, aridity)

    # --- conceptual water balance -----------------------------------------
    soil = np.zeros(n)
    gw = np.zeros(n)
    quick = np.zeros(n)
    aet = np.zeros(n)
    runoff_mm = np.zeros(n)

    s = 0.45 * params.soil_capacity_mm
    g = 40.0
    q_store = 0.0
    for i in range(n):
        wetness = s / params.soil_capacity_mm
        # Saturation-excess style runoff coefficient: the same storm produces
        # far more runoff on wet antecedent soil. This is the relationship the
        # ML model has to recover from SMAP + IMERG.
        rc = 0.03 + 0.72 * wetness**params.runoff_exponent
        quickflow = precip_true[i] * rc
        infiltration = precip_true[i] - quickflow

        s += infiltration
        overflow = max(0.0, s - params.soil_capacity_mm)
        s -= overflow
        quickflow += overflow

        drainable = max(0.0, s - 0.35 * params.soil_capacity_mm)
        percolation = params.percolation_fraction * drainable * 0.05
        s -= percolation
        g += percolation

        et = pet[i] * np.clip(s / (0.6 * params.soil_capacity_mm), 0.0, 1.0)
        et = min(et, s)
        s -= et

        baseflow = params.baseflow_coefficient * g
        g -= baseflow

        q_store = q_store * params.quickflow_recession + quickflow
        routed = q_store * (1.0 - params.quickflow_recession)

        soil[i] = s
        gw[i] = g
        aet[i] = et
        quick[i] = routed
        runoff_mm[i] = routed + baseflow

    flow_true_cfs = runoff_mm * basin.drainage_area_km2 * MM_DAY_KM2_TO_CFS

    # --- sensor degradation ------------------------------------------------
    precip_obs = precip_true * rng.lognormal(0.0, params.imerg_log_error_sd, n)
    precip_obs = np.where(precip_true > 0.1, precip_obs, 0.0)

    sm_true = 0.06 + 0.34 * (soil / params.soil_capacity_mm)
    sm_obs = np.clip(sm_true + rng.normal(0, params.smap_noise, n), 0.02, 0.45)

    et_obs = np.clip(aet + rng.normal(0, params.ecostress_noise_mm, n), 0.0, None)
    gap = rng.random(n) < params.ecostress_gap_fraction
    et_obs[gap] = np.nan  # cloud / revisit gaps, filled downstream

    # SWOT-like inundated area from an at-a-station hydraulic geometry power law.
    extent_ref = 0.0018 * basin.drainage_area_km2**0.6
    extent = extent_ref * np.power(np.maximum(flow_true_cfs, 1.0) / 500.0, 0.38)
    extent_obs = np.clip(extent * rng.lognormal(0.0, 0.07, n), 0.0, None)

    flow_obs = flow_true_cfs * rng.lognormal(0.0, params.gauge_log_error_sd, n)

    frame = pd.DataFrame(
        {
            "date": dates,
            "basin_id": basin.id,
            "precip_mm": precip_obs,
            "soil_moisture": sm_obs,
            "et_mm": et_obs,
            "water_extent_km2": extent_obs,
            "flow_cfs": flow_obs,
            # Hidden truth, kept for diagnostics only. build_features() never
            # reads these columns.
            "_soil_store_mm": soil,
            "_gw_store_mm": gw,
            "_flow_true_cfs": flow_true_cfs,
            "_precip_true_mm": precip_true,
        }
    )
    return frame


def simulate_all(
    basins: list[Basin],
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    seed: int = 0,
    params: SynthParams | None = None,
) -> pd.DataFrame:
    frames = [simulate_basin(b, start=start, end=end, seed=seed, params=params) for b in basins]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["basin_id", "date"]).reset_index(drop=True)
