"""Uncertainty quantification and spatial cross-validation.

Random k-fold on a daily hydrologic table is close to meaningless: neighbouring
days are nearly identical, so a random split reports skill that vanishes the
moment the model sees a new basin. The proposal commits to spatial
cross-validation, so the evaluation here is leave-one-basin-out: fit on every
basin but one, predict the held-out basin, and report per-fold skill. The spread
across folds is the honest estimate of what happens at an ungauged site.

Interval quality is scored with PICP (prediction interval coverage probability).
A nominal 80% interval that covers 45% of observations is a model that is lying
about its own confidence, and that is exactly what a manager must not be handed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .model import HybridCaptureModel, ModelConfig


def quantiles(
    samples: np.ndarray, qs: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> dict[str, np.ndarray]:
    """Row-wise quantiles of an (n_rows, n_members) ensemble array."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("expected a 2-D (rows, members) array")
    return {f"q{int(q * 100):02d}": np.quantile(samples, q, axis=1) for q in qs}


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations inside the interval (PICP)."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


@dataclass
class FoldResult:
    fold: str
    n_train: int
    n_test: int
    mae_log_af: float
    mae_af: float
    event_auc: float | None
    picp_80: float
    mean_interval_af: float
    observed_event_rate: float


def leave_one_basin_out(
    X: pd.DataFrame,
    y_volume: pd.Series,
    y_event: pd.Series,
    groups: pd.Series,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Spatial cross-validation across basins."""
    config = config or ModelConfig()
    results: list[FoldResult] = []
    for held_out in sorted(groups.unique()):
        test_mask = (groups == held_out).to_numpy()
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = HybridCaptureModel(config)
        model.fit(X[train_mask], y_volume[train_mask], y_event[train_mask])

        samples = model.predict_volume_samples(X[test_mask])
        qs = quantiles(samples, (0.1, 0.5, 0.9))
        y_true = y_volume[test_mask].to_numpy(dtype=float)

        event_true = y_event[test_mask].to_numpy(dtype=int)
        if len(np.unique(event_true)) > 1:
            auc = float(roc_auc_score(event_true, model.predict_event_probability(X[test_mask])))
        else:
            auc = None

        results.append(
            FoldResult(
                fold=str(held_out),
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
                mae_log_af=float(mean_absolute_error(np.log1p(y_true), np.log1p(qs["q50"]))),
                mae_af=float(mean_absolute_error(y_true, qs["q50"])),
                event_auc=auc,
                picp_80=coverage(y_true, qs["q10"], qs["q90"]),
                mean_interval_af=interval_width(qs["q10"], qs["q90"]),
                observed_event_rate=float(event_true.mean()),
            )
        )

    if not results:
        raise ValueError("cross-validation produced no folds")
    return pd.DataFrame([r.__dict__ for r in results])


def summarise_folds(folds: pd.DataFrame) -> dict[str, float]:
    """Aggregate fold table into the handful of numbers worth reporting."""
    numeric = folds.select_dtypes(include="number")
    summary = {f"{col}_mean": float(numeric[col].mean()) for col in numeric.columns}
    summary.update({f"{col}_std": float(numeric[col].std(ddof=0)) for col in numeric.columns})
    summary["n_folds"] = int(len(folds))
    return summary
