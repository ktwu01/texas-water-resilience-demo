"""Hybrid physics-AI capture model.

Two heads share one feature set:

* a regression head for the volume of high-magnitude flow in the next window,
  trained on log1p(AF) because the target spans several orders of magnitude and
  is zero most days;
* a classification head for whether any HMF day occurs at all, which is the
  question a manager asks first.

Both heads are wrapped in an explicit bootstrap ensemble. Resampling the
training set and refitting is the cheapest defensible way to get a predictive
distribution out of a gradient-boosted model, and it is what the proposal
commits to for Trustworthy AI. The ensemble spread is the model's own admission
of ignorance; it is carried all the way to the Capture Index rather than being
collapsed to a point forecast.

The "physics" half of the hybrid lives in two places: the features encode
conceptual hydrology (antecedent storage deficit, exponentially weighted
antecedent precipitation, flow-to-threshold ratio), and constraints.py clips
every ensemble member to a feasible envelope. The learner is never allowed to
propose an infeasible action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

Kind = Literal["regressor", "classifier"]


def default_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=140,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
    )


def default_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=140,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
    )


class BootstrapEnsemble:
    """Fit ``n_bootstrap`` copies of a base estimator on resampled training data.

    Bootstrap spread alone measures only *epistemic* uncertainty: how much the
    fitted function moves when the training sample is perturbed. On this problem
    the members agree closely, so intervals built from their spread alone cover
    roughly 5% of observations at a nominal 80% level, which is a model lying
    about its own confidence.

    So the ensemble also records out-of-bag residuals, binned by predicted
    value, giving the *aleatoric* term: the irreducible scatter around the
    conditional mean. Binning matters because the scatter is strongly
    heteroscedastic here. Quiet days are predicted almost exactly and floods are
    not, and a pooled residual would manufacture phantom floods on dry days.
    """

    def __init__(
        self,
        base_factory: Callable[[], object],
        kind: Kind = "regressor",
        n_bootstrap: int = 24,
        random_state: int = 0,
        n_residual_bins: int = 8,
    ) -> None:
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be >= 2 for a usable spread")
        self.base_factory = base_factory
        self.kind = kind
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.n_residual_bins = max(int(n_residual_bins), 1)
        self.members_: list[object] = []
        self.residual_bin_edges_: np.ndarray | None = None
        self.residuals_by_bin_: list[np.ndarray] = []
        self.residual_pool_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BootstrapEnsemble:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        n = len(y_arr)
        rng = np.random.default_rng(self.random_state)
        self.members_ = []
        oob_predictions: list[np.ndarray] = []
        oob_residuals: list[np.ndarray] = []

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if self.kind == "classifier" and len(np.unique(y_arr[idx])) < 2:
                # A resample with one class cannot be fit; keep the full set.
                idx = np.arange(n)
            member = self.base_factory()
            member.fit(X_arr[idx], y_arr[idx])
            self.members_.append(member)

            if self.kind == "regressor":
                out_of_bag = np.ones(n, dtype=bool)
                out_of_bag[idx] = False
                if out_of_bag.any():
                    predicted = member.predict(X_arr[out_of_bag])
                    oob_predictions.append(predicted)
                    oob_residuals.append(y_arr[out_of_bag].astype(float) - predicted)

        if oob_residuals:
            self._fit_residual_bins(
                np.concatenate(oob_predictions), np.concatenate(oob_residuals)
            )
        return self

    def _fit_residual_bins(self, predictions: np.ndarray, residuals: np.ndarray) -> None:
        self.residual_pool_ = residuals
        quantiles = np.linspace(0.0, 1.0, self.n_residual_bins + 1)[1:-1]
        edges = np.unique(np.quantile(predictions, quantiles)) if quantiles.size else np.array([])
        self.residual_bin_edges_ = edges
        bins = np.searchsorted(edges, predictions)
        self.residuals_by_bin_ = [residuals[bins == k] for k in range(len(edges) + 1)]

    def sample_residuals(self, predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw residuals matched to the magnitude of each prediction."""
        if self.residual_pool_ is None or self.residual_pool_.size == 0:
            return np.zeros_like(predictions, dtype=float)
        edges = self.residual_bin_edges_
        bins = np.searchsorted(edges, predictions) if edges is not None and edges.size else None
        out = np.zeros_like(predictions, dtype=float)
        if bins is None:
            return rng.choice(self.residual_pool_, size=predictions.shape, replace=True)
        for index, pool in enumerate(self.residuals_by_bin_):
            mask = bins == index
            if not mask.any():
                continue
            source = pool if pool.size >= 20 else self.residual_pool_
            out[mask] = rng.choice(source, size=int(mask.sum()), replace=True)
        return out

    def _check_fitted(self) -> None:
        if not self.members_:
            raise RuntimeError("ensemble is not fitted")

    def predict_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return an (n_rows, n_bootstrap) array of member predictions."""
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if self.kind == "classifier":
            cols = [m.predict_proba(X_arr)[:, 1] for m in self.members_]
        else:
            cols = [m.predict(X_arr) for m in self.members_]
        return np.column_stack(cols)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_samples(X).mean(axis=1)


@dataclass
class ModelConfig:
    n_bootstrap: int = 24
    random_state: int = 0
    horizon_days: int = 7
    # Aleatoric draws per ensemble member. The predictive sample size is
    # n_bootstrap * residual_draws.
    residual_draws: int = 4


class HybridCaptureModel:
    """Volume + occurrence prediction with a bootstrap predictive distribution."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.volume_ = BootstrapEnsemble(
            default_regressor,
            kind="regressor",
            n_bootstrap=self.config.n_bootstrap,
            random_state=self.config.random_state,
        )
        self.event_ = BootstrapEnsemble(
            default_classifier,
            kind="classifier",
            n_bootstrap=self.config.n_bootstrap,
            random_state=self.config.random_state + 977,
        )
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y_volume_af: pd.Series | np.ndarray,
        y_event: pd.Series | np.ndarray,
    ) -> HybridCaptureModel:
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        y_vol = np.asarray(y_volume_af, dtype=float)
        if (y_vol < 0).any():
            raise ValueError("volume target must be non-negative")
        self.volume_.fit(X, np.log1p(y_vol))
        self.event_.fit(X, np.asarray(y_event, dtype=int))
        return self

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ is not None and isinstance(X, pd.DataFrame):
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(f"missing features at prediction time: {sorted(missing)}")
            return X[self.feature_names_]
        return X

    def predict_volume_samples(
        self, X: pd.DataFrame, include_aleatoric: bool = True, seed: int | None = None
    ) -> np.ndarray:
        """Predictive samples of HMF excess volume in AF.

        Returns an (n_rows, n_bootstrap * residual_draws) array combining
        parameter uncertainty (the ensemble) with residual scatter (out-of-bag
        residuals matched to prediction magnitude). Set ``include_aleatoric`` to
        False to inspect the epistemic term alone.
        """
        log_mean = self.volume_.predict_samples(self._align(X))
        if not include_aleatoric or self.config.residual_draws < 1:
            return np.clip(np.expm1(log_mean), 0.0, None)

        rng = np.random.default_rng(self.config.random_state if seed is None else seed)
        draws = [
            log_mean + self.volume_.sample_residuals(log_mean, rng)
            for _ in range(self.config.residual_draws)
        ]
        return np.clip(np.expm1(np.concatenate(draws, axis=1)), 0.0, None)

    def predict_event_probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.event_.predict(self._align(X))

    def predict_event_samples(self, X: pd.DataFrame) -> np.ndarray:
        return self.event_.predict_samples(self._align(X))

    def permutation_importance(
        self,
        X: pd.DataFrame,
        y_volume_af: pd.Series | np.ndarray,
        n_repeats: int = 3,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Cheap permutation importance on the volume head, in log space."""
        X = self._align(X)
        y_true = np.log1p(np.asarray(y_volume_af, dtype=float))
        rng = np.random.default_rng(seed)
        base = float(np.mean((self.volume_.predict(X) - y_true) ** 2))
        rows = []
        for column in X.columns:
            scores = []
            for _ in range(n_repeats):
                shuffled = X.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                scores.append(float(np.mean((self.volume_.predict(shuffled) - y_true) ** 2)))
            rows.append({"feature": column, "mse_increase": float(np.mean(scores)) - base})
        out = pd.DataFrame(rows).sort_values("mse_increase", ascending=False)
        return out.reset_index(drop=True)
