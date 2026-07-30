"""Learned super-resolution of coarse forcing fields.

The proposal downscales near-real-time NASA forcing (NLDAS-3 class) toward
high-resolution reanalysis (CONUS404 class) with a deep-learning
super-resolution model. A CNN is the right tool there and the wrong tool for a
demo that must run on a laptop in seconds with no GPU and no PyTorch install, so
this module implements the same *problem statement* with a transparent learned
baseline:

    fine_field  ~  interpolated(coarse_field) + f(coarse patch, terrain patch)

A ridge regression on stacked patch features learns the residual. The point the
demo has to make is structural, not architectural: interpolation alone cannot
recover fine-scale precipitation structure, whereas a model conditioned on
static high-resolution covariates (terrain) can, because that structure is not
random. Swapping ``ResidualSuperResolver`` for a CNN is a drop-in change behind
the same fit/predict interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


def bilinear_upsample(field: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a 2-D field by an integer factor with bilinear interpolation."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if field.ndim != 2:
        raise ValueError("field must be 2-D")
    if factor == 1:
        return field.astype(float).copy()

    ny, nx = field.shape
    y_src = (np.arange(ny * factor) + 0.5) / factor - 0.5
    x_src = (np.arange(nx * factor) + 0.5) / factor - 0.5
    y_src = np.clip(y_src, 0, ny - 1)
    x_src = np.clip(x_src, 0, nx - 1)

    y0 = np.floor(y_src).astype(int)
    x0 = np.floor(x_src).astype(int)
    y1 = np.minimum(y0 + 1, ny - 1)
    x1 = np.minimum(x0 + 1, nx - 1)
    wy = (y_src - y0)[:, None]
    wx = (x_src - x0)[None, :]

    top = field[np.ix_(y0, x0)] * (1 - wx) + field[np.ix_(y0, x1)] * wx
    bottom = field[np.ix_(y1, x0)] * (1 - wx) + field[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bottom * wy


def block_mean(field: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate a fine field to a coarse grid by block averaging."""
    ny, nx = field.shape
    if ny % factor or nx % factor:
        raise ValueError("field dimensions must be divisible by factor")
    return field.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def synthetic_terrain(size: int, seed: int = 7) -> np.ndarray:
    """A smooth, fixed, high-resolution covariate standing in for elevation."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    terrain = np.zeros((size, size))
    for wavelength, amplitude in ((1.0, 1.0), (2.3, 0.55), (4.7, 0.3), (9.1, 0.15)):
        phase_y, phase_x = rng.uniform(0, 2 * np.pi, 2)
        terrain += amplitude * np.sin(2 * np.pi * wavelength * yy + phase_y) * np.cos(
            2 * np.pi * wavelength * xx + phase_x
        )
    terrain -= terrain.min()
    return terrain / terrain.max()


def make_paired_fields(
    n_samples: int = 60,
    fine_size: int = 64,
    factor: int = 8,
    seed: int = 0,
    terrain_strength: float = 0.85,
    noise: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (coarse, fine, terrain) synthetic precipitation fields.

    Fine-scale structure is orographic: it is a deterministic function of the
    coarse field and the static terrain, plus irreducible noise. That is the
    honest version of what makes downscaling learnable at all.
    """
    if fine_size % factor:
        raise ValueError("fine_size must be divisible by factor")
    rng = np.random.default_rng(seed)
    terrain = synthetic_terrain(fine_size, seed=seed + 7)
    coarse_size = fine_size // factor

    coarse_fields = np.zeros((n_samples, coarse_size, coarse_size))
    fine_fields = np.zeros((n_samples, fine_size, fine_size))

    for i in range(n_samples):
        # Large-scale storm: a couple of smooth blobs on the coarse grid.
        base = np.zeros((coarse_size, coarse_size))
        yy, xx = np.meshgrid(np.arange(coarse_size), np.arange(coarse_size), indexing="ij")
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.uniform(0, coarse_size, 2)
            width = rng.uniform(1.0, 2.5)
            amplitude = rng.gamma(2.0, 8.0)
            base += amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * width**2))

        smooth_fine = bilinear_upsample(base, factor)
        enhancement = 1.0 + terrain_strength * (terrain - terrain.mean())
        fine = smooth_fine * enhancement
        fine = fine * (1.0 + noise * rng.standard_normal(fine.shape))
        fine = np.clip(fine, 0.0, None)

        # The coarse "observation" is the block mean of the truth, which is what
        # a coarse-resolution product actually measures.
        coarse_fields[i] = block_mean(fine, factor)
        fine_fields[i] = fine

    return coarse_fields, fine_fields, terrain


@dataclass
class SuperResolutionMetrics:
    rmse_interpolated: float
    rmse_model: float
    skill_score: float
    n_test: int

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse_interpolated": self.rmse_interpolated,
            "rmse_model": self.rmse_model,
            "skill_score": self.skill_score,
            "n_test": float(self.n_test),
        }


class ResidualSuperResolver:
    """Predict the interpolation residual from local coarse and terrain context."""

    def __init__(self, factor: int, alpha: float = 1.0) -> None:
        self.factor = factor
        self.model = Ridge(alpha=alpha)
        self.terrain_: np.ndarray | None = None

    def _design(self, coarse_fields: np.ndarray, terrain: np.ndarray) -> np.ndarray:
        rows = []
        for coarse in coarse_fields:
            upsampled = bilinear_upsample(coarse, self.factor)
            terrain_dev = terrain - terrain.mean()
            # Interaction terms: orographic enhancement scales with storm size.
            features = np.stack(
                [
                    upsampled.ravel(),
                    terrain_dev.ravel(),
                    (upsampled * terrain_dev).ravel(),
                    (upsampled * terrain_dev**2).ravel(),
                    np.full(upsampled.size, coarse.mean()),
                ],
                axis=1,
            )
            rows.append(features)
        return np.concatenate(rows, axis=0)

    def fit(self, coarse_fields: np.ndarray, fine_fields: np.ndarray, terrain: np.ndarray):
        self.terrain_ = terrain
        X = self._design(coarse_fields, terrain)
        residual = np.concatenate(
            [
                (fine - bilinear_upsample(coarse, self.factor)).ravel()
                for coarse, fine in zip(coarse_fields, fine_fields, strict=True)
            ]
        )
        self.model.fit(X, residual)
        return self

    def predict(self, coarse_fields: np.ndarray) -> np.ndarray:
        if self.terrain_ is None:
            raise RuntimeError("model is not fitted")
        X = self._design(coarse_fields, self.terrain_)
        residual = self.model.predict(X)
        fine_size = self.terrain_.shape[0]
        residual = residual.reshape(len(coarse_fields), fine_size, fine_size)
        interpolated = np.stack(
            [bilinear_upsample(coarse, self.factor) for coarse in coarse_fields]
        )
        return np.clip(interpolated + residual, 0.0, None)


def evaluate_downscaling(
    n_train: int = 48, n_test: int = 16, fine_size: int = 64, factor: int = 8, seed: int = 0
) -> SuperResolutionMetrics:
    """Train the residual model and compare it against plain interpolation."""
    coarse, fine, terrain = make_paired_fields(
        n_samples=n_train + n_test, fine_size=fine_size, factor=factor, seed=seed
    )
    coarse_tr, fine_tr = coarse[:n_train], fine[:n_train]
    coarse_te, fine_te = coarse[n_train:], fine[n_train:]

    interpolated = np.stack([bilinear_upsample(c, factor) for c in coarse_te])
    predicted = ResidualSuperResolver(factor).fit(coarse_tr, fine_tr, terrain).predict(coarse_te)

    rmse_interp = float(np.sqrt(np.mean((interpolated - fine_te) ** 2)))
    rmse_model = float(np.sqrt(np.mean((predicted - fine_te) ** 2)))
    skill = 1.0 - rmse_model / rmse_interp if rmse_interp > 0 else 0.0
    return SuperResolutionMetrics(rmse_interp, rmse_model, float(skill), n_test)
