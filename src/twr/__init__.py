"""Texas water resilience demo: high-magnitude flow capture decision support.

A runnable, synthetic-data demonstration of an end-to-end analysis chain: from
flood-flow forecasting to managed aquifer recharge decisions, at three nested
scales.

This package is a scaffold and a teaching artefact. It contains no observations
and produces no forecasts. No organisation or person is named, affiliated, or
endorsed anywhere in it. See README.md for the full scope statement.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (  # noqa: F401
    aquifer,
    capture_index,
    config,
    constraints,
    downscale,
    features,
    geo,
    hmf,
    ingest,
    model,
    pipeline,
    scenarios,
    synth,
    uncertainty,
    units,
)

__all__ = [
    "aquifer",
    "capture_index",
    "config",
    "constraints",
    "downscale",
    "features",
    "geo",
    "hmf",
    "ingest",
    "model",
    "pipeline",
    "scenarios",
    "synth",
    "uncertainty",
    "units",
    "__version__",
]
