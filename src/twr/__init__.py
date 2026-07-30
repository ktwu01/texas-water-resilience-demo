"""Texas water resilience demo: high-magnitude flow capture decision support.

A runnable, synthetic-data demonstration of the analysis chain described in
"From Floods to Droughts: AI-Enabled End-to-End Water Resilience for Texas"
(proposal 25-WATER25_2-0155, PI Zong-Liang Yang, The University of Texas at
Austin).

This package is a scaffold and a teaching artefact. It contains no observations
and produces no forecasts. See README.md for the full scope statement.
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
