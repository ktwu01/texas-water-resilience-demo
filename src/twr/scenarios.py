"""Facility-scale scenario evaluation.

Split out of pipeline.py, which orchestrates one run against the world as it is.
This module asks the different question a utility actually brings to the table:
not "what will the river do" but "if I had a bigger intake, or if I recovered
500 AF first, would this flood be mine".

The distinction matters because a scenario sweep holds the forecast fixed and
varies the infrastructure, which is the inverse of everything in the pipeline.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import capture_index, features
from .config import Basin, Site
from .pipeline import PipelineResult


def scenario_sweep(
    result: PipelineResult,
    site: Site,
    basin: Basin,
    headroom_grid: np.ndarray | None = None,
    diversion_scale_grid: np.ndarray | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Facility-scale what-if: how do flags move with headroom and intake size?

    This is the ASR scenario-evaluation deliverable. It answers the question a
    utility actually asks, which is not "what will the river do" but "if I had a
    bigger intake, or if I recovered 500 AF first, would this flood be mine".

    ``as_of`` defaults to the site's most actionable day in the replay window
    rather than the last day of the record. Evaluated on an arbitrary quiet day,
    every scenario returns the same answer and the sweep shows nothing: the
    binding constraint is hydrology in all 40 cells. A capital-planning question
    is only meaningful against a day when there is water to argue about.
    """
    if headroom_grid is None:
        # Sample around the operational threshold, not evenly across capacity.
        # Headroom only changes the answer where it is comparable to the volume
        # being considered; a grid spanning 100 to 2000 AF against a 30 AF
        # threshold is eight identical rows.
        headroom_grid = np.unique(
            np.concatenate(
                [
                    [0.0],
                    site.operational_threshold_af * np.array([0.5, 1.0, 2.0, 4.0]),
                    site.infrastructure.storage_capacity_af * np.array([0.25, 0.5, 1.0]),
                ]
            )
        )
    diversion_scale_grid = (
        np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        if diversion_scale_grid is None
        else diversion_scale_grid
    )

    spec = features.FeatureSpec(horizon_days=result.assessments[0].horizon_days)
    usable = result.table.dropna(subset=spec.feature_columns)

    if as_of is None:
        history = result.history
        subset = (
            history[(history["site_id"] == site.id) & (history["basin_id"] == basin.id)]
            if not history.empty
            else pd.DataFrame()
        )
        as_of = (
            pd.Timestamp(subset.loc[subset["capture_index"].idxmax(), "date"])
            if not subset.empty
            else usable["date"].max()
        )
    as_of = pd.Timestamp(as_of)

    row = usable[(usable["basin_id"] == basin.id) & (usable["date"] == as_of)]
    if row.empty:
        raise ValueError(f"no usable feature row for {basin.id} on {as_of.date()}")
    X = row[spec.feature_columns].astype(float)
    samples = result.model.predict_volume_samples(X)[0]
    event_p = float(result.model.predict_event_probability(X)[0])

    rows = []
    for scale in diversion_scale_grid:
        infra = replace(
            site.infrastructure,
            max_diversion_cfs=site.infrastructure.max_diversion_cfs * float(scale),
            conveyance_cfs=site.infrastructure.conveyance_cfs * float(scale),
            treatment_mgd=site.infrastructure.treatment_mgd * float(scale),
            recharge_wells=max(int(round(site.infrastructure.recharge_wells * float(scale))), 1),
        )
        for head in headroom_grid:
            item = capture_index.assess(
                site_id=site.id,
                basin_id=basin.id,
                date=str(as_of.date()),
                volume_samples_af=samples,
                event_probability=event_p,
                eflow=basin.eflow,
                rights=basin.water_rights,
                infra=infra,
                storage_headroom_af=float(head),
                operational_threshold_af=site.operational_threshold_af,
                horizon_days=spec.horizon_days,
            )
            rows.append(
                {
                    "as_of": str(as_of.date()),
                    "diversion_scale": float(scale),
                    "headroom_af": float(head),
                    "capture_index": item.capture_index,
                    "flag": item.flag,
                    "expected_capturable_af": item.expected_capturable_af,
                    "binding_constraint": item.binding_constraint,
                }
            )
    return pd.DataFrame(rows)
