"""End-to-end orchestration: pixels to flags.

Stages
------
1. ingest      synthetic (or real) multi-sensor daily record per basin
2. features    causal features + forward-looking HMF targets
3. evaluate    leave-one-basin-out spatial CV, bootstrap intervals, PICP
4. fit         final model on all basins
5. aquifer     storage headroom per site
6. assess      constraint chain + Capture Index + flag, at all three scales
7. write       CSV/JSON artefacts for the dashboard and the report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import aquifer, capture_index, constraints, features, hmf, ingest, uncertainty
from .config import (
    OUTPUT_DIR,
    Basin,
    Site,
    basins_by_id,
    load_basins,
    load_sites,
)
from .model import HybridCaptureModel, ModelConfig


@dataclass
class PipelineConfig:
    start: str = "2015-01-01"
    end: str = "2025-12-31"
    seed: int = 0
    horizon_days: int = 7
    n_bootstrap: int = 24
    baseline_years: int = 5
    run_spatial_cv: bool = True
    # Length of the replayed flag record. A single snapshot is a poor demo: most
    # days are quiet, so one date usually shows NO_ACTION everywhere and never
    # exercises the flag ladder. Replaying a window shows the system reacting.
    history_days: int = 365
    as_of: str | None = None
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)


@dataclass
class PipelineResult:
    table: pd.DataFrame
    cv_folds: pd.DataFrame | None
    cv_summary: dict[str, float]
    assessments: list[capture_index.CaptureAssessment]
    statewide: pd.DataFrame
    events: pd.DataFrame
    storage: pd.DataFrame
    importance: pd.DataFrame
    history: pd.DataFrame
    model: HybridCaptureModel

    @property
    def flags(self) -> pd.DataFrame:
        rows = [a.to_dict() for a in self.assessments]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["urgency"] = frame["flag"].map(capture_index.flag_rank)
        return frame.sort_values("urgency", ascending=False).reset_index(drop=True)


def _retrospective_events(table: pd.DataFrame, basins: list[Basin]) -> pd.DataFrame:
    """HMF event catalogue per basin, the retrospective context for the NRT flags."""
    frames = []
    for basin in basins:
        group = table[table["basin_id"] == basin.id]
        if group.empty:
            continue
        threshold = float(group["hmf_threshold_cfs"].iloc[0])
        events = hmf.identify_events(group["date"], group["flow_cfs"], threshold)
        if events.empty:
            continue
        events.insert(0, "basin_id", basin.id)
        events.insert(1, "basin_name", basin.name)
        events["hmf_threshold_cfs"] = threshold
        frames.append(events)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def validate_site_feasibility(sites: list[Site], horizon_days: int) -> None:
    """Refuse to run a site whose own threshold is unreachable by its own hardware.

    Without this check the Capture Index is identically zero for that site and
    the dashboard shows a permanent NO_ACTION that looks like a hydrologic
    finding rather than a configuration error. This is exactly how the first
    version of config/sites.yaml was wrong.
    """
    problems = []
    for site in sites:
        caps = constraints.infrastructure_capacity_af(site.infrastructure, horizon_days)
        ceiling = min(caps.values())
        if site.operational_threshold_af > ceiling:
            limiting = min(caps, key=caps.get)
            problems.append(
                f"{site.id}: operational_threshold_af={site.operational_threshold_af:,.0f} "
                f"exceeds the {ceiling:,.1f} AF that {limiting} can handle in "
                f"{horizon_days} days, so its Capture Index can never leave zero"
            )
    if problems:
        raise ValueError(
            "infeasible site configuration in config/sites.yaml:\n  " + "\n  ".join(problems)
        )


def permit_remaining_by_date(
    table: pd.DataFrame, basin: Basin, site: Site, deplete: bool = True
) -> dict[pd.Timestamp, float]:
    """Remaining annual permit volume for one diverter on one river.

    A permit belongs to a diverter, not to a river, so this is computed per
    (site, basin) pair rather than per basin. The charge against the permit is
    the volume the diverter *could actually have taken*: legally available flow
    capped by their own daily hardware limit. Charging the full legally available
    flood, as an earlier version did, exhausted a 45,000 AF permit in the first
    storm of every year and left every later flood spuriously BLOCKED.

    Statewide screening passes ``deplete=False``: the question at that scale is
    whether physically and environmentally capturable water exists, and there is
    no single diverter whose permit could be drawn down.
    """
    rows = table[table["basin_id"] == basin.id].sort_values("date")
    permit = basin.water_rights.permitted_diversion_af_per_year
    if not deplete or rows.empty:
        return dict.fromkeys(rows["date"], float(permit))

    daily_caps = constraints.infrastructure_capacity_af(site.infrastructure, 1.0)
    daily_ceiling = min(daily_caps.values())
    legally_available = (
        rows["daily_excess_af"]
        * (1.0 - basin.eflow.pulse_protection_fraction)
        * basin.water_rights.unappropriated_fraction
    )
    charged = np.minimum(legally_available, daily_ceiling)
    used = charged.groupby(rows["date"].dt.year).cumsum()
    remaining = (permit - used).clip(lower=0.0)
    return dict(zip(rows["date"], remaining, strict=True))


def _assess_series(
    *,
    site: Site,
    basin: Basin,
    rows: pd.DataFrame,
    model: HybridCaptureModel,
    spec: features.FeatureSpec,
    headroom_by_date: dict[pd.Timestamp, float],
    permit_by_date: dict[pd.Timestamp, float],
) -> list[capture_index.CaptureAssessment]:
    """Assess one site/basin pair over many dates with a single batched predict."""
    if rows.empty:
        raise ValueError(f"no feature rows for basin {basin.id}")
    X = rows[spec.feature_columns].astype(float)
    samples = model.predict_volume_samples(X)
    event_p = model.predict_event_probability(X)
    # Mass-balance bound from the antecedent rainfall actually observed over the
    # catchment. Uses the 30-day accumulation because HMF volume in the coming
    # week draws on stored catchment water, not only on rain still to fall.
    ceilings = np.array(
        [
            constraints.mass_balance_ceiling_af(value, basin.drainage_area_km2)
            for value in rows["precip_30d"].to_numpy(dtype=float)
        ]
    )

    fallback = float(np.median(list(headroom_by_date.values()))) if headroom_by_date else 0.0
    out = []
    for position, (_, row) in enumerate(rows.iterrows()):
        date = row["date"]
        out.append(
            capture_index.assess(
                site_id=site.id,
                basin_id=basin.id,
                date=str(pd.Timestamp(date).date()),
                volume_samples_af=samples[position],
                event_probability=float(event_p[position]),
                eflow=basin.eflow,
                rights=basin.water_rights,
                infra=site.infrastructure,
                storage_headroom_af=headroom_by_date.get(date, fallback),
                operational_threshold_af=site.operational_threshold_af,
                horizon_days=spec.horizon_days,
                permit_remaining_af=permit_by_date.get(date),
                mass_balance_ceiling_af=float(ceilings[position]),
            )
        )
    return out


def run(
    config: PipelineConfig | None = None,
    basins: list[Basin] | None = None,
    sites: list[Site] | None = None,
) -> PipelineResult:
    config = config or PipelineConfig()
    basins = basins or load_basins()
    sites = sites or load_sites()
    lookup = basins_by_id(basins)
    validate_site_feasibility(sites, config.horizon_days)

    # 1. ingest
    records = ingest.load_synthetic(
        basins, start=config.start, end=config.end, seed=config.seed
    )
    ingest.validate_record(records)

    # 2. features
    spec = features.FeatureSpec(
        horizon_days=config.horizon_days, baseline_years=config.baseline_years
    )
    table = features.build_features(
        records, {basin.id: basin.hmf_percentile for basin in basins}, spec
    )
    X, y_volume, y_event, groups = features.training_matrix(table, spec)

    model_config = ModelConfig(
        n_bootstrap=config.n_bootstrap,
        random_state=config.seed,
        horizon_days=config.horizon_days,
    )

    # 3. spatial cross-validation
    cv_folds = None
    cv_summary: dict[str, float] = {}
    if config.run_spatial_cv and groups.nunique() > 1:
        cv_folds = uncertainty.leave_one_basin_out(X, y_volume, y_event, groups, model_config)
        cv_summary = uncertainty.summarise_folds(cv_folds)

    # 4. final fit on every basin
    model = HybridCaptureModel(model_config).fit(X, y_volume, y_event)
    importance = model.permutation_importance(X, y_volume, n_repeats=2, seed=config.seed)

    # 5. aquifer storage per site, and permit accounting per basin
    storage_frames = []
    headroom_lookup: dict[str, dict[pd.Timestamp, float]] = {}
    for site in sites:
        if site.is_statewide:
            driver = records.groupby("date", as_index=False)["precip_mm"].mean()
        else:
            driver = records[records["basin_id"] == site.basin_id][["date", "precip_mm"]]
        driver = driver.sort_values("date")
        frame = aquifer.simulate_storage(
            pd.DatetimeIndex(driver["date"]), site, driver["precip_mm"].to_numpy()
        )
        storage_frames.append(frame)
        headroom_lookup[site.id] = dict(zip(frame["date"], frame["headroom_af"], strict=True))
    storage = pd.concat(storage_frames, ignore_index=True)

    # 6. replay the decision window at all three scales
    usable = table.dropna(subset=spec.feature_columns).copy()
    as_of = pd.Timestamp(config.as_of) if config.as_of else usable["date"].max()
    if as_of not in set(usable["date"]):
        raise ValueError(f"as_of {as_of.date()} has no usable feature row")
    window_start = as_of - pd.Timedelta(days=max(config.history_days, 0))
    window = usable[(usable["date"] > window_start) & (usable["date"] <= as_of)]

    assessments: list[capture_index.CaptureAssessment] = []
    statewide_rows: list[dict] = []
    history_rows: list[dict] = []

    for site in sites:
        target_basins = basins if site.is_statewide else [lookup[site.basin_id]]
        snapshot: list[capture_index.CaptureAssessment] = []
        for basin in target_basins:
            rows = window[window["basin_id"] == basin.id].sort_values("date")
            series = _assess_series(
                site=site,
                basin=basin,
                rows=rows,
                model=model,
                spec=spec,
                headroom_by_date=headroom_lookup[site.id],
                permit_by_date=permit_remaining_by_date(
                    table, basin, site, deplete=not site.is_statewide
                ),
            )
            for item in series:
                record = item.to_dict()
                record.pop("limits_af", None)
                record["basin_name"] = basin.name
                record["scale"] = site.scale
                history_rows.append(record)
            snapshot.append(series[-1])

        if site.is_statewide:
            for basin, item in zip(target_basins, snapshot, strict=True):
                row = item.to_dict()
                row.pop("limits_af", None)
                row["basin_name"] = basin.name
                statewide_rows.append(row)
            # The statewide site's own headline is its most urgent basin.
            worst = max(snapshot, key=lambda a: (capture_index.flag_rank(a.flag), a.capture_index))
            assessments.append(worst)
        else:
            assessments.extend(snapshot)

    statewide = pd.DataFrame(statewide_rows)
    if not statewide.empty:
        statewide["urgency"] = statewide["flag"].map(capture_index.flag_rank)
        statewide = statewide.sort_values(
            ["urgency", "capture_index"], ascending=False
        ).reset_index(drop=True)

    history = pd.DataFrame(history_rows)
    if not history.empty:
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values(["site_id", "basin_id", "date"]).reset_index(drop=True)

    events = _retrospective_events(table, basins)

    return PipelineResult(
        table=table,
        cv_folds=cv_folds,
        cv_summary=cv_summary,
        assessments=assessments,
        statewide=statewide,
        events=events,
        storage=storage,
        importance=importance,
        history=history,
        model=model,
    )


def write_outputs(result: PipelineResult, output_dir: Path | None = None) -> dict[str, Path]:
    """Persist everything the dashboard and report need."""
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _csv(name: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        path = output_dir / name
        frame.to_csv(path, index=False)
        written[name] = path

    _csv("statewide_screening.csv", result.statewide)
    _csv("site_flags.csv", result.flags)
    _csv("flag_history.csv", result.history)
    _csv("hmf_events.csv", result.events)
    _csv("storage_balance.csv", result.storage)
    _csv("feature_importance.csv", result.importance)
    if result.cv_folds is not None:
        _csv("spatial_cv_folds.csv", result.cv_folds)

    keep = [
        "date",
        "basin_id",
        "precip_mm",
        "soil_moisture",
        "et_mm",
        "water_extent_km2",
        "flow_cfs",
        "hmf_threshold_cfs",
        "daily_excess_af",
        "storage_deficit_index",
        "soil_moisture_pct",
    ]
    _csv("daily_timeseries.csv", result.table[keep])

    summary = {
        "as_of": result.assessments[0].date if result.assessments else None,
        "horizon_days": result.assessments[0].horizon_days if result.assessments else None,
        "spatial_cv": result.cv_summary,
        "assessments": [a.to_dict() for a in result.assessments],
        "hmf_event_summary": _event_summary_by_basin(result),
        "flag_thresholds": capture_index.FLAG_THRESHOLDS,
        "flag_history_counts": (
            result.history.groupby(["site_id", "flag"])
            .size()
            .unstack(fill_value=0)
            .to_dict("index")
            if not result.history.empty
            else {}
        ),
        "disclaimer": (
            "Synthetic demonstration data. Not an observation, not a forecast, "
            "and not endorsed by NASA, TWDB, or any named partner."
        ),
    }
    path = output_dir / "run_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    written["run_summary.json"] = path
    return written


def _event_summary_by_basin(result: PipelineResult) -> dict[str, dict[str, float]]:
    if result.events.empty:
        return {}
    span = result.table["date"].max() - result.table["date"].min()
    years = max(span.days / 365.25, 1e-6)
    out = {}
    for basin_id, group in result.events.groupby("basin_id"):
        out[str(basin_id)] = hmf.event_summary(group, years)
    return out
