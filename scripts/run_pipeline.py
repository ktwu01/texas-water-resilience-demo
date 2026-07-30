#!/usr/bin/env python3
"""Run the full pipeline and write artefacts to outputs/.

    python scripts/run_pipeline.py --fast
    python scripts/run_pipeline.py --n-bootstrap 40 --horizon 10
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twr import pipeline, scenarios  # noqa: E402
from twr.config import OUTPUT_DIR, load_basins, load_sites  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=7, help="forecast window in days")
    parser.add_argument("--n-bootstrap", type=int, default=24, help="ensemble members per head")
    parser.add_argument("--no-cv", action="store_true", help="skip leave-one-basin-out CV")
    parser.add_argument("--fast", action="store_true", help="small ensemble, no CV, shorter record")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--scenario", action="store_true", help="also write the ASR scenario sweep")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = pipeline.PipelineConfig(
        start=args.start,
        end="2021-12-31" if args.fast else args.end,
        seed=args.seed,
        horizon_days=args.horizon,
        n_bootstrap=6 if args.fast else args.n_bootstrap,
        run_spatial_cv=not (args.no_cv or args.fast),
        output_dir=args.output_dir,
    )

    basins = load_basins()
    sites = load_sites()
    print(f"basins: {len(basins)}   sites: {len(sites)}   horizon: {config.horizon_days} d")
    print(f"ensemble: {config.n_bootstrap} members per head   spatial CV: {config.run_spatial_cv}")
    print("running...", flush=True)

    result = pipeline.run(config, basins=basins, sites=sites)
    written = pipeline.write_outputs(result, config.output_dir)

    if result.cv_folds is not None:
        print("\nLeave-one-basin-out cross-validation")
        columns = ["fold", "n_test", "mae_log_af", "event_auc", "picp_80", "observed_event_rate"]
        print(result.cv_folds[columns].to_string(index=False, float_format="%.3f"))

    print("\nOperational flags (as of the last usable forecast date)")
    columns = [
        "site_id",
        "basin_id",
        "date",
        "capture_index",
        "flag",
        "binding_constraint",
        "binding_if_captured",
        "expected_capturable_af",
        "storage_headroom_af",
    ]
    print(result.flags[columns].to_string(index=False, float_format="%.3f"))

    # The last date in the record is whatever day the simulation happened to end
    # on, and most days are quiet, so the snapshot above usually shows NO_ACTION
    # everywhere. The replay window is where the flag ladder is visible.
    if not result.history.empty:
        counts = result.history.groupby(["site_id", "flag"]).size().unstack(fill_value=0)
        print(f"\nFlag counts over the {config.history_days}-day replay window")
        print(counts.to_string())

        peak = result.history.loc[result.history["capture_index"].idxmax()]
        print(
            f"\nMost actionable day in the replay window: {peak['date'].date()} "
            f"at {peak['site_id']} ({peak['basin_name']})"
        )
        print(
            f"  Capture Index {peak['capture_index']:.2f} -> {peak['flag']}, "
            f"binding constraint '{peak['binding_if_captured']}', "
            f"expected capturable {peak['expected_capturable_af']:,.0f} AF"
        )
        print(f"  Inspect it with: --as-of {peak['date'].date()}")

    if not result.statewide.empty:
        print("\nStatewide screening for TWDB")
        columns = [
            "basin_name",
            "capture_index",
            "flag",
            "binding_constraint",
            "median_excess_af",
            "q90_excess_af",
        ]
        print(result.statewide[columns].to_string(index=False, float_format="%.3f"))

    if args.scenario:
        lookup = {basin.id: basin for basin in basins}
        facility = next((s for s in sites if s.scale == "facility"), None)
        if facility is not None and facility.basin_id in lookup:
            sweep = scenarios.scenario_sweep(result, facility, lookup[facility.basin_id])
            path = Path(config.output_dir) / "asr_scenario_sweep.csv"
            sweep.to_csv(path, index=False)
            written["asr_scenario_sweep.csv"] = path
            print(
                f"\nASR scenario sweep: {len(sweep)} scenarios for {facility.id}, "
                f"evaluated on its most actionable day ({sweep['as_of'].iloc[0]})"
            )
            grid = sweep.pivot_table(
                index="headroom_af", columns="diversion_scale", values="capture_index"
            )
            print("Capture Index by aquifer headroom (rows, AF) and capacity multiplier (columns)")
            print(grid.to_string(float_format="%.2f"))

    print("\nWrote:")
    for name, path in sorted(written.items()):
        print(f"  {name:<28} {path}")
    print("\nReminder: synthetic data. Not an observation and not a forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
