#!/usr/bin/env python3
"""Generate the synthetic multi-sensor record and print its properties.

Run this first if you want to inspect the data before modelling, or to
regenerate it with a different seed. run_pipeline.py will do it for you.

    python scripts/make_synthetic_data.py --seed 7 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twr import hmf, ingest  # noqa: E402
from twr.config import DATA_DIR, load_basins  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite the cached record")
    args = parser.parse_args(argv)

    cache = DATA_DIR / "processed" / "synthetic_daily.csv"
    if args.force and cache.exists():
        cache.unlink()

    basins = load_basins()
    frame = ingest.load_synthetic(basins, start=args.start, end=args.end, seed=args.seed)
    ingest.validate_record(frame)

    span = frame["date"].max() - frame["date"].min()
    years = span.days / 365.25
    print(f"{len(frame):,} daily records across {len(basins)} basins, {years:.1f} years")
    print(f"cached at {cache}\n")

    print("Sensor streams these columns stand in for:")
    products = ingest.products_table()
    for _, row in products.iterrows():
        print(f"  {row['variable']:<28} {row['mission']:<28} {row['archive']}")

    print("\nPer-basin high-magnitude flow behaviour:")
    header = (
        f"{'basin':<14}{'mean Q':>10}{'HMF thresh':>12}"
        f"{'events':>8}{'ev/yr':>7}{'AF/yr above':>14}"
    )
    print(header)
    print("-" * len(header))
    for basin in basins:
        group = frame[frame["basin_id"] == basin.id]
        threshold = hmf.hmf_threshold(group["flow_cfs"], basin.hmf_percentile)
        events = hmf.identify_events(group["date"], group["flow_cfs"], threshold)
        summary = hmf.event_summary(events, years)
        print(
            f"{basin.id:<14}{group['flow_cfs'].mean():>10,.0f}{threshold:>12,.0f}"
            f"{summary['n_events']:>8}{summary['events_per_year']:>7.1f}"
            f"{summary['annual_excess_af']:>14,.0f}"
        )

    gaps = frame["et_mm"].isna().mean()
    print(f"\nECOSTRESS-proxy gap fraction: {gaps:.0%} (cloud and revisit gaps, filled causally)")
    print("Reminder: synthetic data. Not an observation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
