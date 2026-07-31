#!/usr/bin/env python3
"""Assemble a static site from outputs/ for GitHub Pages.

    python scripts/run_pipeline.py --scenario
    python scripts/make_report.py
    python scripts/build_site.py

The report is the site. This script only stages it: outputs/report.html becomes
_site/index.html, the figures come along, and the CSVs are published next to the
page so a reader can check any number against its source instead of taking the
rendered table on trust.

outputs/ is gitignored on purpose (see .gitignore), so the published site is
always rebuilt from a seed in CI rather than served from committed artefacts.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from twr.config import OUTPUT_DIR  # noqa: E402

# Published for inspection alongside the rendered page. Anything not listed is
# left out rather than swept in, so adding an output does not silently publish it.
DATA_FILES = (
    "site_flags.csv",
    "statewide_screening.csv",
    "flag_history.csv",
    "hmf_events.csv",
    "spatial_cv_folds.csv",
    "feature_importance.csv",
    "asr_scenario_sweep.csv",
    "storage_balance.csv",
    "run_summary.json",
)

BANNER = """
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
     margin:0 auto;padding:14px 24px 0;font-size:13px;color:#5b6670">
Published from <a href="https://github.com/ktwu01/texas-water-resilience-demo">the
repository</a>, rebuilt from a fixed seed on every push. Source data for every table
on this page: {links}.
</div>
"""


def build_site(output_dir: Path, site_dir: Path) -> Path:
    output_dir, site_dir = Path(output_dir), Path(site_dir)
    report = output_dir / "report.html"
    if not report.exists():
        raise FileNotFoundError(
            f"{report} not found. Run scripts/run_pipeline.py then scripts/make_report.py."
        )

    if site_dir.exists():
        shutil.rmtree(site_dir)
    site_dir.mkdir(parents=True)

    figures = output_dir / "figures"
    if figures.exists():
        shutil.copytree(figures, site_dir / "figures")

    # The basin map is a separate document that index.html iframes, so it has to
    # travel with the page or the map slot renders empty.
    basin_map = output_dir / "basin_map.html"
    if basin_map.exists():
        shutil.copy2(basin_map, site_dir / basin_map.name)

    published = []
    data_dir = site_dir / "data"
    data_dir.mkdir()
    for name in DATA_FILES:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, data_dir / name)
            published.append(name)

    links = ", ".join(f'<a href="data/{name}">{name}</a>' for name in published) or "none written"
    html = report.read_text(encoding="utf-8")
    html = html.replace("<body>", "<body>" + BANNER.format(links=links), 1)
    (site_dir / "index.html").write_text(html, encoding="utf-8")

    # Pages otherwise runs the output through Jekyll, which drops _-prefixed paths.
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    return site_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--site-dir", type=Path, default=REPO_ROOT / "_site")
    args = parser.parse_args(argv)
    path = build_site(args.output_dir, args.site_dir)
    print(f"staged {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
