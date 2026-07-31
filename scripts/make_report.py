#!/usr/bin/env python3
"""Render a static HTML briefing from outputs/.

The serverless twin of the Streamlit dashboard: same numbers, same flag system,
same basin map. The map is the dashboard's own pydeck deck exported with
to_html(), so it stays interactive and hoverable in a page with no server behind
it, and there is one map implementation rather than two that drift.

Run scripts/run_pipeline.py first.

    python scripts/make_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# The basin map is shared with the dashboard rather than reimplemented here. Same
# deck, same hover cards, same anchors; pydeck can export a deck as a standalone
# HTML file, so the published page gets the interactive map and not a screenshot
# of one.
sys.path.insert(1, str(REPO_ROOT / "dashboard"))

from basin_map import BASEMAP_ATTRIBUTION, basin_map  # noqa: E402

from twr.capture_index import FLAG_COLORS, FLAG_THRESHOLDS  # noqa: E402
from twr.config import OUTPUT_DIR  # noqa: E402
from twr.geo import load_geography  # noqa: E402
from twr.map_svg import basin_map_svg, size_legend_svg  # noqa: E402

MAP_FILENAME = "basin_map.html"

# Dark, to match the dashboard. A white figure dropped into a dark page reads as
# a hole punched in it.
INK = "#c9d1d9"
PAPER = "#0d1117"
PLOT_STYLE = {
    "figure.dpi": 130,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#30363d",
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": "#30363d",
    "legend.labelcolor": INK,
}


def _hydrograph(timeseries: pd.DataFrame, basin_id: str, out: Path) -> Path | None:
    group = timeseries[timeseries["basin_id"] == basin_id].sort_values("date")
    if group.empty:
        return None
    threshold = float(group["hmf_threshold_cfs"].iloc[0])

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, height_ratios=[2, 1, 1])
    axes[0].fill_between(group["date"], 0, group["flow_cfs"], color="#2e7fd4", alpha=0.35, lw=0)
    axes[0].plot(group["date"], group["flow_cfs"], color="#1b5fa8", lw=0.6)
    axes[0].axhline(
        threshold, color="#b23a48", ls="--", lw=1.0,
        label=f"HMF threshold {threshold:,.0f} cfs",
    )
    above = group[group["flow_cfs"] > threshold]
    axes[0].scatter(
        above["date"], above["flow_cfs"], s=4, color="#b23a48", zorder=3, label="HMF days"
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("discharge (cfs, log)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].set_title(f"{basin_id}: synthetic hydrograph and high-magnitude flows", loc="left")

    axes[1].plot(group["date"], group["soil_moisture"], color="#8a5b2c", lw=0.7)
    axes[1].set_ylabel("SMAP-proxy\nsoil moisture")

    axes[2].plot(group["date"], group["storage_deficit_index"], color="#1e9e63", lw=0.7)
    axes[2].set_ylabel("antecedent\nstorage deficit")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("date")

    fig.tight_layout()
    path = out / f"hydrograph_{basin_id}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _screening_chart(statewide: pd.DataFrame, out: Path) -> Path | None:
    if statewide.empty:
        return None
    frame = statewide.sort_values("capture_index")
    colors = [FLAG_COLORS.get(flag, "#888888") for flag in frame["flag"]]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(frame) + 1.6))
    ax.barh(frame["basin_name"], frame["capture_index"], color=colors)
    for name, threshold in FLAG_THRESHOLDS.items():
        ax.axvline(threshold, color="#444444", ls=":", lw=0.8)
        ax.text(threshold, len(frame) - 0.4, name, rotation=90, fontsize=7, va="top", ha="right")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Capture Index   P(feasible volume >= operational threshold)")
    ax.set_title("Statewide HMF capture screening", loc="left")
    fig.tight_layout()
    path = out / "statewide_screening.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _storage_chart(storage: pd.DataFrame, out: Path) -> Path | None:
    if storage.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for site_id, group in storage.groupby("site_id"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["storage_fraction"], lw=0.9, label=site_id)
    ax.set_ylim(0, 1)
    ax.set_ylabel("storage / capacity")
    ax.set_xlabel("date")
    ax.set_title("Antecedent storage state per decision unit", loc="left")
    ax.legend(frameon=False, fontsize=8, ncols=3)
    fig.tight_layout()
    path = out / "storage_state.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _flag_cards(flags: pd.DataFrame) -> str:
    cards = []
    for _, row in flags.iterrows():
        color = FLAG_COLORS.get(row["flag"], "#888888")
        cards.append(
            f"""<div class="card" style="border-left:6px solid {color}">
  <div class="flag" style="color:{color}">{row['flag']}</div>
  <div class="site">{row['site_id']}</div>
  <table>
    <tr><td>Capture Index</td><td><b>{row['capture_index']:.2f}</b></td></tr>
    <tr><td>P(HMF in window)</td><td>{row['event_probability']:.2f}</td></tr>
    <tr><td>Expected capturable</td><td>{row['expected_capturable_af']:,.0f} AF</td></tr>
    <tr><td>80% interval</td><td>{row['q10_capturable_af']:,.0f}
        to {row['q90_capturable_af']:,.0f} AF</td></tr>
    <tr><td>Binding now</td><td><code>{row['binding_constraint']}</code></td></tr>
    <tr><td>Binding if captured</td><td><code>{row['binding_if_captured']}</code></td></tr>
    <tr><td>Aquifer headroom</td><td>{row['storage_headroom_af']:,.0f} AF</td></tr>
  </table>
  <p class="action">{row['action']}</p>
</div>"""
        )
    return "\n".join(cards)


def _basin_map(statewide: pd.DataFrame, output_dir: Path) -> tuple[str, str]:
    """Export the dashboard's deck as a standalone page and embed it in an iframe.

    Returns (embed_html, caveat_html), both empty if there is nothing to draw. The
    deck is iframed rather than inlined because pydeck emits a whole document,
    scripts and all, and splicing that into this page's <body> would put two
    competing <head>s in one file.
    """
    if statewide.empty:
        return "", ""
    deck, placed = basin_map(statewide, load_geography(), highlight_coastal=True)
    if deck is None:
        return "", ""

    deck.to_html(str(output_dir / MAP_FILENAME), open_browser=False, notebook_display=False)

    geography = load_geography()
    # The deck needs WebGL. Where it is missing it paints nothing and says nothing,
    # so the page ships an SVG of the same data and swaps to it rather than showing
    # an empty rectangle. The swap is done in the page because only the browser
    # knows whether it has WebGL.
    fallback = basin_map_svg(statewide, geography)
    embed = (
        f'<iframe class="map" id="deck-map" data-src="{MAP_FILENAME}" title="Basin map" '
        'loading="lazy"></iframe>'
        f'<div class="map fallback" id="svg-map" hidden>{fallback}</div>'
        f'<div class="legend">{size_legend_svg()}</div>'
        """<script>
  (function () {
    var ok = false;
    try {
      var probe = document.createElement("canvas");
      ok = !!(probe.getContext("webgl2") || probe.getContext("webgl"));
    } catch (error) {
      ok = false;
    }
    var deck = document.getElementById("deck-map");
    if (ok) {
      // Only fetch the deck bundle when it can actually run. Pointing src at it
      // unconditionally boots deck.gl just to watch it fail, which fills the
      // console with errors behind an element nobody can see.
      deck.src = deck.dataset.src;
    } else {
      deck.hidden = true;
      var svg = document.getElementById("svg-map");
      svg.hidden = false;
      svg.insertAdjacentHTML("afterend",
        '<p class="note">This browser has no WebGL, so the interactive map cannot ' +
        'render. Showing the same basins as a static map: same anchors, same flags, ' +
        'same size scale, no panning or hover.</p>');
    }
  })();
</script>"""
    )

    caveat = (
        '<p class="note">Hover a basin for its Capture Index, excess volume, and binding '
        "constraint. The pale markers are the three decision units, and the thin lines "
        "trace each basin to its approximate river mouth.</p>"
        '<p class="note">Markers are hand-placed anchors, not delineated watersheds: this '
        "demo carries no basin boundaries, and an invented outline would overstate what it "
        f"knows. The state polygon is a generalised public-domain national boundary. "
        f"{BASEMAP_ATTRIBUTION}. "
        "The map needs WebGL and network access for the basemap tiles; the state outline is "
        "part of the deck, so Texas still draws without them.</p>"
    )
    unplaced = statewide["basin_id"].nunique() - placed
    if unplaced > 0:
        caveat += (
            f'<p class="note">{unplaced} screened basin(s) have no map anchor and are not '
            "drawn. They are still in the table below.</p>"
        )
    return embed, caveat


def _table_html(frame: pd.DataFrame, columns: list[str], float_fmt: str = "%.3f") -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "<p>No data.</p>"
    return frame[available].to_html(index=False, float_format=lambda v: float_fmt % v, border=0)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Texas HMF Capture Briefing (demo)</title>
<style>
  :root {{ --bg: #0d1117; --panel: #161b22; --line: #30363d; --ink: #c9d1d9;
           --bright: #f0f6fc; --dim: #8b949e; --accent: #58a6ff; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0 auto; max-width: 1080px; padding: 40px 24px 72px; background: var(--bg);
         color: var(--ink); line-height: 1.55; -webkit-font-smoothing: antialiased; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; color: var(--bright); letter-spacing: -0.01em; }}
  h2 {{ font-size: 17px; margin-top: 44px; color: var(--bright); text-transform: uppercase;
        letter-spacing: 0.08em; border-bottom: 1px solid var(--line); padding-bottom: 8px; }}
  a {{ color: var(--accent); }}
  .sub {{ color: var(--dim); margin-top: 0; font-size: 14px; }}
  .banner {{ background: rgba(210, 153, 34, 0.10); border: 1px solid rgba(210, 153, 34, 0.45);
             border-radius: 8px; padding: 14px 18px; font-size: 13px; margin: 20px 0 8px; }}
  .cards {{ display: grid; gap: 16px;
            grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); }}
  .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
           padding: 16px 18px; }}
  .flag {{ font-weight: 700; letter-spacing: 0.08em; font-size: 12px; }}
  .site {{ font-size: 15px; font-weight: 600; margin-bottom: 10px; color: var(--bright); }}
  .card table {{ width: 100%; font-size: 12.5px; }}
  .card td {{ padding: 3px 0; border: 0; }}
  .card td:last-child {{ text-align: right; color: var(--bright); }}
  .action {{ font-size: 12px; color: var(--dim); margin: 12px 0 0; }}
  .note {{ font-size: 12.5px; color: var(--dim); margin: 10px 0 0; }}
  .map {{ width: 100%; height: 560px; border: 1px solid var(--line); border-radius: 10px;
          margin: 14px 0 6px; background: var(--panel); }}
  .map.fallback {{ height: auto; padding: 8px; }}
  .legend {{ margin: 6px 0 2px; }}
  table {{ border-collapse: collapse; font-size: 13px; width: 100%;
           background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }}
  th, td {{ padding: 7px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
  thead th {{ color: var(--dim); font-weight: 600; font-size: 11.5px; text-transform: uppercase;
              letter-spacing: 0.06em; }}
  tbody tr:last-child td {{ border-bottom: 0; }}
  tbody tr:hover {{ background: rgba(88, 166, 255, 0.06); }}
  img {{ max-width: 100%; margin: 14px 0; border-radius: 10px; border: 1px solid var(--line); }}
  code {{ background: rgba(110, 118, 129, 0.2); padding: 1px 6px; border-radius: 4px;
          font-size: 12px; color: var(--bright); }}
  footer {{ margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--line);
            font-size: 12px; color: var(--dim); }}
</style></head><body>
<h1>From Floods to Droughts: HMF Capture Briefing</h1>
<p class="sub">Demo artefact, generated by <code>scripts/make_report.py</code>. Forecast window:
{horizon} days. As of: {as_of}.</p>

<div class="banner"><b>Synthetic data.</b> Every number on this page comes from a stochastic
simulator in <code>twr/synth.py</code>. Nothing here is an observation, a forecast, or an
endorsement by any agency, district, utility, or other organisation. Infrastructure
capacities are illustrative
placeholders, and the map markers are hand-placed anchors rather than delineated watersheds.
If you reached this page by a link rather than by cloning the repository, read
<a href="https://github.com/ktwu01/texas-water-resilience-demo/blob/main/docs/LIMITATIONS.md">
LIMITATIONS.md</a> before quoting anything on it.</div>

<h2>Operational flags by scale</h2>
<div class="cards">
{cards}
</div>

<h2>Scale 1: statewide screening</h2>
{map_embed}
{map_caveat}
{screening_img}
{screening_table}
<p><code>binding_constraint</code> is what limits the planning case and is what makes a row
<code>BLOCKED</code> rather than <code>NO_ACTION</code>.
<code>binding_if_captured</code> is what would limit the response if the forecast opportunity
does materialise, which is what a crew prepares against. On a heavy-tailed forecast the
two legitimately differ.</p>

<h2>Scale 2 and 3: antecedent storage state</h2>
<p>Capture potential is jointly limited by what the river delivers and by what the aquifer can
accept. A full aquifer converts a flood into a <code>storage_headroom</code> block.</p>
{storage_img}

<h2>Trustworthy AI: spatial cross-validation</h2>
<p>Leave-one-basin-out. Each row is a basin held out entirely from training, which is the
relevant test for transfer to an ungauged or newly instrumented basin. <code>picp_80</code> is
the observed coverage of the nominal 80% bootstrap interval; far below 0.80 means the model
is overconfident.</p>
{cv_table}

<h2>Retrospective HMF context</h2>
{events_table}

<h2>Hydrograph and antecedent conditions</h2>
{hydrograph_imgs}

<h2>What the model leans on</h2>
<p>Permutation importance on the volume head, in log space. Increase in mean squared error
when a feature is shuffled.</p>
{importance_table}

<footer>
Demonstration scaffold. Synthetic data throughout. No organisation or person is
named, affiliated, or endorsed.
Flag thresholds: {thresholds}.
</footer>
</body></html>"""


def build_report(output_dir: Path, max_hydrographs: int = 2) -> Path:
    output_dir = Path(output_dir)
    summary_path = output_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. Run scripts/run_pipeline.py first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def _read(name: str) -> pd.DataFrame:
        path = output_dir / name
        if not path.exists():
            return pd.DataFrame()
        parse = ["date"] if name in {"daily_timeseries.csv", "storage_balance.csv"} else None
        return pd.read_csv(path, parse_dates=parse)

    flags = _read("site_flags.csv")
    statewide = _read("statewide_screening.csv")
    storage = _read("storage_balance.csv")
    timeseries = _read("daily_timeseries.csv")
    events = _read("hmf_events.csv")
    importance = _read("feature_importance.csv")
    cv = _read("spatial_cv_folds.csv")

    map_embed, map_caveat = _basin_map(statewide, output_dir)

    with plt.rc_context(PLOT_STYLE):
        screening_png = _screening_chart(statewide, figures)
        storage_png = _storage_chart(storage, figures)
        hydro_pngs = []
        if not timeseries.empty:
            ranked = (
                statewide["basin_id"].tolist()
                if not statewide.empty
                else sorted(timeseries["basin_id"].unique())
            )
            for basin_id in ranked[:max_hydrographs]:
                png = _hydrograph(timeseries, basin_id, figures)
                if png:
                    hydro_pngs.append(png)

    def _img(path: Path | None) -> str:
        return f'<img src="figures/{path.name}" alt="{path.stem}">' if path else ""

    events_summary = pd.DataFrame()
    if events is not None and not events.empty:
        events_summary = (
            events.groupby("basin_name")
            .agg(
                events=("excess_af", "size"),
                median_excess_af=("excess_af", "median"),
                max_excess_af=("excess_af", "max"),
                mean_duration_days=("duration_days", "mean"),
            )
            .reset_index()
            .sort_values("max_excess_af", ascending=False)
        )

    html = TEMPLATE.format(
        horizon=summary.get("horizon_days", "n/a"),
        as_of=summary.get("as_of", "n/a"),
        cards=_flag_cards(flags) if not flags.empty else "<p>No assessments.</p>",
        map_embed=map_embed,
        map_caveat=map_caveat,
        screening_img=_img(screening_png),
        screening_table=_table_html(
            statewide,
            ["basin_name", "capture_index", "flag", "binding_constraint", "binding_if_captured",
             "median_excess_af", "q90_excess_af", "expected_capturable_af", "event_probability"],
        ),
        storage_img=_img(storage_png),
        cv_table=_table_html(
            cv, ["fold", "n_train", "n_test", "mae_log_af", "event_auc", "picp_80",
                 "mean_interval_af", "observed_event_rate"]
        ),
        events_table=_table_html(events_summary, list(events_summary.columns), "%.1f"),
        hydrograph_imgs="\n".join(_img(path) for path in hydro_pngs),
        importance_table=_table_html(importance.head(10), ["feature", "mse_increase"], "%.4f"),
        thresholds=", ".join(f"{k} >= {v}" for k, v in FLAG_THRESHOLDS.items()),
    )

    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hydrographs", type=int, default=2)
    args = parser.parse_args(argv)
    path = build_report(args.output_dir, args.hydrographs)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
