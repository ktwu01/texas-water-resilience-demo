#!/usr/bin/env python3
"""Interactive prototype dashboard.

    pip install streamlit
    python scripts/run_pipeline.py --scenario
    streamlit run dashboard/app.py

Three tabs for the three nested scales in the proposal. Everything is read from
outputs/, so the dashboard never fits a model and starts instantly. If Streamlit
is not installed, scripts/make_report.py renders the same content as static HTML.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from twr.capture_index import FLAG_ACTIONS, FLAG_COLORS, FLAG_THRESHOLDS  # noqa: E402
from twr.config import OUTPUT_DIR  # noqa: E402

st.set_page_config(page_title="Texas HMF Capture (demo)", page_icon="💧", layout="wide")

DATE_COLUMNS = {
    "daily_timeseries.csv": ["date"],
    "storage_balance.csv": ["date"],
    "flag_history.csv": ["date"],
}


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=DATE_COLUMNS.get(name))


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    path = OUTPUT_DIR / "run_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def flag_badge(flag: str) -> str:
    color = FLAG_COLORS.get(flag, "#888888")
    return (
        f'<span style="background:{color};color:#fff;padding:3px 12px;border-radius:12px;'
        f'font-weight:700;font-size:13px;letter-spacing:0.06em">{flag}</span>'
    )


def show_flag_card(row: pd.Series) -> None:
    st.markdown(flag_badge(row["flag"]), unsafe_allow_html=True)
    st.metric("Capture Index", f"{row['capture_index']:.2f}")
    st.caption(FLAG_ACTIONS.get(row["flag"], ""))
    left, right = st.columns(2)
    left.metric("Expected capturable", f"{row['expected_capturable_af']:,.0f} AF")
    right.metric("P(HMF in window)", f"{row['event_probability']:.2f}")
    st.caption(
        f"80% interval {row['q10_capturable_af']:,.0f} to {row['q90_capturable_af']:,.0f} AF "
        f"| aquifer headroom {row['storage_headroom_af']:,.0f} AF"
    )
    st.caption(
        f"Binding now: `{row['binding_constraint']}` "
        f"| if the opportunity materialises: `{row['binding_if_captured']}`"
    )


def flag_timeline(history: pd.DataFrame, site_id: str) -> None:
    """Capture Index over the replayed window, with the flag thresholds marked."""
    group = history[history["site_id"] == site_id]
    if group.empty:
        st.info("No replay history for this site.")
        return
    if group["basin_id"].nunique() > 1:
        series = group.pivot_table(
            index="date", columns="basin_name", values="capture_index", aggfunc="max"
        )
    else:
        series = group.set_index("date")[["capture_index"]]
    st.line_chart(series, height=260)
    thresholds = ", ".join(f"{name} >= {value}" for name, value in FLAG_THRESHOLDS.items())
    st.caption(f"Flag thresholds: {thresholds}")

    counts = group["flag"].value_counts()
    columns = st.columns(len(counts))
    for column, (flag, count) in zip(columns, counts.items(), strict=True):
        column.markdown(flag_badge(flag), unsafe_allow_html=True)
        column.caption(f"{count} of {len(group)} days")


def main() -> None:
    summary = load_summary()
    flags = load("site_flags.csv")
    if flags.empty:
        st.error(
            "No outputs found. Run `python scripts/run_pipeline.py --scenario` first, "
            f"which writes to {OUTPUT_DIR}."
        )
        st.stop()

    st.title("From Floods to Droughts")
    st.caption(
        "AI-enabled high-magnitude-flow capture decision support for Texas. "
        f"Forecast window {summary.get('horizon_days', '?')} days, "
        f"as of {summary.get('as_of', '?')}."
    )
    st.warning(
        "**Synthetic demonstration data.** Every value here is simulated. Nothing on this "
        "page is an observation, a forecast, or an endorsement by NASA, TWDB, or any named "
        "partner. Infrastructure capacities are illustrative placeholders.",
        icon="⚠️",
    )

    statewide = load("statewide_screening.csv")
    history = load("flag_history.csv")
    timeseries = load("daily_timeseries.csv")
    storage = load("storage_balance.csv")
    events = load("hmf_events.csv")
    cv = load("spatial_cv_folds.csv")
    sweep = load("asr_scenario_sweep.csv")

    state_tab, watershed_tab, facility_tab, trust_tab = st.tabs(
        [
            "1. Statewide screening (TWDB)",
            "2. Watershed MAR (Rolling Plains GCD)",
            "3. ASR operations (City of Kerrville)",
            "Trustworthy AI",
        ]
    )

    # --- scale 1 ---------------------------------------------------------
    with state_tab:
        st.subheader("Where in Texas is high-magnitude flow capturable this week?")
        if statewide.empty:
            st.info("No statewide screening output.")
        else:
            display = statewide[
                ["basin_name", "flag", "capture_index", "binding_constraint",
                 "binding_if_captured", "median_excess_af", "q90_excess_af",
                 "expected_capturable_af", "event_probability"]
            ].copy()
            st.dataframe(
                display.style.background_gradient(subset=["capture_index"], cmap="YlGnBu"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Excess volume is reported as a median and a 90th percentile, never a mean: "
                "the predictive distribution is heavy tailed and its mean is not a number to "
                "plan against."
            )
        st.markdown("#### Capture Index by basin over the replay window")
        flag_timeline(history, "twdb_statewide")

    # --- scale 2 ---------------------------------------------------------
    with watershed_tab:
        row = flags[flags["site_id"] == "rolling_plains_gcd"]
        st.subheader("Watershed-scale runoff forecasting for managed aquifer recharge")
        if row.empty:
            st.info("No assessment for this site.")
        else:
            record = row.iloc[0]
            left, right = st.columns([1, 2])
            with left:
                show_flag_card(record)
            with right:
                basin = record["basin_id"]
                group = timeseries[timeseries["basin_id"] == basin].set_index("date")
                if not group.empty:
                    st.markdown("**Discharge and HMF threshold**")
                    st.line_chart(group[["flow_cfs", "hmf_threshold_cfs"]], height=200)
                    st.markdown("**Antecedent storage deficit (SMAP proxy)**")
                    st.line_chart(group[["storage_deficit_index"]], height=150)
        flag_timeline(history, "rolling_plains_gcd")

    # --- scale 3 ---------------------------------------------------------
    with facility_tab:
        row = flags[flags["site_id"] == "kerrville_asr"]
        st.subheader("Operational ASR decision support and scenario evaluation")
        if row.empty:
            st.info("No assessment for this site.")
        else:
            record = row.iloc[0]
            left, right = st.columns([1, 2])
            with left:
                show_flag_card(record)
            with right:
                group = storage[storage["site_id"] == "kerrville_asr"].set_index("date")
                if not group.empty:
                    st.markdown("**Aquifer storage state**")
                    st.area_chart(group[["storage_fraction"]], height=200)
                    st.caption(
                        "Recovery draws the bucket down through summer. Recharge is only "
                        "possible into the headroom that remains."
                    )
        flag_timeline(history, "kerrville_asr")

        st.markdown("#### Scenario evaluation: what would change the answer?")
        if sweep.empty:
            st.info("Run `python scripts/run_pipeline.py --scenario` to generate the sweep.")
        else:
            grid = sweep.pivot_table(
                index="headroom_af", columns="diversion_scale", values="capture_index"
            )
            st.dataframe(
                grid.style.background_gradient(cmap="YlGnBu", vmin=0, vmax=1),
                use_container_width=True,
            )
            st.caption(
                "Rows: available aquifer headroom (AF). Columns: multiplier on intake, "
                "conveyance, treatment, and well count. Each cell is the Capture Index for "
                "the same forecast, which separates a hydrologic limit from a capital one."
            )
            binding = sweep["binding_constraint"].value_counts()
            st.write("Binding constraint across the scenario grid:")
            st.bar_chart(binding)

    # --- trust -----------------------------------------------------------
    with trust_tab:
        st.subheader("What this model does and does not know")
        if cv.empty:
            st.info("No cross-validation output. Run the pipeline without `--fast` or `--no-cv`.")
        else:
            st.markdown("**Leave-one-basin-out cross-validation**")
            st.dataframe(cv, use_container_width=True, hide_index=True)
            picp = cv["picp_80"].mean()
            st.metric("Mean coverage of the nominal 80% interval", f"{picp:.0%}")
            st.caption(
                "Each row holds out an entire basin, which is the relevant test for transfer "
                "to an ungauged site. Random k-fold on daily hydrology would report far better "
                "numbers and mean nothing, because adjacent days are near-duplicates. "
                "Coverage well below 80% would mean the intervals are overconfident."
            )

        importance = load("feature_importance.csv")
        if not importance.empty:
            st.markdown("**Permutation importance (volume head, log space)**")
            st.bar_chart(importance.set_index("feature")["mse_increase"].head(10))

        if not history.empty and "mass_balance_clipped_fraction" in history.columns:
            clipped = history["mass_balance_clipped_fraction"].mean()
            st.metric("Predictive samples clipped by the mass-balance bound", f"{clipped:.1%}")
            st.caption(
                "The learner works in log space, so its upper tail can exceed the water the "
                "catchment physically holds. Those samples are clipped to the antecedent "
                "rainfall bound before any decision is made, and the clipped share is "
                "reported rather than hidden."
            )

        if not events.empty:
            st.markdown("**Retrospective HMF catalogue**")
            st.dataframe(
                events.groupby("basin_name")
                .agg(
                    events=("excess_af", "size"),
                    median_excess_af=("excess_af", "median"),
                    max_excess_af=("excess_af", "max"),
                    mean_duration_days=("duration_days", "mean"),
                )
                .reset_index(),
                use_container_width=True,
                hide_index=True,
            )

    with st.sidebar:
        st.header("Flag system")
        for flag, action in FLAG_ACTIONS.items():
            st.markdown(flag_badge(flag), unsafe_allow_html=True)
            st.caption(action)
        st.divider()
        st.caption(
            "Capture Index = P(feasible capturable volume >= the site's operational "
            "threshold), estimated across the bootstrap ensemble after every member is "
            "pushed through the hydrologic, legal, and infrastructural constraint chain."
        )


if __name__ == "__main__":
    main()
