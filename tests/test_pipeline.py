"""End-to-end pipeline, config loading, aquifer accounting, and downscaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twr import aquifer, downscale, ingest, pipeline, scenarios
from twr.capture_index import FLAG_COLORS, FLAG_ORDER
from twr.config import load_basins, load_sites
from twr.downscale import bilinear_upsample, block_mean, evaluate_downscaling


@pytest.fixture(scope="module")
def basins():
    return load_basins()


@pytest.fixture(scope="module")
def sites():
    return load_sites()


@pytest.fixture(scope="module")
def result(basins, sites):
    """One small pipeline run shared by every end-to-end assertion."""
    config = pipeline.PipelineConfig(
        start="2015-01-01",
        end="2020-12-31",
        n_bootstrap=3,
        baseline_years=3,
        history_days=120,
        run_spatial_cv=False,
    )
    return pipeline.run(config, basins=basins, sites=sites)


# --- configuration -------------------------------------------------------


def test_shipped_config_loads(basins, sites):
    assert len(basins) >= 5
    assert {site.scale for site in sites} == {"state", "watershed", "facility"}
    assert sum(site.is_statewide for site in sites) == 1


def test_config_does_not_claim_real_gauge_numbers(basins):
    """Inventing a plausible gauge site number would be worse than a blank."""
    for basin in basins:
        assert basin.gauge_site_no is None
        assert basin.gauge_id.startswith("SYN-")


def test_infrastructure_numbers_are_marked_illustrative(sites):
    for site in sites:
        assert site.provenance == "illustrative"


def test_every_site_threshold_is_reachable_by_its_own_hardware(sites):
    """Regression test for the bug that pinned two sites at CI = 0 forever."""
    pipeline.validate_site_feasibility(sites, horizon_days=7)


def test_infeasible_site_is_rejected_loudly(sites):
    broken = [
        site.__class__(
            **{
                **site.__dict__,
                "operational_threshold_af": 1e12,
            }
        )
        for site in sites[:1]
    ]
    with pytest.raises(ValueError, match="never leave zero"):
        pipeline.validate_site_feasibility(broken, horizon_days=7)


# --- ingest --------------------------------------------------------------


def test_product_registry_names_an_archive_for_every_stream():
    table = ingest.products_table()
    assert len(table) >= 6
    assert table["archive"].str.len().gt(0).all()
    assert table["product_short_name"].str.len().gt(0).all()


def test_real_ingest_refuses_rather_than_inventing_data(basins):
    with pytest.raises(NotImplementedError, match="Earthdata"):
        ingest.load_observed(basins, "2024-01-01", "2024-02-01")


def test_validate_record_rejects_negative_discharge(basins):
    frame = ingest.load_synthetic(basins[:1], start="2015-01-01", end="2016-12-31")
    broken = frame.copy()
    broken.loc[broken.index[0], "flow_cfs"] = -5.0
    with pytest.raises(ValueError, match="negative discharge"):
        ingest.validate_record(broken)


def test_validate_record_rejects_duplicate_days(basins):
    frame = ingest.load_synthetic(basins[:1], start="2015-01-01", end="2016-12-31")
    with pytest.raises(ValueError, match="duplicate"):
        ingest.validate_record(pd.concat([frame, frame.head(3)], ignore_index=True))


# --- aquifer -------------------------------------------------------------


def test_storage_stays_within_capacity(sites):
    site = next(s for s in sites if s.scale == "facility")
    dates = pd.date_range("2020-01-01", periods=800, freq="D")
    rng = np.random.default_rng(0)
    precip = rng.gamma(0.6, 20.0, len(dates))
    frame = aquifer.simulate_storage(dates, site, precip)
    capacity = site.infrastructure.storage_capacity_af
    assert frame["storage_af"].between(0.0, capacity).all()
    assert np.allclose(frame["storage_af"] + frame["headroom_af"], capacity)


def test_storage_actually_cycles(sites):
    """A bucket pinned at empty or full is not a storage model.

    Regression test: an earlier version scaled natural recharge by a constant
    rather than by capacity, so every site drained to zero in the first year and
    the storage constraint could never bind.
    """
    site = next(s for s in sites if s.scale == "facility")
    dates = pd.date_range("2016-01-01", periods=1_500, freq="D")
    rng = np.random.default_rng(4)
    precip = np.where(rng.random(len(dates)) < 0.2, rng.gamma(0.6, 25.0, len(dates)), 0.0)
    frame = aquifer.simulate_storage(dates, site, precip)

    fraction = frame["storage_fraction"]
    settled = fraction.iloc[365:]  # ignore the spin-up from the initial condition
    assert settled.max() > 0.5, "storage never fills, headroom can never bind"
    assert settled.min() < settled.max() - 0.05, "storage does not cycle"
    assert settled.std() > 0.01


def test_summer_drawdown_is_visible(sites):
    site = next(s for s in sites if s.scale == "facility")
    dates = pd.date_range("2016-01-01", periods=1_460, freq="D")
    precip = np.full(len(dates), 2.5)
    frame = aquifer.simulate_storage(dates, site, precip).iloc[365:]
    months = frame["date"].dt.month
    late_summer = frame.loc[months.isin([8, 9]), "storage_fraction"].mean()
    spring = frame.loc[months.isin([3, 4]), "storage_fraction"].mean()
    assert late_summer < spring


def test_managed_recharge_raises_storage(sites):
    site = next(s for s in sites if s.scale == "facility")
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    precip = np.full(len(dates), 2.0)
    baseline = aquifer.simulate_storage(dates, site, precip)
    with_mar = aquifer.simulate_storage(
        dates, site, precip, captured_af_per_day=np.full(len(dates), 5.0)
    )
    assert with_mar["storage_af"].iloc[-1] > baseline["storage_af"].iloc[-1]


def test_storage_rejects_misaligned_recharge_series(sites):
    site = next(s for s in sites if s.scale == "facility")
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    with pytest.raises(ValueError):
        aquifer.simulate_storage(
            dates, site, np.zeros(100), captured_af_per_day=np.zeros(50)
        )


def test_recovery_demand_peaks_in_summer():
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    demand = aquifer.seasonal_demand_af_per_day(dates, annual_af=1_000.0)
    summer = demand[(dates.month >= 6) & (dates.month <= 8)].mean()
    winter = demand[(dates.month <= 2) | (dates.month == 12)].mean()
    assert summer > winter
    assert demand.sum() == pytest.approx(1_000.0, rel=0.05)


# --- pipeline ------------------------------------------------------------


def test_pipeline_produces_one_assessment_per_site(result, sites):
    assert len(result.assessments) == len(sites)
    assert {a.site_id for a in result.assessments} == {s.id for s in sites}


def test_every_flag_is_known_and_coloured(result):
    valid = set(FLAG_ORDER) | {"BLOCKED"}
    for assessment in result.assessments:
        assert assessment.flag in valid
        assert assessment.flag in FLAG_COLORS
        assert 0.0 <= assessment.capture_index <= 1.0


def test_statewide_screening_covers_every_basin(result, basins):
    assert set(result.statewide["basin_id"]) == {basin.id for basin in basins}
    assert result.statewide["urgency"].is_monotonic_decreasing


def test_history_replays_the_requested_window(result, sites):
    assert not result.history.empty
    span = result.history["date"].max() - result.history["date"].min()
    assert span <= pd.Timedelta(days=120)
    assert set(result.history["site_id"]) == {site.id for site in sites}


def test_capturable_volume_never_exceeds_forecast_excess(result):
    for assessment in result.assessments:
        assert assessment.q50_capturable_af <= assessment.median_excess_af + 1e-6


def test_capture_volume_respects_aquifer_headroom(result):
    for assessment in result.assessments:
        assert assessment.q90_capturable_af <= assessment.storage_headroom_af + 1e-6


def test_mass_balance_bound_is_applied_everywhere(result):
    history = result.history
    assert (history["median_excess_af"] <= history["mass_balance_ceiling_af"] + 1e-6).all()


def test_permit_accounting_never_goes_negative(result, basins, sites):
    site = next(s for s in sites if s.scale == "watershed")
    basin = next(b for b in basins if b.id == site.basin_id)
    remaining = pipeline.permit_remaining_by_date(result.table, basin, site, deplete=True)
    values = np.array(list(remaining.values()))
    assert (values >= 0).all()
    assert values.max() <= basin.water_rights.permitted_diversion_af_per_year


def test_statewide_screening_does_not_deplete_a_permit(result, basins, sites):
    """No single diverter owns a statewide screening, so nothing is drawn down."""
    site = next(s for s in sites if s.is_statewide)
    basin = basins[0]
    remaining = pipeline.permit_remaining_by_date(result.table, basin, site, deplete=False)
    values = set(remaining.values())
    assert values == {basin.water_rights.permitted_diversion_af_per_year}


def test_retrospective_events_are_found(result, basins):
    assert not result.events.empty
    assert set(result.events["basin_id"]) <= {basin.id for basin in basins}
    assert (result.events["excess_af"] > 0).all()
    assert (result.events["peak_cfs"] > result.events["hmf_threshold_cfs"]).all()


def test_write_outputs_creates_the_dashboard_artefacts(result, tmp_path):
    written = pipeline.write_outputs(result, tmp_path)
    for name in (
        "statewide_screening.csv",
        "site_flags.csv",
        "flag_history.csv",
        "daily_timeseries.csv",
        "run_summary.json",
    ):
        assert name in written
        assert written[name].exists()
        assert written[name].stat().st_size > 0


def test_as_of_outside_the_record_is_rejected(basins, sites):
    config = pipeline.PipelineConfig(
        start="2015-01-01", end="2020-12-31", n_bootstrap=2, baseline_years=3,
        history_days=10, run_spatial_cv=False, as_of="2099-01-01",
    )
    with pytest.raises(ValueError, match="no usable feature row"):
        pipeline.run(config, basins=basins[:2], sites=sites[-1:])


def test_scenario_sweep_uses_an_actionable_day(result, basins, sites):
    """A capital-planning sweep run on a quiet day answers nothing.

    Regression test: defaulting to the last day of the record made all 40 cells
    identical, with hydrology binding everywhere.
    """
    site = next(s for s in sites if s.scale == "facility")
    basin = next(b for b in basins if b.id == site.basin_id)
    sweep = scenarios.scenario_sweep(result, site, basin)

    chosen = pd.Timestamp(sweep["as_of"].iloc[0])
    history = result.history
    subset = history[(history["site_id"] == site.id) & (history["basin_id"] == basin.id)]
    assert chosen == pd.Timestamp(subset.loc[subset["capture_index"].idxmax(), "date"])
    # The grid must actually discriminate between scenarios.
    assert sweep["capture_index"].nunique() > 1


def test_scenario_sweep_honours_an_explicit_date(result, basins, sites):
    site = next(s for s in sites if s.scale == "facility")
    basin = next(b for b in basins if b.id == site.basin_id)
    target = result.history["date"].max()
    sweep = scenarios.scenario_sweep(
        result, site, basin,
        headroom_grid=np.array([1_000.0]), diversion_scale_grid=np.array([1.0]),
        as_of=target,
    )
    assert sweep["as_of"].iloc[0] == str(pd.Timestamp(target).date())


def test_scenario_sweep_rejects_a_date_outside_the_record(result, basins, sites):
    site = next(s for s in sites if s.scale == "facility")
    basin = next(b for b in basins if b.id == site.basin_id)
    with pytest.raises(ValueError, match="no usable feature row"):
        scenarios.scenario_sweep(result, site, basin, as_of="2099-01-01")


def test_scenario_sweep_shows_more_wells_helping(result, basins, sites):
    site = next(s for s in sites if s.scale == "facility")
    basin = next(b for b in basins if b.id == site.basin_id)
    sweep = scenarios.scenario_sweep(
        result, site, basin,
        headroom_grid=np.array([site.infrastructure.storage_capacity_af]),
        diversion_scale_grid=np.array([1.0, 4.0]),
    )
    assert len(sweep) == 2
    small, large = sweep.iloc[0], sweep.iloc[1]
    assert large["expected_capturable_af"] >= small["expected_capturable_af"]


def test_scenario_sweep_shows_a_full_aquifer_hurting(result, basins, sites):
    site = next(s for s in sites if s.scale == "facility")
    basin = next(b for b in basins if b.id == site.basin_id)
    sweep = scenarios.scenario_sweep(
        result, site, basin,
        headroom_grid=np.array([0.0, site.infrastructure.storage_capacity_af]),
        diversion_scale_grid=np.array([1.0]),
    )
    assert sweep.iloc[0]["capture_index"] <= sweep.iloc[1]["capture_index"]
    assert sweep.iloc[0]["expected_capturable_af"] == pytest.approx(0.0)


# --- downscaling ---------------------------------------------------------


def test_bilinear_upsample_preserves_a_constant_field():
    field = np.full((4, 4), 3.0)
    assert np.allclose(bilinear_upsample(field, 4), 3.0)


def test_bilinear_upsample_changes_shape_correctly():
    assert bilinear_upsample(np.zeros((8, 8)), 4).shape == (32, 32)
    field = np.arange(9.0).reshape(3, 3)
    assert np.allclose(bilinear_upsample(field, 1), field)


def test_block_mean_inverts_a_uniform_upsample():
    coarse = np.arange(16.0).reshape(4, 4)
    assert np.allclose(block_mean(np.repeat(np.repeat(coarse, 4, 0), 4, 1), 4), coarse)


def test_block_mean_rejects_indivisible_shapes():
    with pytest.raises(ValueError):
        block_mean(np.zeros((5, 5)), 2)


def test_paired_fields_are_physically_consistent():
    coarse, fine, terrain = downscale.make_paired_fields(
        n_samples=4, fine_size=32, factor=8, seed=1
    )
    assert coarse.shape == (4, 4, 4)
    assert fine.shape == (4, 32, 32)
    assert terrain.shape == (32, 32)
    assert (fine >= 0).all()
    # The coarse product is the block mean of the truth, by construction.
    assert np.allclose(block_mean(fine[0], 8), coarse[0])


def test_learned_downscaling_beats_interpolation():
    """The claim the module exists to support."""
    metrics = evaluate_downscaling(n_train=24, n_test=8, fine_size=32, factor=8, seed=0)
    assert metrics.rmse_model < metrics.rmse_interpolated
    assert metrics.skill_score > 0.0
