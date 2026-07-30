"""Leakage guards on the feature builder, plus model and uncertainty behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twr import features, synth
from twr.config import Basin, EnvironmentalFlow, WaterRights
from twr.model import BootstrapEnsemble, HybridCaptureModel, ModelConfig, default_regressor
from twr.uncertainty import coverage, leave_one_basin_out, quantiles, summarise_folds


@pytest.fixture(scope="module")
def basin():
    return Basin(
        id="test_basin",
        name="Test",
        gauge_id="SYN-TEST-01",
        drainage_area_km2=20_000,
        mean_annual_precip_mm=800,
        hmf_percentile=95.0,
        eflow=EnvironmentalFlow(subsistence_cfs=20, base_cfs=80, pulse_protection_fraction=0.35),
        water_rights=WaterRights(
            unappropriated_fraction=0.25, permitted_diversion_af_per_year=20_000
        ),
    )


@pytest.fixture(scope="module")
def record(basin):
    return synth.simulate_basin(basin, start="2015-01-01", end="2021-12-31", seed=3)


@pytest.fixture(scope="module")
def table(basin, record):
    return features.build_basin_features(record, basin.hmf_percentile)


def test_synthetic_record_has_the_sensor_columns(record):
    for column in synth.SENSOR_COLUMNS:
        assert column in record.columns
    assert (record["flow_cfs"] >= 0).all()
    assert record["soil_moisture"].between(0.0, 0.5).all()


def test_synthetic_evapotranspiration_has_realistic_gaps(record):
    gap_fraction = record["et_mm"].isna().mean()
    assert 0.15 < gap_fraction < 0.45


def test_synthetic_record_is_reproducible(basin):
    first = synth.simulate_basin(basin, start="2015-01-01", end="2016-12-31", seed=11)
    second = synth.simulate_basin(basin, start="2015-01-01", end="2016-12-31", seed=11)
    pd.testing.assert_frame_equal(first, second)


def test_basin_seed_is_stable_across_processes():
    """Reproducibility must survive PYTHONHASHSEED, not just a single process.

    Regression test: the seed was originally derived from the builtin ``hash()``
    of the basin id, which is salted per interpreter. Every fresh run produced
    different data, and no in-process test could detect it.
    """
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    program = (
        f"import sys; sys.path.insert(0, {src!r});"
        "from twr.synth import basin_seed;"
        "print(basin_seed('brazos', 0), basin_seed('guadalupe', 7))"
    )
    outputs = set()
    for salt in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"},
        )
        outputs.add(completed.stdout.strip())
    assert len(outputs) == 1, f"seed depends on PYTHONHASHSEED: {outputs}"


def test_different_basins_get_different_weather(basin):
    """Two basins must not share a hydrograph."""
    other = Basin(**{**basin.__dict__, "id": "other_basin", "name": "Other"})
    first = synth.simulate_basin(basin, start="2015-01-01", end="2016-12-31", seed=0)
    second = synth.simulate_basin(other, start="2015-01-01", end="2016-12-31", seed=0)
    assert not np.allclose(first["flow_cfs"], second["flow_cfs"])


def test_wet_soil_amplifies_runoff(record):
    """The relationship the model has to learn must actually be present."""
    wet = record["_soil_store_mm"] > record["_soil_store_mm"].quantile(0.75)
    dry = record["_soil_store_mm"] < record["_soil_store_mm"].quantile(0.25)
    storms = record["_precip_true_mm"] > 5.0
    wet_response = record.loc[wet & storms, "_flow_true_cfs"].median()
    dry_response = record.loc[dry & storms, "_flow_true_cfs"].median()
    assert wet_response > dry_response


def test_targets_are_strictly_forward_looking(table):
    """The 7-day target at t must equal the realised excess over t+1..t+7."""
    excess = table["daily_excess_af"].to_numpy()
    target = table[features.TARGET_VOLUME].to_numpy()
    for index in (100, 500, 1_200, 2_000):
        assert target[index] == pytest.approx(excess[index + 1 : index + 8].sum())


def test_event_target_matches_the_volume_target(table):
    clean = table.dropna(subset=[features.TARGET_VOLUME, features.TARGET_EVENT])
    has_volume = clean[features.TARGET_VOLUME] > 0
    is_event = clean[features.TARGET_EVENT] > 0
    # Any forward window with excess volume must be flagged as an event.
    assert (has_volume <= is_event).all()


def test_tail_rows_have_no_target(table):
    assert table[features.TARGET_VOLUME].tail(7).isna().all()


def test_rolling_features_use_only_the_past(table):
    precip = table["precip_mm"].to_numpy()
    for index in (50, 400, 1_500):
        assert table["precip_7d"].to_numpy()[index] == pytest.approx(
            precip[index - 6 : index + 1].sum()
        )


def test_antecedent_index_is_causal():
    values = np.array([0.0, 10.0, 0.0, 0.0, 5.0])
    api = features._antecedent_precip_index(values, 0.9)
    # A spike cannot affect earlier days, and must decay afterwards.
    assert api[0] == 0.0
    assert api[1] == pytest.approx(10.0)
    assert api[2] == pytest.approx(9.0)
    assert api[3] == pytest.approx(8.1)


def test_storage_deficit_is_bounded_and_inverted(table):
    deficit = table["storage_deficit_index"]
    assert deficit.between(0.0, 1.0).all()
    # Wetter soil must mean less room for water.
    assert table["soil_moisture"].corr(deficit) < -0.9


def test_threshold_comes_from_the_baseline_period_only(basin, record):
    spec = features.FeatureSpec(baseline_years=3)
    table = features.build_basin_features(record, basin.hmf_percentile, spec)
    threshold = float(table["hmf_threshold_cfs"].iloc[0])
    baseline = record[record["date"] < record["date"].iloc[0] + pd.DateOffset(years=3)]
    assert threshold == pytest.approx(np.percentile(baseline["flow_cfs"], 95.0))
    assert table["hmf_threshold_cfs"].nunique() == 1


def test_baseline_shorter_than_a_year_is_rejected(basin):
    short = synth.simulate_basin(basin, start="2015-01-01", end="2015-06-30", seed=1)
    with pytest.raises(ValueError):
        features.build_basin_features(short, basin.hmf_percentile)


def test_duplicate_dates_are_rejected(basin, record):
    doubled = pd.concat([record, record.head(10)], ignore_index=True)
    with pytest.raises(ValueError):
        features.build_basin_features(doubled, basin.hmf_percentile)


def test_training_matrix_drops_incomplete_rows(table):
    X, y_volume, y_event, groups = features.training_matrix(table)
    assert len(X) == len(y_volume) == len(y_event) == len(groups)
    assert not X.isna().any().any()
    assert (y_volume >= 0).all()
    assert set(y_event.unique()) <= {0, 1}


def test_bootstrap_ensemble_members_disagree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=400)
    ensemble = BootstrapEnsemble(default_regressor, n_bootstrap=5, random_state=0).fit(X, y)
    samples = ensemble.predict_samples(X)
    assert samples.shape == (400, 5)
    assert samples.std(axis=1).mean() > 0


def test_bootstrap_ensemble_rejects_a_single_member():
    with pytest.raises(ValueError):
        BootstrapEnsemble(default_regressor, n_bootstrap=1)


def test_unfitted_ensemble_raises():
    with pytest.raises(RuntimeError):
        BootstrapEnsemble(default_regressor, n_bootstrap=3).predict_samples(np.zeros((2, 2)))


def test_residual_sampling_widens_the_predictive_distribution(table):
    X, y_volume, y_event, _ = features.training_matrix(table)
    subset = slice(0, 1_500)
    model = HybridCaptureModel(ModelConfig(n_bootstrap=4, residual_draws=4)).fit(
        X[subset], y_volume[subset], y_event[subset]
    )
    epistemic = model.predict_volume_samples(X[subset], include_aleatoric=False)
    full = model.predict_volume_samples(X[subset], include_aleatoric=True)
    narrow = quantiles(epistemic, (0.1, 0.9))
    wide = quantiles(full, (0.1, 0.9))
    epistemic_width = float(np.mean(narrow["q90"] - narrow["q10"]))
    full_width = float(np.mean(wide["q90"] - wide["q10"]))
    assert full_width > epistemic_width
    assert full.shape[1] == 4 * 4


def test_model_rejects_negative_volume_targets(table):
    X, y_volume, y_event, _ = features.training_matrix(table)
    with pytest.raises(ValueError):
        HybridCaptureModel(ModelConfig(n_bootstrap=2)).fit(X[:200], -y_volume[:200], y_event[:200])


def test_model_rejects_missing_features_at_prediction_time(table):
    X, y_volume, y_event, _ = features.training_matrix(table)
    model = HybridCaptureModel(ModelConfig(n_bootstrap=2)).fit(
        X[:500], y_volume[:500], y_event[:500]
    )
    with pytest.raises(ValueError):
        model.predict_volume_samples(X[:5].drop(columns=["precip_7d"]))


def test_event_probabilities_are_probabilities(table):
    X, y_volume, y_event, _ = features.training_matrix(table)
    model = HybridCaptureModel(ModelConfig(n_bootstrap=3)).fit(
        X[:1_500], y_volume[:1_500], y_event[:1_500]
    )
    probabilities = model.predict_event_probability(X[:200])
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_quantiles_reject_a_one_dimensional_array():
    with pytest.raises(ValueError):
        quantiles(np.zeros(10))


def test_coverage_counts_points_inside_the_interval():
    y = np.array([1.0, 5.0, 9.0])
    assert coverage(y, np.zeros(3), np.full(3, 10.0)) == 1.0
    assert coverage(y, np.full(3, 6.0), np.full(3, 10.0)) == pytest.approx(1 / 3)


def test_spatial_cross_validation_holds_out_whole_basins(basin):
    basins = []
    for index, name in enumerate(["alpha", "beta", "gamma"]):
        basins.append(
            Basin(
                id=name, name=name.title(), gauge_id=f"SYN-{name[:3].upper()}-01",
                drainage_area_km2=10_000 * (index + 1),
                mean_annual_precip_mm=700 + 100 * index,
                hmf_percentile=95.0, eflow=basin.eflow, water_rights=basin.water_rights,
            )
        )
    records = synth.simulate_all(basins, start="2015-01-01", end="2019-12-31", seed=5)
    table = features.build_features(
        records, {b.id: b.hmf_percentile for b in basins}, features.FeatureSpec(baseline_years=2)
    )
    X, y_volume, y_event, groups = features.training_matrix(table)
    folds = leave_one_basin_out(X, y_volume, y_event, groups, ModelConfig(n_bootstrap=3))

    assert len(folds) == 3
    assert set(folds["fold"]) == {"alpha", "beta", "gamma"}
    # Every fold must be trained without its test basin.
    for _, row in folds.iterrows():
        assert row["n_train"] + row["n_test"] == len(X)
        assert 0.0 <= row["picp_80"] <= 1.0

    summary = summarise_folds(folds)
    assert summary["n_folds"] == 3
    assert "picp_80_mean" in summary
