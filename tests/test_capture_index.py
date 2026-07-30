"""Capture Index semantics and the flag ladder."""

from __future__ import annotations

import numpy as np
import pytest

from twr.capture_index import (
    FLAG_BLOCKED,
    FLAG_ORDER,
    FLAG_THRESHOLDS,
    assess,
    assign_flag,
    flag_rank,
)
from twr.config import EnvironmentalFlow, Infrastructure, WaterRights


@pytest.fixture
def setting():
    return {
        "eflow": EnvironmentalFlow(
            subsistence_cfs=30, base_cfs=110, pulse_protection_fraction=0.40
        ),
        "rights": WaterRights(
            unappropriated_fraction=0.50, permitted_diversion_af_per_year=100_000
        ),
        "infra": Infrastructure(
            max_diversion_cfs=500, conveyance_cfs=500, treatment_mgd=200,
            recharge_wells=40, well_capacity_gpm=1_000, storage_capacity_af=50_000,
        ),
    }


def _assess(samples, setting, headroom=50_000.0, threshold=100.0, **kwargs):
    return assess(
        site_id="test_site",
        basin_id="test_basin",
        date="2025-06-01",
        volume_samples_af=np.asarray(samples, dtype=float),
        event_probability=0.5,
        eflow=setting["eflow"],
        rights=setting["rights"],
        infra=setting["infra"],
        storage_headroom_af=headroom,
        operational_threshold_af=threshold,
        horizon_days=7,
        **kwargs,
    )


def test_flag_ladder_is_ordered():
    assert assign_flag(0.05, "none") == "NO_ACTION"
    assert assign_flag(0.30, "none") == "WATCH"
    assert assign_flag(0.55, "none") == "STANDBY"
    assert assign_flag(0.95, "none") == "CAPTURE"


def test_flag_boundaries_are_inclusive_from_below():
    for flag, threshold in FLAG_THRESHOLDS.items():
        assert assign_flag(threshold, "none") == flag


def test_blocked_overrides_no_action_when_a_hard_limit_binds():
    assert assign_flag(0.0, "storage_headroom") == FLAG_BLOCKED
    assert assign_flag(0.0, "annual_permit") == FLAG_BLOCKED
    # Hydrology is not a blockage, it is an absence of water.
    assert assign_flag(0.0, "no_hmf") == "NO_ACTION"
    assert assign_flag(0.0, "hydrologic_availability") == "NO_ACTION"


def test_blocked_does_not_override_an_actionable_index():
    assert assign_flag(0.80, "storage_headroom") == "CAPTURE"


def test_assign_flag_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        assign_flag(1.5, "none")


def test_flag_rank_orders_urgency():
    ranks = [flag_rank(flag) for flag in FLAG_ORDER]
    assert ranks == sorted(ranks)
    assert flag_rank("NO_ACTION") < flag_rank(FLAG_BLOCKED) < flag_rank("WATCH")


def test_flag_rank_rejects_unknown_flag():
    with pytest.raises(ValueError):
        flag_rank("MAYBE")


def test_capture_index_is_a_probability_over_feasible_volume(setting):
    # Every sample is large, so every sample clears the threshold after the chain.
    result = _assess(np.full(100, 1e6), setting)
    assert result.capture_index == pytest.approx(1.0)
    assert result.flag == "CAPTURE"


def test_capture_index_is_zero_when_no_water_is_forecast(setting):
    result = _assess(np.zeros(100), setting)
    assert result.capture_index == 0.0
    assert result.binding_constraint == "no_hmf"
    assert result.flag == "NO_ACTION"


def test_capture_index_reflects_the_fraction_of_members_that_clear(setting):
    # Half the members forecast a large flood, half forecast nothing.
    samples = np.concatenate([np.full(50, 1e6), np.zeros(50)])
    result = _assess(samples, setting)
    assert result.capture_index == pytest.approx(0.5)
    assert result.flag == "STANDBY"


def test_a_full_aquifer_blocks_a_real_flood(setting):
    """The point of the index: hydrology alone does not make an opportunity."""
    wet = _assess(np.full(50, 1e6), setting, headroom=50_000.0)
    full = _assess(np.full(50, 1e6), setting, headroom=0.0)
    assert wet.flag == "CAPTURE"
    assert full.capture_index == 0.0
    assert full.flag == FLAG_BLOCKED
    assert full.binding_constraint == "storage_headroom"


def test_index_is_monotone_in_storage_headroom(setting):
    samples = np.random.default_rng(0).lognormal(6.0, 1.5, 200)
    indices = [
        _assess(samples, setting, headroom=head).capture_index
        for head in [0, 50, 200, 1_000, 10_000]
    ]
    assert indices == sorted(indices)


def test_index_is_monotone_in_threshold(setting):
    samples = np.random.default_rng(1).lognormal(7.0, 1.2, 200)
    indices = [
        _assess(samples, setting, threshold=threshold).capture_index
        for threshold in [10, 100, 1_000, 10_000]
    ]
    assert indices == sorted(indices, reverse=True)


def test_quantiles_are_ordered(setting):
    samples = np.random.default_rng(2).lognormal(6.0, 1.5, 300)
    result = _assess(samples, setting)
    assert result.q10_capturable_af <= result.q50_capturable_af <= result.q90_capturable_af


def test_excess_is_reported_as_median_and_upper_quantile(setting):
    samples = np.random.default_rng(3).lognormal(5.0, 2.0, 500)
    result = _assess(samples, setting)
    assert result.median_excess_af == pytest.approx(np.median(samples))
    assert result.q90_excess_af >= result.median_excess_af


def test_mass_balance_ceiling_clips_impossible_samples(setting):
    """A learner tail beyond what the catchment holds must not reach the index."""
    samples = np.full(100, 1e9)
    unbounded = _assess(samples, setting, threshold=100.0)
    bounded = _assess(samples, setting, threshold=100.0, mass_balance_ceiling_af=50.0)
    assert unbounded.median_excess_af == pytest.approx(1e9)
    assert bounded.median_excess_af == pytest.approx(50.0)
    assert bounded.mass_balance_clipped_fraction == pytest.approx(1.0)
    # 50 AF of excess cannot yield 100 AF of capture.
    assert bounded.capture_index == 0.0


def test_clipped_fraction_is_zero_when_the_bound_is_slack(setting):
    result = _assess(np.full(10, 5.0), setting, mass_balance_ceiling_af=1e9)
    assert result.mass_balance_clipped_fraction == 0.0


def test_assess_rejects_empty_ensemble(setting):
    with pytest.raises(ValueError):
        _assess(np.array([]), setting)


def test_assess_rejects_non_positive_threshold(setting):
    with pytest.raises(ValueError):
        _assess(np.full(10, 100.0), setting, threshold=0.0)


def test_assess_rejects_negative_ceiling(setting):
    with pytest.raises(ValueError):
        _assess(np.full(10, 100.0), setting, mass_balance_ceiling_af=-1.0)


def test_assessment_serialises_for_the_dashboard(setting):
    payload = _assess(np.full(10, 5_000.0), setting).to_dict()
    for key in ("capture_index", "flag", "action", "binding_constraint", "limits_af"):
        assert key in payload
    assert isinstance(payload["limits_af"], dict)
