"""The binding constraint must be actionable, not an artefact of the median.

Regression tests for three ways earlier versions misreported it:

1. A quiet river was labelled `water_rights`, because the legal limits are
   proportional to availability and so are always smallest when there is no
   water. Fixed by the materiality floor.
2. A `WATCH` flag was paired with `no_hmf`, because the median of a heavy-tailed
   forecast is zero even when a quarter of the ensemble shows a real flood.
   Fixed by adding `binding_if_captured`.
3. Evaluating the *flag-driving* constraint conditionally instead labelled 1701
   of 2555 statewide days `BLOCKED`, because a legal limit binds the flood case
   almost every day. Fixed by keeping `binding_constraint` unconditional.

The two fields answer two different questions and both are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from twr.capture_index import assess
from twr.config import EnvironmentalFlow, Infrastructure, WaterRights


@pytest.fixture
def setting():
    return {
        "eflow": EnvironmentalFlow(
            subsistence_cfs=30, base_cfs=110, pulse_protection_fraction=0.40
        ),
        "rights": WaterRights(
            unappropriated_fraction=0.50, permitted_diversion_af_per_year=1e9
        ),
        # Recharge is the tight link: 4 wells x 450 gpm over 7 days = 56 AF.
        "infra": Infrastructure(
            max_diversion_cfs=5_000, conveyance_cfs=5_000, treatment_mgd=5_000,
            recharge_wells=4, well_capacity_gpm=450, storage_capacity_af=100_000,
        ),
    }


def _assess(samples, setting, threshold=30.0, headroom=100_000.0):
    return assess(
        site_id="site", basin_id="basin", date="2025-05-22",
        volume_samples_af=np.asarray(samples, dtype=float),
        event_probability=0.5,
        eflow=setting["eflow"], rights=setting["rights"], infra=setting["infra"],
        storage_headroom_af=headroom, operational_threshold_af=threshold,
        horizon_days=7,
    )


def test_actionable_flag_says_what_to_prepare_for(setting):
    """A mostly-dry ensemble with a real flood tail."""
    samples = np.concatenate([np.zeros(70), np.full(30, 500_000.0)])
    result = _assess(samples, setting)

    assert result.capture_index == pytest.approx(0.30)
    assert result.flag == "WATCH"
    # The planning case genuinely has no water, and says so.
    assert result.median_excess_af == 0.0
    assert result.binding_constraint == "no_hmf"
    # But if the flood arrives, the well field is what limits the response.
    assert result.binding_if_captured == "recharge_capacity"


def test_the_conditional_constraint_tracks_the_real_bottleneck(setting):
    """With a nearly full aquifer, the same tail must point at storage instead."""
    samples = np.concatenate([np.zeros(70), np.full(30, 500_000.0)])
    result = _assess(samples, setting, headroom=40.0)
    assert result.binding_if_captured == "storage_headroom"


def test_a_low_index_from_a_thin_tail_is_not_blocked(setting):
    """BLOCKED must mean 'water is there and something stops you', not 'no water'.

    Labelling every day whose flood case is legally limited as BLOCKED buried the
    real blockages under 1701 false ones.
    """
    samples = np.concatenate([np.zeros(95), np.full(5, 500_000.0)])
    result = _assess(samples, setting)
    assert result.capture_index == pytest.approx(0.05)
    assert result.flag == "NO_ACTION"
    assert result.binding_if_captured == "recharge_capacity"


def test_a_real_blockage_is_still_reported(setting):
    """Water present in the planning case, aquifer full: this is a true BLOCKED."""
    result = _assess(np.full(100, 500_000.0), setting, headroom=0.0)
    assert result.capture_index == 0.0
    assert result.flag == "BLOCKED"
    assert result.binding_constraint == "storage_headroom"


def test_a_genuinely_dry_forecast_still_reports_hydrology(setting):
    """When nothing clears the threshold, the honest answer is 'no water'."""
    result = _assess(np.zeros(100), setting)
    assert result.capture_index == 0.0
    assert result.binding_constraint == "no_hmf"
    assert result.flag == "NO_ACTION"


def test_a_small_forecast_reports_hydrologic_availability(setting):
    result = _assess(np.full(100, 2.0), setting, threshold=30.0)
    assert result.capture_index == 0.0
    assert result.binding_constraint == "hydrologic_availability"
    assert result.flag == "NO_ACTION"


def test_binding_constraint_names_the_hardware_that_can_be_bought(setting):
    """A large flood into an empty aquifer: the well field is the limit."""
    result = _assess(np.full(100, 1e6), setting)
    assert result.flag == "CAPTURE"
    assert result.binding_constraint == "recharge_capacity"
    # With no uncertainty in the ensemble, both views must agree.
    assert result.binding_if_captured == result.binding_constraint
    # And the reported capture volume equals that limit exactly.
    assert result.q50_capturable_af == pytest.approx(result.limits_af["recharge_capacity"])


def test_limits_are_all_reported_so_the_second_constraint_is_visible(setting):
    """An operator needs to know what binds next if they fix the first limit."""
    result = _assess(np.full(100, 1e6), setting)
    limits = result.limits_af
    assert limits["recharge_capacity"] < limits["treatment"]
    assert min(limits.values()) == pytest.approx(limits["recharge_capacity"])
