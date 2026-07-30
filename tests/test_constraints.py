"""The constraint chain: mass balance, environmental flow, rights, hardware."""

from __future__ import annotations

import numpy as np
import pytest

from twr.config import EnvironmentalFlow, Infrastructure, WaterRights
from twr.constraints import (
    apply_constraints,
    apply_constraints_vectorised,
    check_mass_balance,
    infrastructure_capacity_af,
    mass_balance_ceiling_af,
)
from twr.units import CFS_DAY_TO_AF, GPM_DAY_TO_AF, MGD_TO_AF_PER_DAY


@pytest.fixture
def eflow():
    return EnvironmentalFlow(subsistence_cfs=30, base_cfs=110, pulse_protection_fraction=0.40)


@pytest.fixture
def rights():
    return WaterRights(unappropriated_fraction=0.50, permitted_diversion_af_per_year=100_000)


@pytest.fixture
def infra():
    return Infrastructure(
        max_diversion_cfs=1_000,
        conveyance_cfs=1_000,
        treatment_mgd=1_000,
        recharge_wells=100,
        well_capacity_gpm=5_000,
        storage_capacity_af=1_000_000,
    )


def test_environmental_flow_reserves_its_share(eflow, rights, infra):
    result = apply_constraints(
        1_000.0, eflow=eflow, rights=rights, infra=infra,
        storage_headroom_af=1e9, days=7,
    )
    # 1000 AF -> 60% survives eflow -> 50% of that is unappropriated.
    assert result.capturable_af == pytest.approx(1_000 * 0.60 * 0.50)


def test_capture_never_exceeds_available_water(eflow, rights, infra):
    for excess in (0.0, 1.0, 500.0, 1e6, 1e9):
        result = apply_constraints(
            excess, eflow=eflow, rights=rights, infra=infra,
            storage_headroom_af=1e12, days=7,
        )
        assert result.capturable_af <= excess + 1e-9
        check_mass_balance(excess, result.capturable_af, result.passed_to_river_af)


def test_negative_excess_is_clamped(eflow, rights, infra):
    result = apply_constraints(
        -50.0, eflow=eflow, rights=rights, infra=infra, storage_headroom_af=1e6
    )
    assert result.capturable_af == 0.0
    assert result.binding == "no_hmf"


def test_storage_headroom_binds_and_is_labelled(eflow, rights, infra):
    result = apply_constraints(
        1e6, eflow=eflow, rights=rights, infra=infra, storage_headroom_af=25.0, days=7
    )
    assert result.capturable_af == pytest.approx(25.0)
    assert result.binding == "storage_headroom"


def test_full_aquifer_yields_zero_capture(eflow, rights, infra):
    result = apply_constraints(
        1e6, eflow=eflow, rights=rights, infra=infra, storage_headroom_af=0.0
    )
    assert result.capturable_af == 0.0
    assert result.binding == "storage_headroom"


def test_recharge_capacity_can_be_the_binding_limit(eflow, rights):
    small = Infrastructure(
        max_diversion_cfs=1_000, conveyance_cfs=1_000, treatment_mgd=1_000,
        recharge_wells=4, well_capacity_gpm=450, storage_capacity_af=10_000,
    )
    result = apply_constraints(
        1e6, eflow=eflow, rights=rights, infra=small, storage_headroom_af=10_000, days=7
    )
    assert result.binding == "recharge_capacity"
    assert result.capturable_af == pytest.approx(4 * 450 * 7 * GPM_DAY_TO_AF)


def test_small_excess_is_attributed_to_hydrology_not_law(eflow, rights, infra):
    """A quiet river must not be reported as a water-rights problem."""
    result = apply_constraints(
        5.0, eflow=eflow, rights=rights, infra=infra,
        storage_headroom_af=1e6, days=7, materiality_af=100.0,
    )
    assert result.binding == "hydrologic_availability"


def test_material_excess_is_attributed_to_the_real_limit(eflow, rights, infra):
    result = apply_constraints(
        1e5, eflow=eflow, rights=rights, infra=infra,
        storage_headroom_af=40.0, days=7, materiality_af=100.0,
    )
    assert result.binding == "storage_headroom"


def test_infrastructure_capacities_match_unit_conversions():
    infra = Infrastructure(
        max_diversion_cfs=10, conveyance_cfs=20, treatment_mgd=5,
        recharge_wells=3, well_capacity_gpm=100, storage_capacity_af=1_000,
    )
    caps = infrastructure_capacity_af(infra, days=7)
    assert caps["diversion"] == pytest.approx(10 * 7 * CFS_DAY_TO_AF)
    assert caps["conveyance"] == pytest.approx(20 * 7 * CFS_DAY_TO_AF)
    assert caps["treatment"] == pytest.approx(5 * 7 * MGD_TO_AF_PER_DAY)
    assert caps["recharge"] == pytest.approx(3 * 100 * 7 * GPM_DAY_TO_AF)


def test_infrastructure_capacity_rejects_zero_days():
    infra = Infrastructure(1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        infrastructure_capacity_af(infra, days=0)


def test_vectorised_matches_scalar(eflow, rights, infra):
    samples = np.array([0.0, 10.0, 1_000.0, 50_000.0, 1e7])
    vector = apply_constraints_vectorised(
        samples, eflow=eflow, rights=rights, infra=infra,
        storage_headroom_af=5_000.0, days=7,
    )
    scalar = [
        apply_constraints(
            value, eflow=eflow, rights=rights, infra=infra,
            storage_headroom_af=5_000.0, days=7,
        ).capturable_af
        for value in samples
    ]
    assert vector == pytest.approx(scalar)


def test_capture_is_monotone_in_available_water(eflow, rights, infra):
    values = [
        apply_constraints(
            excess, eflow=eflow, rights=rights, infra=infra,
            storage_headroom_af=1e9, days=7,
        ).capturable_af
        for excess in [0, 100, 1_000, 10_000, 100_000]
    ]
    assert values == sorted(values)


def test_mass_balance_ceiling_scales_with_rain_and_area():
    small = mass_balance_ceiling_af(50.0, 1_000.0, max_runoff_fraction=0.8)
    doubled_rain = mass_balance_ceiling_af(100.0, 1_000.0, max_runoff_fraction=0.8)
    doubled_area = mass_balance_ceiling_af(50.0, 2_000.0, max_runoff_fraction=0.8)
    assert doubled_rain == pytest.approx(2 * small)
    assert doubled_area == pytest.approx(2 * small)


def test_mass_balance_ceiling_cannot_exceed_rainfall():
    depth_mm, area_km2 = 80.0, 5_000.0
    from twr.units import mm_over_area_to_af

    assert mass_balance_ceiling_af(depth_mm, area_km2, 1.0) == pytest.approx(
        mm_over_area_to_af(depth_mm, area_km2)
    )
    assert mass_balance_ceiling_af(depth_mm, area_km2, 0.5) < mm_over_area_to_af(depth_mm, area_km2)


def test_mass_balance_ceiling_rejects_bad_runoff_fraction():
    with pytest.raises(ValueError):
        mass_balance_ceiling_af(10.0, 10.0, max_runoff_fraction=1.5)


def test_check_mass_balance_rejects_over_capture():
    with pytest.raises(AssertionError):
        check_mass_balance(excess_af=100.0, capturable_af=150.0, passed_af=-50.0)


def test_pulse_protection_of_one_blocks_everything(rights, infra):
    strict = EnvironmentalFlow(
        subsistence_cfs=10, base_cfs=50, pulse_protection_fraction=1.0
    )
    result = apply_constraints(
        1e6, eflow=strict, rights=rights, infra=infra, storage_headroom_af=1e9
    )
    assert result.capturable_af == pytest.approx(0.0)
    assert result.binding == "eflow_pulse"


def test_environmental_flow_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        EnvironmentalFlow(subsistence_cfs=1, base_cfs=2, pulse_protection_fraction=1.5)
