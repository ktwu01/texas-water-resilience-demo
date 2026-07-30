"""Unit conversions and HMF event identification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twr import hmf, units


def test_cfs_day_to_af_matches_hand_calculation():
    # 1 cfs for 1 day = 86,400 ft3; 1 acre-foot = 43,560 ft3.
    assert units.CFS_DAY_TO_AF == pytest.approx(86_400 / 43_560, rel=1e-12)
    assert units.cfs_days_to_af(100, 7) == pytest.approx(1388.43, rel=1e-4)


def test_flow_volume_round_trip():
    volume = units.cfs_days_to_af(250.0, 3.0)
    assert units.af_to_cfs(volume, 3.0) == pytest.approx(250.0)


def test_af_to_cfs_rejects_zero_duration():
    with pytest.raises(ValueError):
        units.af_to_cfs(100.0, 0.0)


def test_mm_over_area_to_af():
    # 1 mm over 1 km2 = 1000 m3 = 0.8107 AF
    assert units.mm_over_area_to_af(1.0, 1.0) == pytest.approx(0.810713, rel=1e-4)


def test_mm_to_cfs_consistent_with_daily_volume():
    # A depth rate converted to cfs, then to a daily volume, must equal the
    # direct depth-to-volume conversion.
    depth, area = 5.0, 1_000.0
    via_flow = units.cfs_days_to_af(units.mm_to_cfs(depth, area), 1.0)
    direct = units.mm_over_area_to_af(depth, area)
    assert via_flow == pytest.approx(direct, rel=1e-6)


def test_hmf_threshold_is_a_high_percentile():
    flow = np.arange(1.0, 101.0)
    assert hmf.hmf_threshold(flow, 95.0) == pytest.approx(np.percentile(flow, 95))


def test_hmf_threshold_rejects_low_percentile():
    with pytest.raises(ValueError):
        hmf.hmf_threshold(np.arange(1.0, 10.0), 25.0)


def test_excess_volume_is_zero_below_threshold():
    excess = hmf.excess_volume_af(np.array([10.0, 50.0, 200.0]), 100.0)
    assert excess[0] == 0.0
    assert excess[1] == 0.0
    assert excess[2] == pytest.approx(100.0 * units.CFS_DAY_TO_AF)


def test_identify_events_merges_short_recessions():
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    # Two peaks separated by a single below-threshold day: one event, not two.
    flow = np.array([10, 10, 500, 600, 50, 700, 800, 10, 10, 10, 10, 10], dtype=float)
    events = hmf.identify_events(dates, flow, threshold_cfs=100.0, merge_gap_days=2)
    assert len(events) == 1
    assert events.loc[0, "peak_cfs"] == 800.0
    assert events.loc[0, "duration_days"] == 5


def test_identify_events_splits_on_long_recession():
    dates = pd.date_range("2020-01-01", periods=14, freq="D")
    flow = np.array([10, 500, 10, 10, 10, 10, 10, 600, 10, 10, 10, 10, 10, 10], dtype=float)
    events = hmf.identify_events(dates, flow, threshold_cfs=100.0, merge_gap_days=2)
    assert len(events) == 2


def test_identify_events_returns_empty_frame_when_nothing_exceeds():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    events = hmf.identify_events(dates, np.full(5, 10.0), threshold_cfs=100.0)
    assert events.empty
    assert "excess_af" in events.columns


def test_identify_events_rejects_mismatched_lengths():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    with pytest.raises(ValueError):
        hmf.identify_events(dates, np.full(4, 10.0), threshold_cfs=1.0)


def test_event_summary_handles_empty_input():
    summary = hmf.event_summary(pd.DataFrame(), years=5.0)
    assert summary["n_events"] == 0
    assert summary["annual_excess_af"] == 0.0
