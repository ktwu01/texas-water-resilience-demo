"""Shared test fixtures.

The important one is cache isolation. ``ingest.load_synthetic`` memoises the
synthetic record to ``data/processed/synthetic_daily.csv`` inside the repository,
which is right for a demo and wrong for a test suite: results then depend on
whatever a previous pipeline run happened to leave on disk. That produced a real
flake, where ``test_scenario_sweep_uses_an_actionable_day`` failed in the full
suite and passed in isolation, because a stale cache from an earlier run supplied
different weather than the fixture expected.

Redirecting the cache to a per-session temp directory makes every run start from
the same state and stops the suite writing into the user's data directory.
"""

from __future__ import annotations

import pytest

from twr import ingest


@pytest.fixture(scope="session", autouse=True)
def isolated_data_cache(tmp_path_factory):
    """Point the synthetic-record cache at a temp directory for the whole session."""
    root = tmp_path_factory.mktemp("twr_data")
    (root / "processed").mkdir(parents=True, exist_ok=True)
    original = ingest.DATA_DIR
    ingest.DATA_DIR = root
    yield root
    ingest.DATA_DIR = original
