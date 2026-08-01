"""Fetching the published outputs instead of recomputing them.

This path exists because recomputing got the hosted app CPU throttled. It runs
on a fresh server, in front of a first-time visitor, over a network that may be
slow or blocked, so its failure modes matter more than its happy path. Every test
here uses a fake opener: a test suite that reaches the real internet fails for
reasons that have nothing to do with the code.
"""

from __future__ import annotations

import gzip
import io
import urllib.error

import pytest

from twr.published import (
    OPTIONAL_FILES,
    PUBLISHED_BASES,
    REQUIRED_FILES,
    clear,
    fetch_published,
)

ALL_FILES = (*REQUIRED_FILES, *OPTIONAL_FILES)


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, encoding: str | None = None) -> None:
        super().__init__(payload)
        self.headers = {"Content-Encoding": encoding} if encoding else {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


def _opener(*, fail=(), encoding=None, body=b"col\n1\n", record=None):
    """A urlopen stand-in. `fail` names files that should raise."""

    def open_url(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if record is not None:
            record.append((url, dict(getattr(request, "headers", {}))))
        if any(url.endswith(name) for name in fail):
            raise urllib.error.URLError("nope")
        payload = gzip.compress(body) if encoding == "gzip" else body
        return _Response(payload, encoding)

    return open_url


def test_downloads_every_file_and_reports_them(tmp_path):
    result = fetch_published(tmp_path, opener=_opener())
    assert result.usable
    assert sorted(result.fetched) == sorted(ALL_FILES)
    assert result.missing == []
    for name in ALL_FILES:
        assert (tmp_path / name).read_bytes() == b"col\n1\n"


def test_decompresses_a_gzipped_response(tmp_path):
    """Pages compresses CSV when asked, and the bodies are megabytes."""
    result = fetch_published(tmp_path, opener=_opener(encoding="gzip", body=b"a,b\n1,2\n"))
    assert result.usable
    assert (tmp_path / "site_flags.csv").read_text() == "a,b\n1,2\n"


def test_asks_for_compression(tmp_path):
    calls: list = []
    fetch_published(tmp_path, opener=_opener(record=calls))
    assert calls
    for _, headers in calls:
        assert headers.get("Accept-encoding") == "gzip" or headers.get("Accept-Encoding") == "gzip"


def test_a_missing_optional_file_still_leaves_the_bundle_usable(tmp_path):
    result = fetch_published(tmp_path, opener=_opener(fail=("asr_scenario_sweep.csv",)))
    assert result.usable, "one absent scenario sweep should not fail the whole page"
    assert result.missing == ["asr_scenario_sweep.csv"]
    assert not (tmp_path / "asr_scenario_sweep.csv").exists()


def test_a_missing_required_file_makes_the_bundle_unusable(tmp_path):
    result = fetch_published(tmp_path, opener=_opener(fail=("site_flags.csv",)))
    assert not result.usable
    assert "site_flags.csv" in result.missing


def test_falls_through_to_the_next_base(tmp_path):
    """The canonical URL is a custom domain. It should not be a single point of failure."""
    first, second = PUBLISHED_BASES[0], PUBLISHED_BASES[1]

    def open_url(request, timeout=None):
        if request.full_url.startswith(first):
            raise urllib.error.URLError("custom domain down")
        return _Response(b"col\n1\n")

    result = fetch_published(tmp_path, opener=open_url)
    assert result.usable
    assert result.base == second


def test_reports_unusable_when_every_base_fails(tmp_path):
    result = fetch_published(tmp_path, opener=_opener(fail=ALL_FILES))
    assert not result.usable
    assert sorted(result.missing) == sorted(ALL_FILES)
    assert list(tmp_path.iterdir()) == []


def test_existing_files_are_not_refetched(tmp_path):
    for name in ALL_FILES:
        (tmp_path / name).write_text("local")
    calls: list = []
    result = fetch_published(tmp_path, opener=_opener(record=calls))
    assert calls == [], "a populated outputs/ must not be overwritten from the network"
    assert result.base == "local"
    assert result.usable
    assert (tmp_path / "site_flags.csv").read_text() == "local"


def test_a_partial_download_leaves_no_truncated_file(tmp_path):
    """A half-written CSV parses fine and is silently wrong, which is the danger."""

    def open_url(request, timeout=None):
        class Exploding(_Response):
            def read(self, *_):
                raise OSError("connection reset")

        return Exploding(b"")

    result = fetch_published(tmp_path, opener=open_url)
    assert not result.usable
    assert list(tmp_path.glob("*.csv")) == []
    assert list(tmp_path.glob("*.part")) == []


def test_corrupt_gzip_is_treated_as_a_failure_not_as_data(tmp_path):
    def open_url(request, timeout=None):
        return _Response(b"not actually gzip", "gzip")

    result = fetch_published(tmp_path, opener=open_url)
    assert not result.usable
    assert list(tmp_path.glob("*.csv")) == []


def test_clear_removes_the_downloaded_bundle(tmp_path):
    fetch_published(tmp_path, opener=_opener())
    clear(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_published_bases_are_https():
    """Plain http would let a proxy rewrite the numbers on the page."""
    for base in PUBLISHED_BASES:
        assert base.startswith("https://"), base


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_required_files_are_the_ones_the_dashboard_cannot_start_without(name):
    assert name in ("site_flags.csv", "statewide_screening.csv", "run_summary.json")
