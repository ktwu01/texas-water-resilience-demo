"""Fetch the published outputs instead of recomputing them.

A hosted dashboard starts with an empty ``outputs/`` directory, because those
files are gitignored on purpose. The first version ran the full pipeline on first
request: leave-one-basin-out cross-validation plus a scenario sweep, about two
and a half minutes of scikit-learn. On a free-tier container that is enough CPU
to get the app throttled, which is what happened.

The work is already done elsewhere. CI runs the pipeline on every push to main
and publishes the CSVs next to the static briefing, so the hosted app can
download the same artefacts in a couple of seconds and spend no CPU at all. That
also means the dashboard and the published page are showing byte-identical
numbers rather than two runs that happen to share a seed.

Falls back to a local run when the download fails, so a fresh clone with no
network still works. Nothing here is required for local use: if ``outputs/`` is
already populated, this module never runs.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

#: Where CI publishes. Tried in order, because the canonical Pages URL is a
#: custom domain and the default one is the fallback if that ever lapses.
PUBLISHED_BASES = (
    "https://koutian.is-a.dev/texas-water-resilience-demo/data",
    "https://ktwu01.github.io/texas-water-resilience-demo/data",
)

#: Without these the dashboard has nothing to show and must fall back.
REQUIRED_FILES = (
    "site_flags.csv",
    "statewide_screening.csv",
    "run_summary.json",
)

#: Individual tabs degrade to their own empty states without these, which is a
#: better outcome than failing the whole page over a missing scenario sweep.
OPTIONAL_FILES = (
    "flag_history.csv",
    "daily_timeseries.csv",
    "storage_balance.csv",
    "hmf_events.csv",
    "spatial_cv_folds.csv",
    "feature_importance.csv",
    "asr_scenario_sweep.csv",
)

DEFAULT_TIMEOUT = 25.0


@dataclass
class FetchResult:
    base: str | None = None
    fetched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when every file the dashboard cannot start without arrived."""
        return not set(REQUIRED_FILES) & set(self.missing)


def _download(url: str, destination: Path, timeout: float, opener) -> None:
    """Fetch one file, decompressing if the server compressed it.

    Written to a temporary name and moved into place, so an interrupted download
    cannot leave a truncated CSV that reads as valid but short.
    """
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with opener(request, timeout=timeout) as response:
        payload = response.read()
        encoding = (response.headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        payload = gzip.decompress(payload)

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def fetch_published(
    destination: Path,
    *,
    bases: tuple[str, ...] = PUBLISHED_BASES,
    timeout: float = DEFAULT_TIMEOUT,
    opener=urllib.request.urlopen,
) -> FetchResult:
    """Populate ``destination`` from the published site.

    Files already present are left alone, so this is cheap to call on every
    request and safe to call over a populated local ``outputs/``.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = [
        name for name in (*REQUIRED_FILES, *OPTIONAL_FILES)
        if not (destination / name).exists()
    ]
    if not wanted:
        return FetchResult(base="local", fetched=[], missing=[])

    last = FetchResult(missing=list(wanted))
    for base in bases:
        result = FetchResult(base=base)
        for name in wanted:
            try:
                _download(f"{base}/{name}", destination / name, timeout, opener)
            except (urllib.error.URLError, OSError, ValueError, gzip.BadGzipFile):
                result.missing.append(name)
            else:
                result.fetched.append(name)
        if result.usable:
            return result
        last = result
    return last


def clear(destination: Path, names: tuple[str, ...] | None = None) -> None:
    """Remove downloaded artefacts. Used by tests and by a forced refresh."""
    destination = Path(destination)
    for name in names or (*REQUIRED_FILES, *OPTIONAL_FILES):
        target = destination / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
