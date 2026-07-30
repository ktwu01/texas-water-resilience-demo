"""Data access layer.

The demo runs entirely on synthetic data. This module exists so that the switch
to real observations is a single, obvious change rather than a rewrite: every
loader returns the same daily long-format frame with the columns in
``synth.SENSOR_COLUMNS``, and each real connector is a stub that raises
``NotImplementedError`` with the product short name and DAAC it needs.

Nothing here fabricates an endpoint. Product short names and the hosting archive
are recorded because they are stable and checkable; exact URLs, collection
versions, and subsetting parameters must be confirmed against the archive's own
catalogue before use. See docs/DATA_SOURCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, Basin
from .synth import SENSOR_COLUMNS, simulate_all


@dataclass(frozen=True)
class ProductSpec:
    """What a real connector would have to go get."""

    variable: str
    mission: str
    product_short_name: str
    archive: str
    native_resolution: str
    latency: str
    notes: str = ""


# Registry of the observation streams the proposal fuses. Verify versions against
# the archive catalogue before wiring a connector to it.
PRODUCTS: dict[str, ProductSpec] = {
    "precip_mm": ProductSpec(
        variable="precipitation",
        mission="GPM",
        product_short_name="GPM_3IMERGHHL (Late Run, half-hourly) / GPM_3IMERGDL (daily)",
        archive="NASA GES DISC",
        native_resolution="0.1 deg, 30 min",
        latency="~14 h (Late Run)",
        notes="Late Run is the near-real-time option; Final Run is research grade.",
    ),
    "soil_moisture": ProductSpec(
        variable="surface soil moisture",
        mission="SMAP",
        product_short_name="SPL3SMP_E (L3 enhanced radiometer, 9 km)",
        archive="NASA NSIDC DAAC",
        native_resolution="9 km, 1-3 day revisit",
        latency="~24-50 h",
        notes="Drives the antecedent storage capacity term.",
    ),
    "water_extent_km2": ProductSpec(
        variable="surface water extent and elevation",
        mission="SWOT",
        product_short_name="SWOT L2 HR River Single Pass (RiverSP) and Water Mask Pixel Cloud",
        archive="NASA PO.DAAC",
        native_resolution="~100 m river nodes/reaches",
        latency="days; ~21-day orbit repeat",
        notes=(
            "Revisit is the binding limit for operational use; treat as event "
            "confirmation, not forecast input."
        ),
    ),
    "et_mm": ProductSpec(
        variable="evapotranspiration",
        mission="ECOSTRESS",
        product_short_name="ECO_L3T_JET (Collection 2 tiled ET)",
        archive="NASA LP DAAC",
        native_resolution="~70 m, irregular revisit",
        latency="days",
        notes="Cloud gaps are large; the demo forward-fills them causally.",
    ),
    "forcing": ProductSpec(
        variable="meteorological forcing",
        mission="NLDAS-3",
        product_short_name="NLDAS-3 (1 km CONUS forcing, NASA)",
        archive="NASA GES DISC",
        native_resolution="1 km, hourly",
        latency="near real time",
        notes="Confirm current collection status and version before relying on it.",
    ),
    "reanalysis": ProductSpec(
        variable="high-resolution reanalysis",
        mission="CONUS404",
        product_short_name="CONUS404 (4 km WRF hydroclimate reanalysis, USGS/NCAR)",
        archive="USGS HyTEST / cloud object store",
        native_resolution="4 km, hourly",
        latency="retrospective only",
        notes="Used as the high-resolution training target for downscaling, not as NRT input.",
    ),
    "flow_cfs": ProductSpec(
        variable="streamflow",
        mission="USGS streamgauge network",
        product_short_name="NWIS instantaneous values (parameter 00060, discharge)",
        archive="USGS Water Services (waterservices.usgs.gov)",
        native_resolution="point, 15 min",
        latency="~1 h",
        notes="Provisional data; subject to revision after rating-curve updates.",
    ),
}


def products_table() -> pd.DataFrame:
    rows = [{"key": key, **spec.__dict__} for key, spec in PRODUCTS.items()]
    return pd.DataFrame(rows)


def load_synthetic(
    basins: list[Basin],
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    seed: int = 0,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Generate (or reload) the synthetic record for the requested basins and span.

    The cache is only reused when it actually covers the request. An earlier
    version returned the whole cached frame regardless of ``start`` and ``end``,
    which silently ignored the caller's date range: ``run_pipeline.py --fast``
    asked for a shorter record and got the full eleven years, so it was not fast.
    """
    cache_path = cache_path or DATA_DIR / "processed" / "synthetic_daily.csv"
    want = {basin.id for basin in basins}
    requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)

    if cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["date"])
        covers_basins = want.issubset(set(frame["basin_id"].unique()))
        covers_span = (
            frame["date"].min() <= requested_start and frame["date"].max() >= requested_end
        )
        if covers_basins and covers_span:
            subset = frame[
                frame["basin_id"].isin(want)
                & frame["date"].between(requested_start, requested_end)
            ]
            return subset.sort_values(["basin_id", "date"]).reset_index(drop=True)

    frame = simulate_all(basins, start=start, end=end, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def load_observed(basins: list[Basin], start: str, end: str) -> pd.DataFrame:
    """Placeholder for the real multi-sensor fusion path.

    Implementing this means: authenticate to NASA Earthdata, subset each product
    in PRODUCTS to the basin polygons, aggregate to daily, join on date, and pull
    matching USGS discharge. The output must have exactly the columns in
    ``synth.SENSOR_COLUMNS`` plus ``date`` and ``basin_id`` so that the rest of
    the pipeline is unchanged.
    """
    raise NotImplementedError(
        "Real-data ingest is not implemented in this demo. "
        f"Required columns: {['date', 'basin_id', *SENSOR_COLUMNS]}. "
        "Products and archives are listed in twr.ingest.PRODUCTS and "
        "docs/DATA_SOURCES.md. NASA Earthdata Login credentials are required."
    )


def validate_record(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly on a record that the rest of the pipeline cannot use."""
    required = ["date", "basin_id", *SENSOR_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"record is missing columns: {missing}")
    if frame.empty:
        raise ValueError("record is empty")
    if frame["flow_cfs"].isna().all():
        raise ValueError("no streamflow values present")
    if (frame["flow_cfs"].dropna() < 0).any():
        raise ValueError("negative discharge values present")
    duplicated = frame.duplicated(subset=["basin_id", "date"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate basin/date rows")
    return frame
