"""Map geography for the dashboard, loaded from config/geography.yaml.

This module is deliberately thin and has no dependency on the modelling code.
It answers one question: where on a basemap should a basin or a site be drawn?

The coordinates it serves are approximate real-world anchors, not watershed
polygons, and not synthetic. See the header of config/geography.yaml for the
provenance rules and for what a real deployment must replace them with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR, DATA_DIR, read_yaml

# Vendored Texas state outline. Real geometry, generalised for display, with its
# provenance recorded inside the file. See load_texas_outline.
TEXAS_OUTLINE_PATH = DATA_DIR / "geo" / "texas_state.geojson"


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"latitude out of range: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"longitude out of range: {self.lon}")


@dataclass(frozen=True)
class BasinGeometry:
    """Visual anchors for one basin.

    ``centroid`` is a hand-placed midpoint, not a computed area centroid.
    ``outlet`` is the approximate downstream terminus. No boundary polygon is
    carried, because this repository has not delineated one.
    """

    id: str
    centroid: Point
    outlet: Point
    reaches_coast: bool


@dataclass(frozen=True)
class SiteGeometry:
    id: str
    location: Point
    label: str


@dataclass(frozen=True)
class MapView:
    """Initial camera for the basemap."""

    latitude: float
    longitude: float
    zoom: float


@dataclass(frozen=True)
class Geography:
    provenance: str
    view: MapView
    basins: dict[str, BasinGeometry]
    sites: dict[str, SiteGeometry]

    @property
    def coastal_basin_ids(self) -> tuple[str, ...]:
        """Basins whose outlet reaches the Gulf.

        Note there is no separate "Texas Coast" decision unit in this repo. The
        coast is a property of where these basins discharge, which is why this
        is a filter over basins rather than a region of its own.
        """
        return tuple(
            basin_id for basin_id, geom in self.basins.items() if geom.reaches_coast
        )


def load_texas_outline(path: Path | None = None) -> list[list[list[float]]]:
    """Texas state boundary as a list of [lon, lat] rings.

    This is real geometry (US Census cartographic boundary, public domain), not a
    synthetic shape, and it is generalised for display: roughly 150 vertices for
    the whole state, with a simplified Gulf coastline. It exists so the map shows
    recognisable Texas even where the basemap tiles are unavailable, which is the
    common case behind a corporate proxy or on a machine with no WebGL-accelerated
    tile rendering.

    Do not measure anything from it, and do not mistake it for a watershed
    delineation. Returns an empty list if the file is absent, so the map degrades
    to markers rather than failing.
    """
    target = path or TEXAS_OUTLINE_PATH
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    rings: list[list[list[float]]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry", {})
        kind = geometry.get("type")
        if kind == "Polygon":
            rings.extend(geometry["coordinates"])
        elif kind == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                rings.extend(polygon)
    return rings


def _point(raw: dict[str, float]) -> Point:
    return Point(lat=float(raw["lat"]), lon=float(raw["lon"]))


def load_geography(path: Path | None = None) -> Geography:
    payload = read_yaml(path or CONFIG_DIR / "geography.yaml")

    basins: dict[str, BasinGeometry] = {}
    for raw in payload.get("basins", []):
        basin_id = raw["id"]
        if basin_id in basins:
            raise ValueError(f"duplicate basin geometry: {basin_id}")
        basins[basin_id] = BasinGeometry(
            id=basin_id,
            centroid=_point(raw["centroid"]),
            outlet=_point(raw["outlet"]),
            reaches_coast=bool(raw.get("reaches_coast", False)),
        )

    sites: dict[str, SiteGeometry] = {}
    for raw in payload.get("sites", []):
        site_id = raw["id"]
        if site_id in sites:
            raise ValueError(f"duplicate site geometry: {site_id}")
        sites[site_id] = SiteGeometry(
            id=site_id,
            location=_point(raw["location"]),
            label=raw["label"],
        )

    if not basins:
        raise ValueError("no basin geometry defined")

    raw_view = payload.get("view", {})
    view = MapView(
        latitude=float(raw_view.get("latitude", 31.2)),
        longitude=float(raw_view.get("longitude", -99.3)),
        zoom=float(raw_view.get("zoom", 4.7)),
    )

    return Geography(
        provenance=payload.get("provenance", "unknown"),
        view=view,
        basins=basins,
        sites=sites,
    )
