"""Map geography and the dashboard's basin map layer.

Two classes of thing are guarded here.

*Config drift.* The geography file is a second, parallel list of basin and site
ids. If someone adds a basin to ``basins.yaml`` and forgets ``geography.yaml``,
the map silently drops it. The coverage tests below turn that into a failure.

*Tooltip resolution.* pydeck has one tooltip template per deck, shared by every
pickable layer, and it interpolates ``{field}`` against the hovered row. A field
missing from a layer's rows renders as a literal, unsubstituted ``{field}`` in the
hover card. That is why both frames must carry ``tooltip_title`` and
``tooltip_body``, and why it is worth asserting rather than eyeballing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dashboard"))

from basin_map import (  # noqa: E402
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    TOOLTIP_HTML,
    basin_map,
    build_basin_frame,
    build_site_frame,
)

from twr.capture_index import FLAG_COLORS  # noqa: E402
from twr.config import load_basins, load_sites  # noqa: E402
from twr.geo import Point, load_geography  # noqa: E402


@pytest.fixture(scope="module")
def geography():
    return load_geography()


def _screening_row(basin_id: str, name: str, capture_index: float, flag: str) -> dict:
    return {
        "basin_id": basin_id,
        "basin_name": name,
        "capture_index": capture_index,
        "flag": flag,
        "binding_constraint": "hydrologic_availability",
        "binding_if_captured": "water_rights",
        "expected_capturable_af": 1234.0,
        "median_excess_af": 42.0,
        "q90_excess_af": 999.0,
        "event_probability": 0.31,
    }


@pytest.fixture
def screening() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _screening_row("brazos", "Brazos", 0.0, "NO_ACTION"),
            _screening_row("guadalupe", "Guadalupe", 1.0, "CAPTURE"),
            _screening_row("nueces", "Nueces", 0.5, "WATCH"),
        ]
    )


# --- config coverage ------------------------------------------------------


def test_every_basin_has_map_geometry(geography):
    configured = {basin.id for basin in load_basins()}
    assert configured == set(geography.basins)


def test_every_site_has_map_geometry(geography):
    configured = {site.id for site in load_sites()}
    assert configured == set(geography.sites)


def test_anchors_are_inside_a_texas_bounding_box(geography):
    """Catches a transposed lat/lon pair, the classic geospatial typo.

    A swapped pair puts the marker in the Indian Ocean, which is obvious on a
    map and invisible in a diff.
    """
    for geom in geography.basins.values():
        for point in (geom.centroid, geom.outlet):
            assert 25.0 <= point.lat <= 37.0, geom.id
            assert -107.0 <= point.lon <= -93.0, geom.id
    for site in geography.sites.values():
        assert 25.0 <= site.location.lat <= 37.0, site.id
        assert -107.0 <= site.location.lon <= -93.0, site.id


def test_outlets_are_downstream_of_centroids(geography):
    """Every Texas basin here drains south or east to the Gulf."""
    for geom in geography.basins.values():
        assert geom.outlet.lat <= geom.centroid.lat, geom.id


def test_point_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        Point(lat=95.0, lon=-99.0)
    with pytest.raises(ValueError):
        Point(lat=31.0, lon=-200.0)


def test_coastal_basins_are_flagged(geography):
    """All seven basins reach the Gulf, so the filter should return all of them.

    This is the closest thing the repo has to a "Texas Coast region": there is no
    such decision unit, only basins that discharge to the coast.
    """
    assert set(geography.coastal_basin_ids) == set(geography.basins)


def test_duplicate_basin_geometry_is_rejected(tmp_path):
    path = tmp_path / "geography.yaml"
    path.write_text(
        "basins:\n"
        "  - id: brazos\n"
        "    centroid: {lat: 31.4, lon: -97.4}\n"
        "    outlet: {lat: 28.9, lon: -95.4}\n"
        "  - id: brazos\n"
        "    centroid: {lat: 31.4, lon: -97.4}\n"
        "    outlet: {lat: 28.9, lon: -95.4}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate basin geometry"):
        load_geography(path)


# --- frame construction ---------------------------------------------------


def test_basin_frame_places_every_screened_basin(screening, geography):
    frame = build_basin_frame(screening, geography)
    assert len(frame) == len(screening)
    assert set(frame["basin_id"]) == set(screening["basin_id"])


def test_basin_frame_drops_basins_without_geometry(screening, geography):
    """An unknown basin is dropped, not plotted at a guessed location."""
    unknown = pd.concat(
        [screening, pd.DataFrame([_screening_row("atchafalaya", "Nope", 0.9, "WATCH")])],
        ignore_index=True,
    )
    frame = build_basin_frame(unknown, geography)
    assert "atchafalaya" not in set(frame["basin_id"])
    assert len(frame) == len(screening)


def test_radius_is_monotone_in_capture_index_and_bounded(screening, geography):
    frame = build_basin_frame(screening, geography).sort_values("capture_index")
    radii = frame["radius"].tolist()
    assert radii == sorted(radii)
    assert radii[0] == pytest.approx(MIN_RADIUS_M)
    assert radii[-1] == pytest.approx(MAX_RADIUS_M)


def test_zero_capture_index_is_still_visible(screening, geography):
    """A Capture Index of 0 must stay hoverable, or NO_ACTION basins vanish."""
    frame = build_basin_frame(screening, geography)
    zero = frame[frame["capture_index"] == 0.0].iloc[0]
    assert zero["radius"] >= MIN_RADIUS_M > 0


def test_colors_track_the_flag_palette(screening, geography):
    frame = build_basin_frame(screening, geography)
    for row in frame.to_dict("records"):
        expected = FLAG_COLORS[row["flag"]].lstrip("#")
        assert row["color"] == [int(expected[i : i + 2], 16) for i in (0, 2, 4)]


def test_unknown_flag_falls_back_to_grey(geography):
    frame = build_basin_frame(
        pd.DataFrame([_screening_row("brazos", "Brazos", 0.4, "SOMETHING_NEW")]),
        geography,
    )
    assert frame.iloc[0]["color"] == [0x88, 0x88, 0x88]


def test_empty_screening_yields_no_frame_and_no_deck(geography):
    assert build_basin_frame(pd.DataFrame(), geography).empty
    deck, placed = basin_map(pd.DataFrame(), geography)
    assert deck is None
    assert placed == 0


# --- tooltip resolution ---------------------------------------------------


def test_tooltip_template_fields_exist_on_both_pickable_frames(screening, geography):
    """Every {field} in the template must be a column on every hoverable layer."""
    fields = {"tooltip_title", "tooltip_body"}
    assert all("{" + name + "}" in TOOLTIP_HTML for name in fields)
    basins = build_basin_frame(screening, geography)
    sites = build_site_frame(geography)
    assert fields <= set(basins.columns)
    assert fields <= set(sites.columns)
    for frame in (basins, sites):
        assert not frame["tooltip_title"].isna().any()
        assert not frame["tooltip_body"].isna().any()


def test_basin_tooltip_reports_the_decision_numbers(screening, geography):
    frame = build_basin_frame(screening, geography)
    row = frame[frame["basin_id"] == "guadalupe"].iloc[0]
    assert row["tooltip_title"] == "Guadalupe"
    body = row["tooltip_body"]
    assert "CAPTURE" in body
    assert "1,234 AF" in body            # expected capturable, thousands-separated
    assert "hydrologic_availability" in body
    assert "water_rights" in body
    assert "0.31" in body                # event probability
    assert "{" not in body               # nothing left unsubstituted


def test_site_tooltip_keeps_the_illustrative_caveat(geography):
    sites = build_site_frame(geography)
    assert len(sites) == len(geography.sites)
    assert all("illustrative" in body for body in sites["tooltip_body"])


# --- deck assembly --------------------------------------------------------


def test_deck_marks_only_the_point_layers_pickable(screening, geography):
    """Hover must hit basins and sites, never the basemap or the outlet lines."""
    deck, placed = basin_map(screening, geography, highlight_coastal=True)
    assert placed == len(screening)
    spec = __import__("json").loads(deck.to_json())
    pickable = [
        layer["@@type"] for layer in spec["layers"] if layer.get("pickable") is True
    ]
    assert pickable == ["ScatterplotLayer", "ScatterplotLayer"]


def test_coastal_toggle_adds_outlet_geometry(screening, geography):
    with_coast, _ = basin_map(screening, geography, highlight_coastal=True)
    without, _ = basin_map(screening, geography, highlight_coastal=False)
    assert len(with_coast.layers) == len(without.layers) + 2
    types = [layer.type for layer in with_coast.layers]
    assert "LineLayer" in types
    assert "LineLayer" not in [layer.type for layer in without.layers]


def test_deck_renders_to_html_with_a_keyless_basemap(screening, geography):
    """The map must work for a fresh clone: no API key, no Mapbox token."""
    deck, _ = basin_map(screening, geography, highlight_coastal=True)
    html = deck.to_html(as_string=True)
    assert "cartocdn" in html
    assert "tooltip_title" in html
    assert "mapbox_key" not in html.lower()
