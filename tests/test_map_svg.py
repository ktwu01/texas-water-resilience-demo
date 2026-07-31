"""The no-WebGL fallback map and the size legend.

Two failure modes drive these tests. A fallback that is itself broken is worse
than none, because it only renders in the situation nobody tests. And a legend
whose labels fall outside the viewBox is silently clipped by the browser, which
is how the size scale shipped unlabelled the first time.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from twr.capture_index import FLAG_COLORS
from twr.geo import load_geography
from twr.map_svg import (
    LEGEND_STOPS,
    MAX_R,
    MIN_R,
    basin_map_svg,
    size_legend_svg,
)

SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def geography():
    return load_geography()


@pytest.fixture
def screening(geography):
    basin_ids = list(geography.basins)
    return pd.DataFrame(
        {
            "basin_id": basin_ids,
            "basin_name": [b.replace("_", " ").title() for b in basin_ids],
            "flag": ["NO_ACTION", "WATCH", "STANDBY", "CAPTURE", "BLOCKED", "WATCH", "NO_ACTION"][
                : len(basin_ids)
            ],
            "capture_index": np.linspace(0.0, 1.0, len(basin_ids)),
        }
    )


def _parse(svg: str) -> ET.Element:
    """Parse, which also asserts the SVG is well formed rather than nearly so."""
    return ET.fromstring(svg)


def _circles(root: ET.Element) -> list[ET.Element]:
    return root.findall(f".//{SVG_NS}circle")


def _basin_circles(root: ET.Element) -> list[ET.Element]:
    """Only the basin markers.

    Outlet dots and site pins are circles too, and the outlet dots are drawn
    first, so indexing into every circle on the page picks the wrong one. Basin
    markers are the ones grouped with a Capture Index title.
    """
    found = []
    for group in root.findall(f".//{SVG_NS}g"):
        title = group.find(f"{SVG_NS}title")
        if title is not None and "Capture Index" in (title.text or ""):
            found.extend(group.findall(f"{SVG_NS}circle"))
    return found


def test_fallback_is_well_formed_svg(screening, geography):
    root = _parse(basin_map_svg(screening, geography))
    assert root.tag == f"{SVG_NS}svg"
    assert root.get("viewBox")


def test_fallback_draws_the_state_outline(screening, geography):
    root = _parse(basin_map_svg(screening, geography))
    polygons = root.findall(f".//{SVG_NS}polygon")
    assert polygons, "no state outline: the fallback would be dots on a void"
    assert len(polygons[0].get("points", "").split()) > 50


def test_every_screened_basin_is_drawn(screening, geography):
    svg = basin_map_svg(screening, geography)
    for name in screening["basin_name"]:
        assert name in svg


def test_a_basin_without_geometry_is_dropped_not_misplaced(screening, geography):
    frame = pd.concat(
        [
            screening,
            pd.DataFrame(
                [{"basin_id": "not_a_basin", "basin_name": "Nowhere",
                  "flag": "WATCH", "capture_index": 0.5}]
            ),
        ],
        ignore_index=True,
    )
    assert "Nowhere" not in basin_map_svg(frame, geography)


def test_radius_is_monotone_in_capture_index_and_bounded(geography):
    basin_id = next(iter(geography.basins))
    radii = []
    for index in (0.0, 0.25, 0.5, 1.0):
        svg = basin_map_svg(
            pd.DataFrame([{"basin_id": basin_id, "basin_name": "X", "flag": "WATCH",
                           "capture_index": index}]),
            geography,
        )
        radii.append(float(_basin_circles(_parse(svg))[0].get("r")))
    assert radii == sorted(radii)
    assert radii[0] == pytest.approx(MIN_R)
    assert radii[-1] == pytest.approx(MAX_R)


def test_zero_capture_index_stays_visible(geography):
    """A marker that shrinks to nothing removes the basin from the map."""
    basin_id = next(iter(geography.basins))
    svg = basin_map_svg(
        pd.DataFrame([{"basin_id": basin_id, "basin_name": "X", "flag": "NO_ACTION",
                       "capture_index": 0.0}]),
        geography,
    )
    assert float(_basin_circles(_parse(svg))[0].get("r")) >= MIN_R


def test_non_finite_capture_index_does_not_produce_nan_geometry(geography):
    """NaN in an SVG coordinate makes the browser drop the shape, silently."""
    basin_id = next(iter(geography.basins))
    svg = basin_map_svg(
        pd.DataFrame([{"basin_id": basin_id, "basin_name": "X", "flag": "WATCH",
                       "capture_index": float("nan")}]),
        geography,
    )
    assert "nan" not in svg.lower()
    assert float(_basin_circles(_parse(svg))[0].get("r")) == pytest.approx(MIN_R)


def test_colour_tracks_the_flag_palette(geography):
    basin_id = next(iter(geography.basins))
    for flag, color in FLAG_COLORS.items():
        svg = basin_map_svg(
            pd.DataFrame([{"basin_id": basin_id, "basin_name": "X", "flag": flag,
                           "capture_index": 0.5}]),
            geography,
        )
        assert color in svg, flag


def test_markers_carry_a_hover_title(screening, geography):
    root = _parse(basin_map_svg(screening, geography))
    titles = [element.text or "" for element in root.findall(f".//{SVG_NS}title")]
    assert any("Capture Index" in title for title in titles)
    assert any("decision unit" in title for title in titles)


def test_basin_names_are_escaped(geography):
    """A name is data. Rendered unescaped it becomes markup."""
    basin_id = next(iter(geography.basins))
    svg = basin_map_svg(
        pd.DataFrame([{"basin_id": basin_id, "basin_name": "<script>x</script>",
                       "flag": "WATCH", "capture_index": 0.4}]),
        geography,
    )
    assert "<script>" not in svg
    _parse(svg)


def test_empty_screening_yields_no_svg(geography):
    assert basin_map_svg(pd.DataFrame(columns=["basin_id"]), geography) == ""


def test_every_drawn_point_is_inside_the_viewbox(screening, geography):
    root = _parse(basin_map_svg(screening, geography))
    _, _, width, height = (float(v) for v in root.get("viewBox").split())
    for circle in _circles(root):
        assert 0 <= float(circle.get("cx")) <= width
        assert 0 <= float(circle.get("cy")) <= height


def test_legend_labels_fit_inside_its_viewbox():
    """Clipped labels leave the size scale unexplained, which is the whole point."""
    root = _parse(size_legend_svg())
    _, _, width, height = (float(v) for v in root.get("viewBox").split())
    for circle in _circles(root):
        assert float(circle.get("cy")) + float(circle.get("r")) <= height
    for text in root.findall(f".//{SVG_NS}text"):
        assert float(text.get("y")) <= height, "label falls outside the viewBox"
        assert float(text.get("x")) <= width


def test_legend_says_what_size_means():
    svg = size_legend_svg()
    assert "Capture Index" in svg
    assert "NOT basin area" in svg


def test_legend_shows_a_reference_circle_per_stop():
    root = _parse(size_legend_svg())
    labels = {(element.text or "").strip() for element in root.findall(f".//{SVG_NS}text")}
    for stop in LEGEND_STOPS:
        assert f"{stop:.1f}" in labels


def test_legend_covers_every_flag_including_blocked():
    """BLOCKED is outside the ordered ladder and is the one worth recognising."""
    svg = size_legend_svg()
    for flag, color in FLAG_COLORS.items():
        assert flag in svg
        assert color in svg
