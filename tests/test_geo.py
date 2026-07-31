"""The map layer: boundary integrity, honest joins, perceptual marker scaling.

The map is presentation, not inference, so these tests are about not misleading
a viewer. The failure mode that matters is a basin drawn in the wrong place, or
silently dropped, or scaled so one flood swallows the state.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from twr.config import load_basins
from twr.geo import (
    BASIN_POINTS,
    SITE_POINTS,
    load_boundary,
    locate_basins,
    marker_sizes,
    render_texas_map,
    unlocated_basins,
)

# Generous box around Texas. A point outside this is a coordinate-order bug or a
# sign error, both of which put a marker in the ocean.
TEXAS_BOX = (-107.5, 25.0, -92.5, 37.5)


def _statewide(basin_ids, flags=None, volumes=None):
    n = len(basin_ids)
    if volumes is None:
        volumes = np.arange(1, n + 1) * 100.0
    return pd.DataFrame(
        {
            "basin_id": list(basin_ids),
            "basin_name": [b.title() for b in basin_ids],
            "flag": flags or ["WATCH"] * n,
            "capture_index": np.linspace(0.1, 0.9, n),
            "expected_capturable_af": volumes,
        }
    )


def test_boundary_loads_and_is_a_closed_ring_in_texas():
    boundary = load_boundary()
    assert boundary.rings
    for ring in boundary.rings:
        assert ring.shape[1] == 2
        assert np.allclose(ring[0], ring[-1]), "ring must close or fill() will draw a chord"
    min_lon, min_lat, max_lon, max_lat = boundary.bounds
    assert TEXAS_BOX[0] < min_lon < max_lon < TEXAS_BOX[2]
    assert TEXAS_BOX[1] < min_lat < max_lat < TEXAS_BOX[3]


def test_boundary_carries_its_provenance():
    """A vendored dataset with no source recorded is a citation waiting to be faked."""
    boundary = load_boundary()
    assert "Census" in boundary.source


def test_boundary_file_declares_coordinate_order_and_licence():
    from twr.geo import BOUNDARY_PATH

    doc = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    assert doc["coordinate_order"] == "[longitude, latitude]"
    assert doc["source_url"].startswith("https://")
    assert "public domain" in doc["license"]


def test_missing_boundary_file_raises_rather_than_drawing_nothing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_boundary(tmp_path / "absent.json")


def test_every_configured_basin_has_a_point():
    """A new basin in config/basins.yaml must not silently vanish from the map."""
    configured = {basin.id for basin in load_basins()}
    assert configured == set(BASIN_POINTS), (
        "config/basins.yaml and twr.geo.BASIN_POINTS disagree; "
        "add the missing coordinate or the basin will be dropped from the map"
    )


def test_all_points_are_inside_texas():
    for name, (lat, lon) in {**BASIN_POINTS, **SITE_POINTS}.items():
        assert TEXAS_BOX[1] < lat < TEXAS_BOX[3], name
        assert TEXAS_BOX[0] < lon < TEXAS_BOX[2], name


def test_statewide_site_is_deliberately_unplaced():
    """Screening every basin is not a location; pinning it to a centroid would lie."""
    assert "twdb_statewide" not in SITE_POINTS


def test_locate_basins_attaches_coordinates_and_provenance():
    located = locate_basins(_statewide(["brazos", "nueces"]))
    assert located.loc[0, "lat"] == pytest.approx(BASIN_POINTS["brazos"][0])
    assert located.loc[0, "lon"] == pytest.approx(BASIN_POINTS["brazos"][1])
    assert set(located["geo_provenance"]) == {"approximate"}


def test_unknown_basin_is_reported_not_placed_at_null_island():
    frame = _statewide(["brazos", "not_a_basin"])
    located = locate_basins(frame)
    assert np.isnan(located.loc[1, "lat"])
    assert unlocated_basins(frame) == ["not_a_basin"]


def test_locate_basins_survives_empty_input():
    located = locate_basins(pd.DataFrame(columns=["basin_id"]))
    assert located.empty
    assert unlocated_basins(pd.DataFrame()) == []


def test_marker_sizes_scale_by_area_not_radius():
    """Four times the volume should read as twice the width, not four times."""
    sizes = marker_sizes(pd.Series([25.0, 100.0]))
    assert np.sqrt(sizes[1]) > np.sqrt(sizes[0])
    # sqrt scaling of area means the larger value maps to the ceiling.
    assert sizes[1] > sizes[0]
    assert sizes.min() >= 40.0


def test_marker_sizes_handle_degenerate_and_dirty_input():
    assert np.all(marker_sizes(pd.Series([0.0, 0.0])) == 40.0)
    assert np.all(np.isfinite(marker_sizes(pd.Series([np.nan, -5.0, 10.0]))))
    assert marker_sizes(pd.Series([], dtype=float)).size == 0


def test_render_writes_a_png(tmp_path):
    out = render_texas_map(
        _statewide(["brazos", "guadalupe", "rio_grande"], flags=["CAPTURE", "WATCH", "BLOCKED"]),
        tmp_path / "figures" / "texas_map.png",
        site_flags=pd.DataFrame({"site_id": ["kerrville_asr"], "flag": ["CAPTURE"]}),
    )
    assert out is not None and out.exists()
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_render_returns_none_when_nothing_is_locatable(tmp_path):
    out = render_texas_map(_statewide(["not_a_basin"]), tmp_path / "map.png")
    assert out is None
    assert not (tmp_path / "map.png").exists()
