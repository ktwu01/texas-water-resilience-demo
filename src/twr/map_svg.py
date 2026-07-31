"""Fallback map and size legend, as inline SVG.

Two problems this solves, both of them ways the deck.gl map misleads or vanishes.

**No WebGL, no map.** ``dashboard/basin_map.py`` renders through deck.gl, which
needs WebGL. Where it is unavailable (a locked-down browser, a VM with no GPU
passthrough, some remote desktop setups) the deck paints nothing at all: no
error, no warning, just an empty rectangle where the answer to "where in Texas"
should be. This module draws the same thing in SVG, which needs nothing but a
browser, so the fallback is a real map rather than an apology.

**Marker size reads as basin size.** Circle area encodes the Capture Index, not
drainage area, so a big circle can mean a high index in a small basin. A caption
saying so is easy to skip. ``size_legend_svg`` draws the scale itself, with
labelled reference circles, next to the flag colours.

No matplotlib and no pydeck here on purpose: the fallback must not depend on the
thing it is a fallback for, and a page that fails to render the primary map
should not need a plotting stack to render the substitute.
"""

from __future__ import annotations

import math
from html import escape

import pandas as pd

from .capture_index import FLAG_COLORS
from .geo import Geography, load_texas_outline

# Viewport in SVG user units. The aspect ratio is corrected for latitude below,
# so these only set the coordinate space, not the shape of the state.
WIDTH = 720.0
HEIGHT = 680.0
PAD = 24.0

# Circle radii in user units, matching the deck's floor-and-ceiling approach so
# a Capture Index of zero stays visible rather than collapsing to a dot.
MIN_R = 5.0
MAX_R = 26.0

INK = "#c9d1d9"
DIM = "#8b949e"
LAND = "#1b2430"
COAST = "#3d4a5a"
SITE_FILL = "#f0f6fc"
SITE_EDGE = "#0d1117"

# Reference values for the size legend. Chosen to span the flag ladder: below
# WATCH, around STANDBY, and at the top of the scale.
LEGEND_STOPS = (0.2, 0.5, 1.0)


class _Projection:
    """Equirectangular, with longitude compressed by cos(mean latitude).

    Good enough for a state-sized fallback map and it keeps Texas the shape
    people recognise. Anything more would be pretending to be a GIS.
    """

    def __init__(self, lons: list[float], lats: list[float]) -> None:
        self.min_lon, self.max_lon = min(lons), max(lons)
        self.min_lat, self.max_lat = min(lats), max(lats)
        mean_lat = math.radians((self.min_lat + self.max_lat) / 2.0)
        span_x = max((self.max_lon - self.min_lon) * math.cos(mean_lat), 1e-9)
        span_y = max(self.max_lat - self.min_lat, 1e-9)
        self.scale = min((WIDTH - 2 * PAD) / span_x, (HEIGHT - 2 * PAD) / span_y)
        self.cos_lat = math.cos(mean_lat)
        used_x = span_x * self.scale
        used_y = span_y * self.scale
        self.offset_x = (WIDTH - used_x) / 2.0
        self.offset_y = (HEIGHT - used_y) / 2.0

    def __call__(self, lon: float, lat: float) -> tuple[float, float]:
        x = self.offset_x + (lon - self.min_lon) * self.cos_lat * self.scale
        # SVG y grows downward, so latitude is flipped.
        y = self.offset_y + (self.max_lat - lat) * self.scale
        return round(x, 2), round(y, 2)


def _radius(capture_index: float) -> float:
    index = 0.0 if not math.isfinite(capture_index) else min(max(capture_index, 0.0), 1.0)
    # Area, not radius, carries the perception of magnitude.
    return round(MIN_R + (MAX_R - MIN_R) * math.sqrt(index), 2)


def basin_map_svg(screening: pd.DataFrame, geography: Geography) -> str:
    """Draw the screened basins over the Texas outline. Returns inline SVG.

    Returns an empty string when nothing is placeable, so a caller can fall
    through to its own empty state rather than embedding a blank frame.
    """
    if screening.empty:
        return ""

    placed = [
        (record, geography.basins[record["basin_id"]])
        for record in screening.to_dict("records")
        if record["basin_id"] in geography.basins
    ]
    if not placed:
        return ""

    rings = load_texas_outline()
    lons = [point[0] for ring in rings for point in ring]
    lats = [point[1] for ring in rings for point in ring]
    for _, geom in placed:
        lons.extend([geom.centroid.lon, geom.outlet.lon])
        lats.extend([geom.centroid.lat, geom.outlet.lat])
    for site in geography.sites.values():
        lons.append(site.location.lon)
        lats.append(site.location.lat)
    project = _Projection(lons, lats)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0f} {HEIGHT:.0f}" '
        f'role="img" aria-label="Capture Index by basin, Texas" '
        f'style="width:100%;height:auto;display:block">'
    ]

    for ring in rings:
        points = " ".join(f"{x},{y}" for x, y in (project(lon, lat) for lon, lat in ring))
        parts.append(
            f'<polygon points="{points}" fill="{LAND}" stroke="{COAST}" stroke-width="1.2"/>'
        )

    # Anchor to outlet, the same reach the deck draws when the coastal layer is on.
    for _record, geom in placed:
        if not geom.reaches_coast:
            continue
        x1, y1 = project(geom.centroid.lon, geom.centroid.lat)
        x2, y2 = project(geom.outlet.lon, geom.outlet.lat)
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#60a5fa" '
            f'stroke-width="1" stroke-opacity="0.55"/>'
            f'<circle cx="{x2}" cy="{y2}" r="3" fill="#60a5fa" fill-opacity="0.8"/>'
        )

    # Largest first, so a big NO_ACTION circle cannot hide a small CAPTURE one.
    for record, geom in sorted(placed, key=lambda item: -_radius(float(item[0]["capture_index"]))):
        index = float(record["capture_index"])
        x, y = project(geom.centroid.lon, geom.centroid.lat)
        color = FLAG_COLORS.get(record["flag"], "#888888")
        name = escape(str(record.get("basin_name", record["basin_id"])))
        shown = 0.0 if not math.isfinite(index) else index
        parts.append(
            f'<g><title>{name}: {record["flag"]}, Capture Index {shown:.2f}</title>'
            f'<circle cx="{x}" cy="{y}" r="{_radius(index)}" fill="{color}" '
            f'fill-opacity="0.82" stroke="#ffffff" stroke-opacity="0.75" stroke-width="1.2"/>'
            f"</g>"
        )
        parts.append(
            f'<text x="{x}" y="{y + _radius(index) + 12}" text-anchor="middle" '
            f'font-size="11" fill="{INK}" '
            f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">'
            f"{name} {shown:.2f}</text>"
        )

    for site in geography.sites.values():
        x, y = project(site.location.lon, site.location.lat)
        label = escape(site.label)
        parts.append(
            f'<g><title>{label}: decision unit</title>'
            f'<circle cx="{x}" cy="{y}" r="4.5" fill="{SITE_FILL}" stroke="{SITE_EDGE}" '
            f'stroke-width="1.5"/></g>'
            f'<text x="{x + 8}" y="{y + 4}" font-size="10.5" fill="{DIM}" font-style="italic" '
            f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">'
            f"{label}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


def size_legend_svg() -> str:
    """The size scale, drawn rather than described.

    Marker area encodes the Capture Index. Saying that in a caption leaves the
    reader free to assume the obvious wrong thing, which is that a bigger circle
    is a bigger basin.
    """
    # Tall enough for the largest reference circle plus its label underneath. At
    # 74 the labels fell outside the viewBox and were clipped away, which left the
    # scale unlabelled and so useless.
    row_height = 100.0
    width = 620.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {row_height:.0f}" '
        f'role="img" aria-label="Legend: circle area is the Capture Index" '
        f'style="width:100%;max-width:{width:.0f}px;height:auto;display:block">'
    ]
    font = "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

    parts.append(
        f'<text x="0" y="12" font-size="11" fill="{DIM}" font-family="{font}">'
        f"circle area = Capture Index, NOT basin area</text>"
    )
    x = MAX_R
    for stop in LEGEND_STOPS:
        radius = _radius(stop)
        parts.append(
            f'<circle cx="{x:.1f}" cy="54" r="{radius}" fill="none" stroke="{DIM}" '
            f'stroke-width="1.2"/>'
            f'<text x="{x:.1f}" y="{54 + MAX_R + 15:.0f}" text-anchor="middle" font-size="10" '
            f'fill="{DIM}" font-family="{font}">{stop:.1f}</text>'
        )
        x += 2 * MAX_R + 14

    x += 10
    # Every flag in the palette, BLOCKED included: it is the one a reader most
    # needs to recognise and it is not part of the ordered ladder.
    for flag, color in FLAG_COLORS.items():
        parts.append(
            f'<circle cx="{x:.1f}" cy="54" r="5" fill="{color}"/>'
            f'<text x="{x + 10:.1f}" y="58" font-size="10" fill="{DIM}" '
            f'font-family="{font}">{flag}</text>'
        )
        x += 22 + 6.1 * len(flag)

    parts.append("</svg>")
    return "".join(parts)
