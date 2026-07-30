"""Interactive basin map with hover tooltips, for the statewide screening tab.

Rendered with pydeck, which ships with Streamlit, so this adds no dependency and
needs no API key. Two independent sources of geography, deliberately:

1. The CARTO vector basemap, via pydeck's keyless `carto` provider, which gives
   city, road, river, and coastline labels.
2. A vendored Texas state outline drawn as a deck layer (data/geo/). The basemap
   needs network access and a working WebGL tile renderer; this polygon is part of
   the deck itself, so the map still reads as Texas when tiles do not arrive.

WHY NOT GOOGLE MAPS: the Google Maps JavaScript API requires a billed API key,
which cannot be committed to a public repository, and it would make the dashboard
fail closed for anyone cloning the repo. Nothing here needs Google's routing,
places, or Street View data.

WHAT THE MARKERS MEAN: each basin is a point at a hand-placed visual anchor, not
a filled watershed polygon, because this repository has not delineated basin
boundaries. Radius encodes the Capture Index, NOT basin size. See
config/geography.yaml.
"""

from __future__ import annotations

import math

import pandas as pd
import pydeck as pdk

from twr.capture_index import FLAG_COLORS
from twr.geo import Geography, load_texas_outline

# CARTO's vector basemap, served through pydeck's `carto` provider, which needs no
# access token. Voyager labels cities, roads, and rivers, which is what makes the
# map orienting rather than decorative.
#
# Do NOT try to serve the basemap as a hand-rolled TileLayer with
# `map_provider=None`. That path leaves pydeck's mapStyle as the literal
# placeholder "__MAP_STYLE__", the base render surface never initialises, and the
# tiles silently never paint: you get scatterplot dots floating on the page
# background with no error anywhere. The deck still renders, the HTML still
# contains the tile URL, and the log stays clean, so it passes every check short
# of actually looking at the map.
BASEMAP_PROVIDER = "carto"
BASEMAP_ATTRIBUTION = "Basemap (c) OpenStreetMap contributors, (c) CARTO"

# The dashboard runs on Streamlit's dark theme, so the basemap is the dark CARTO
# style. Voyager (CARTO_ROAD) is light and reads as a bright rectangle pasted into
# a dark page. Dark Matter keeps city, road, and coastline labels while letting the
# flag colours stay the brightest thing on the map.
BASEMAP_STYLE = pdk.map_styles.CARTO_DARK

# Point radius in metres. A Capture Index of 0 still has to be visible and
# hoverable, hence the floor.
MIN_RADIUS_M = 16_000
MAX_RADIUS_M = 62_000

# Decision units, drawn on a dark basemap: a near-white pin with a dark ring, so
# it stays legible against both land and water without competing with the flags.
SITE_COLOR = [245, 245, 245]
SITE_LINE_COLOR = [17, 24, 39, 220]
SITE_TEXT_COLOR = [255, 255, 255, 255]
SITE_TEXT_OUTLINE = [0, 0, 0, 255]
MARKER_LINE_COLOR = [255, 255, 255, 200]
# Texas itself: a faint landmass wash with a clear edge, so the state reads as a
# shape without competing with the flag colours on top of it.
STATE_FILL_COLOR = [70, 90, 115, 60]
STATE_LINE_COLOR = [150, 180, 210, 190]

COAST_LINE_COLOR = [96, 165, 250, 170]
COAST_POINT_COLOR = [96, 165, 250, 220]


def _hex_to_rgb(value: str) -> list[int]:
    value = value.lstrip("#")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


# Hover card. pydeck interpolates {field} against the hovered row, and a deck has
# one tooltip template shared by every pickable layer. Rather than emit empty
# divs on whichever layer lacks a field, both frames carry the same two columns:
# `tooltip_title` and `tooltip_body` (pre-rendered HTML). The template is then
# trivial and always resolves.
TOOLTIP_HTML = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            font-size:12px;line-height:1.5;max-width:280px">
  <div style="font-weight:700;font-size:14px;margin-bottom:4px">{tooltip_title}</div>
  {tooltip_body}
</div>
"""


def _number(value: object, spec: str) -> str:
    """Format a number for the hover card, showing gaps as such.

    A NaN rendered by an f-string reads as the literal "nan", which in a decision
    tool looks like a value rather than a missing one.
    """
    number = float(value)  # type: ignore[arg-type]
    return format(number, spec) if math.isfinite(number) else "n/a"


def _basin_tooltip_body(record: dict, index: float) -> str:
    return (
        f'<div style="opacity:0.75;margin-bottom:6px">{record["flag"]}'
        f" &middot; Capture Index {index:.2f}</div>"
        f'<div>Expected capturable: <b>'
        f'{_number(record["expected_capturable_af"], ",.0f")} AF</b></div>'
        f'<div>Excess volume: {_number(record["median_excess_af"], ",.0f")} AF median,'
        f' {_number(record["q90_excess_af"], ",.0f")} AF p90</div>'
        f'<div>P(HMF in window): {_number(record["event_probability"], ".2f")}</div>'
        f'<div style="margin-top:6px;opacity:0.75">Binding now: '
        f'<code>{record["binding_constraint"]}</code></div>'
        f'<div style="opacity:0.75">If captured: '
        f'<code>{record["binding_if_captured"]}</code></div>'
    )


TOOLTIP_STYLE = {
    "backgroundColor": "rgba(255,255,255,0.97)",
    "color": "#111827",
    "border": "1px solid rgba(0,0,0,0.12)",
    "borderRadius": "8px",
    "padding": "10px 12px",
    "boxShadow": "0 4px 16px rgba(0,0,0,0.18)",
}


REQUIRED_COLUMNS = (
    "basin_id",
    "basin_name",
    "capture_index",
    "flag",
    "binding_constraint",
    "binding_if_captured",
    "expected_capturable_af",
    "median_excess_af",
    "q90_excess_af",
    "event_probability",
)


def build_basin_frame(screening: pd.DataFrame, geography: Geography) -> pd.DataFrame:
    """Join the screening table to map anchors, one row per basin.

    Basins with no geometry are dropped rather than guessed at, and the caller is
    expected to say so if the count changes.

    Raises KeyError, naming every missing column at once, if the screening table
    does not carry the fields the hover card reports. Failing here beats failing
    per-row deep inside the layer construction.
    """
    if screening.empty:
        return pd.DataFrame()

    missing = [name for name in REQUIRED_COLUMNS if name not in screening.columns]
    if missing:
        raise KeyError(f"screening table is missing columns: {', '.join(missing)}")

    rows = []
    for record in screening.to_dict("records"):
        geom = geography.basins.get(record["basin_id"])
        if geom is None:
            continue
        # A non-finite index would serialise as a bare NaN in the layer JSON, and
        # deck.gl then drops the marker with no error: the basin silently
        # vanishes from the map. Treat it as the bottom of the scale instead.
        index = float(record["capture_index"])
        if not math.isfinite(index):
            index = 0.0
        index = min(max(index, 0.0), 1.0)
        flag = record["flag"]
        rows.append(
            {
                "basin_id": record["basin_id"],
                "basin_name": record["basin_name"],
                "lon": geom.centroid.lon,
                "lat": geom.centroid.lat,
                "outlet_lon": geom.outlet.lon,
                "outlet_lat": geom.outlet.lat,
                "reaches_coast": geom.reaches_coast,
                "capture_index": index,
                "capture_index_text": f"{index:.2f}",
                "flag": flag,
                "color": _hex_to_rgb(FLAG_COLORS.get(flag, "#888888")),
                "radius": MIN_RADIUS_M + index * (MAX_RADIUS_M - MIN_RADIUS_M),
                "binding_constraint": record["binding_constraint"],
                "binding_if_captured": record["binding_if_captured"],
                "tooltip_title": record["basin_name"],
                "tooltip_body": _basin_tooltip_body(record, index),
            }
        )
    return pd.DataFrame(rows)


def build_site_frame(geography: Geography) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_id": geom.id,
                "label": geom.label,
                "lon": geom.location.lon,
                "lat": geom.location.lat,
                "tooltip_title": geom.label,
                "tooltip_body": (
                    '<div style="opacity:0.75">Decision unit. Infrastructure '
                    "capacities are illustrative placeholders.</div>"
                ),
            }
            for geom in geography.sites.values()
        ]
    )


def basin_map(
    screening: pd.DataFrame,
    geography: Geography,
    *,
    highlight_coastal: bool = False,
) -> tuple[pdk.Deck | None, int]:
    """Build the deck and report how many basins were placed.

    ``highlight_coastal`` outlines the basins that discharge to the Gulf and
    draws a line from each anchor to its outlet. There is no separate Texas
    Coast decision unit in this repository, so "the coast" can only be shown as
    a property of these basins, not as a region of its own.
    """
    frame = build_basin_frame(screening, geography)
    if frame.empty:
        return None, 0

    layers: list[pdk.Layer] = []

    # Texas itself, drawn from vendored real geometry rather than relying on the
    # basemap. Tiles need network access and a working WebGL tile renderer; this
    # polygon is part of the deck, so the map still reads as Texas (and its Gulf
    # coast) when they are unavailable.
    outline = load_texas_outline()
    if outline:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": ring} for ring in outline],
                get_polygon="polygon",
                get_fill_color=STATE_FILL_COLOR,
                get_line_color=STATE_LINE_COLOR,
                line_width_min_pixels=1.5,
                stroked=True,
                filled=True,
                pickable=False,
            )
        )

    if highlight_coastal:
        coastal = frame[frame["reaches_coast"]]
        if not coastal.empty:
            # Anchor -> outlet, so "reaches the coast" is visible rather than
            # asserted in a caption.
            layers.append(
                pdk.Layer(
                    "LineLayer",
                    data=coastal,
                    get_source_position="[lon, lat]",
                    get_target_position="[outlet_lon, outlet_lat]",
                    get_color=COAST_LINE_COLOR,
                    get_width=2,
                    pickable=False,
                )
            )
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=coastal,
                    get_position="[outlet_lon, outlet_lat]",
                    get_fill_color=COAST_POINT_COLOR,
                    get_radius=9_000,
                    pickable=False,
                )
            )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=frame,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=6,
            radius_max_pixels=90,
            stroked=True,
            get_line_color=MARKER_LINE_COLOR,
            line_width_min_pixels=1.5,
            opacity=0.82,
            pickable=True,
            auto_highlight=True,
            highlight_color=[250, 204, 21, 235],
        )
    )

    sites = build_site_frame(geography)
    if not sites.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=sites,
                get_position="[lon, lat]",
                get_fill_color=SITE_COLOR,
                get_radius=6_500,
                radius_min_pixels=4,
                stroked=True,
                get_line_color=SITE_LINE_COLOR,
                line_width_min_pixels=1.5,
                pickable=True,
                auto_highlight=True,
            )
        )
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=sites,
                get_position="[lon, lat]",
                get_text="label",
                get_size=11,
                get_color=SITE_TEXT_COLOR,
                get_pixel_offset=[0, -16],
                get_text_anchor="'middle'",
                get_alignment_baseline="'bottom'",
                font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                outline_width=3,
                outline_color=SITE_TEXT_OUTLINE,
                font_settings={"sdf": True},
                pickable=False,
            )
        )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=geography.view.latitude,
            longitude=geography.view.longitude,
            zoom=geography.view.zoom,
        ),
        tooltip={"html": TOOLTIP_HTML, "style": TOOLTIP_STYLE},
        map_provider=BASEMAP_PROVIDER,
        map_style=BASEMAP_STYLE,
    )
    return deck, len(frame)
