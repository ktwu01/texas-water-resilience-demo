"""Interactive basin map with hover tooltips, for the statewide screening tab.

Rendered with pydeck, which ships with Streamlit, so this adds no dependency and
needs no API key. The basemap is CARTO's free raster tile service.

WHY NOT GOOGLE MAPS: the Google Maps JavaScript API requires a billed API key,
which cannot be committed to a public repository, and it would make the dashboard
fail closed for anyone cloning the repo. Nothing here needs Google's routing,
places, or Street View data; the requirement is "show me where these basins are
and let me hover one", which an open basemap satisfies. If a deployment needs
Google specifically, the tile URL below is the single line to change.

WHAT THE MARKERS MEAN: each basin is a point at a hand-placed visual anchor, not
a filled watershed polygon, because this repository has not delineated basin
boundaries. Radius encodes the Capture Index. See config/geography.yaml.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk

from twr.capture_index import FLAG_COLORS
from twr.geo import Geography

# CARTO Voyager. Light and low-chroma, so the flag colours stay readable, but it
# still labels cities and rivers, which is what makes the map orienting.
BASEMAP_URL = "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
BASEMAP_ATTRIBUTION = "Basemap (c) OpenStreetMap contributors, (c) CARTO"

# Point radius in metres. A Capture Index of 0 still has to be visible and
# hoverable, hence the floor.
MIN_RADIUS_M = 16_000
MAX_RADIUS_M = 62_000

SITE_COLOR = [17, 24, 39]


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


def _basin_tooltip_body(record: dict, index: float) -> str:
    return (
        f'<div style="opacity:0.75;margin-bottom:6px">{record["flag"]}'
        f" &middot; Capture Index {index:.2f}</div>"
        f'<div>Expected capturable: <b>{float(record["expected_capturable_af"]):,.0f}'
        f" AF</b></div>"
        f'<div>Excess volume: {float(record["median_excess_af"]):,.0f} AF median,'
        f' {float(record["q90_excess_af"]):,.0f} AF p90</div>'
        f'<div>P(HMF in window): {float(record["event_probability"]):.2f}</div>'
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


def build_basin_frame(screening: pd.DataFrame, geography: Geography) -> pd.DataFrame:
    """Join the screening table to map anchors, one row per basin.

    Basins with no geometry are dropped rather than guessed at, and the caller is
    expected to say so if the count changes.
    """
    if screening.empty:
        return pd.DataFrame()

    rows = []
    for record in screening.to_dict("records"):
        geom = geography.basins.get(record["basin_id"])
        if geom is None:
            continue
        index = float(record["capture_index"])
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

    layers = [
        pdk.Layer(
            "TileLayer",
            data=BASEMAP_URL,
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            opacity=1.0,
        )
    ]

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
                    get_color=[37, 99, 235, 130],
                    get_width=2,
                    pickable=False,
                )
            )
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=coastal,
                    get_position="[outlet_lon, outlet_lat]",
                    get_fill_color=[37, 99, 235, 200],
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
            get_line_color=[255, 255, 255, 235],
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
                get_line_color=[255, 255, 255, 255],
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
                get_color=[17, 24, 39, 255],
                get_pixel_offset=[0, -16],
                get_text_anchor="'middle'",
                get_alignment_baseline="'bottom'",
                font_family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                outline_width=3,
                outline_color=[255, 255, 255, 255],
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
        map_provider=None,
    )
    return deck, len(frame)
