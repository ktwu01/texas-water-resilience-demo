"""Where things are: a map layer for the statewide screening view.

Two kinds of geography live here and they have very different provenance, so
they are kept apart on purpose.

* The Texas outline in ``data/geo/texas_boundary.json`` is real. It is the US
  Census 2022 cartographic boundary file at 1:20,000,000, simplified, public
  domain. It is cartographic and not survey grade, which is fine for a map that
  exists to answer "which part of the state is this row about".

* The point locations below are NOT real. They are single representative
  coordinates standing in for whole river basins and for districts that cover
  several counties. A basin is a polygon, not a dot, and the dot is placed by
  eye near the mid-basin reach the synthetic gauge is meant to evoke. They are
  labelled ``provenance: approximate`` for the same reason the infrastructure
  numbers in ``config/sites.yaml`` are labelled ``illustrative``: anyone wiring
  this to a real operation must replace them with the actual gauge coordinates
  (USGS site numbers) and the actual basin and district boundaries (TWDB major
  river basins, GCD service areas) before any of it is put in front of an
  operator.

Nothing here feeds the model. The map is a read-out of numbers computed
upstream, so a wrong dot misleads a viewer but cannot bias a forecast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .capture_index import FLAG_COLORS
from .config import DATA_DIR

BOUNDARY_PATH = DATA_DIR / "geo" / "texas_boundary.json"

#: Approximate mid-basin points for the seven synthetic gauges, (lat, lon).
#: Placeholders. See the module docstring.
BASIN_POINTS: dict[str, tuple[float, float]] = {
    "brazos": (31.20, -97.40),
    "colorado_tx": (30.60, -98.40),
    "trinity": (32.30, -96.30),
    "guadalupe": (29.60, -97.60),
    "nueces": (28.40, -98.50),
    "san_antonio": (29.30, -98.05),
    "rio_grande": (29.30, -101.00),
}

#: Approximate points for the three decision units, (lat, lon). Placeholders.
#: ``twdb_statewide`` has no single location by construction; it screens every
#: basin, so it is deliberately absent rather than pinned to a centroid.
SITE_POINTS: dict[str, tuple[float, float]] = {
    "rolling_plains_gcd": (33.20, -99.90),
    "kerrville_asr": (30.05, -99.14),
}

#: Hand-placed label offsets in typographic points, ``(dx, dy, ha, va)``.
#: The Guadalupe, San Antonio, and Nueces gauges sit within about a degree of
#: each other, so an automatic "label below the marker" rule stacks three names
#: on top of one another and the map becomes unreadable. Seven basins is few
#: enough that placing labels by hand beats a collision solver.
LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "brazos": (10, 0, "left", "center"),
    "colorado_tx": (-10, -2, "right", "center"),
    "trinity": (10, 2, "left", "center"),
    "guadalupe": (10, 4, "left", "bottom"),
    "san_antonio": (-10, 2, "right", "center"),
    "nueces": (-6, -8, "right", "top"),
    "rio_grande": (-4, -12, "right", "top"),
}
_DEFAULT_LABEL_OFFSET = (10.0, 0.0, "left", "center")

SITE_LABEL_OFFSETS: dict[str, tuple[float, float, str, str]] = {
    "rolling_plains_gcd": (10, 2, "left", "center"),
    "kerrville_asr": (-9, -6, "right", "top"),
}

PROVENANCE = "approximate"

_MARKER_MIN = 40.0
_MARKER_MAX = 460.0


@dataclass(frozen=True)
class Boundary:
    """A vendored state outline, with its own provenance attached."""

    name: str
    source: str
    rings: tuple[np.ndarray, ...]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat) over every ring."""
        stacked = np.vstack(self.rings)
        return (
            float(stacked[:, 0].min()),
            float(stacked[:, 1].min()),
            float(stacked[:, 0].max()),
            float(stacked[:, 1].max()),
        )


def load_boundary(path: Path | None = None) -> Boundary:
    """Read the vendored Texas outline. Offline: no network at build time."""
    path = Path(path) if path is not None else BOUNDARY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The vendored Texas outline is tracked in the repo; "
            "see src/twr/geo.py for its source."
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    rings = tuple(np.asarray(ring, dtype=float) for ring in doc["rings"])
    if not rings or any(ring.ndim != 2 or ring.shape[1] != 2 for ring in rings):
        raise ValueError(f"{path} does not contain [lon, lat] rings")
    return Boundary(name=doc.get("name", "Texas"), source=doc.get("source", "unknown"), rings=rings)


def locate_basins(statewide: pd.DataFrame) -> pd.DataFrame:
    """Attach approximate coordinates to statewide screening rows.

    Basins with no known point are dropped and reported by the caller rather
    than silently placed at (0, 0) in the Gulf of Guinea.
    """
    if statewide.empty:
        return statewide.assign(lat=pd.Series(dtype=float), lon=pd.Series(dtype=float))
    frame = statewide.copy()
    frame["lat"] = frame["basin_id"].map(lambda b: BASIN_POINTS.get(b, (np.nan, np.nan))[0])
    frame["lon"] = frame["basin_id"].map(lambda b: BASIN_POINTS.get(b, (np.nan, np.nan))[1])
    frame["geo_provenance"] = PROVENANCE
    return frame


def unlocated_basins(statewide: pd.DataFrame) -> list[str]:
    """Basin ids present in the screening output but missing from BASIN_POINTS."""
    if statewide.empty or "basin_id" not in statewide.columns:
        return []
    return sorted(set(statewide["basin_id"]) - set(BASIN_POINTS))


def marker_sizes(values: pd.Series) -> np.ndarray:
    """Scale a volume column to marker areas.

    Square-root scaling, because a marker's *area* is what a reader perceives
    as magnitude. Linear area scaling on a heavy-tailed volume distribution
    turns one flood into a dot that covers the state.
    """
    array = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    top = float(array.max()) if array.size else 0.0
    if top <= 0.0:
        return np.full(array.shape, _MARKER_MIN)
    return _MARKER_MIN + (_MARKER_MAX - _MARKER_MIN) * np.sqrt(array / top)


def render_texas_map(
    statewide: pd.DataFrame,
    out_path: Path,
    *,
    site_flags: pd.DataFrame | None = None,
    boundary_path: Path | None = None,
) -> Path | None:
    """Draw the statewide screening as a map and write it to ``out_path``.

    Colour is the operational flag, so the map and the flag cards cannot
    disagree. Marker area is expected capturable volume. Returns ``None`` if
    there is nothing locatable to draw.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    located = locate_basins(statewide).dropna(subset=["lat", "lon"])
    if located.empty:
        return None

    boundary = load_boundary(boundary_path)
    min_lon, min_lat, max_lon, max_lat = boundary.bounds

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    for ring in boundary.rings:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#f2f5f7", edgecolor="#9aa7b1", lw=0.9, zorder=1)

    sizes = marker_sizes(located.get("expected_capturable_af", pd.Series(0.0, index=located.index)))
    colors = [FLAG_COLORS.get(flag, "#888888") for flag in located.get("flag", [])]
    ax.scatter(
        located["lon"], located["lat"], s=sizes, c=colors,
        edgecolor="#33404a", linewidth=0.7, zorder=3, alpha=0.92,
    )
    for (_, row), size in zip(located.iterrows(), sizes, strict=True):
        label = f"{row.get('basin_name', row['basin_id'])}\nCI {row['capture_index']:.2f}"
        dx, dy, ha, va = LABEL_OFFSETS.get(row["basin_id"], _DEFAULT_LABEL_OFFSET)
        # Push the label clear of the marker's own radius, which varies per row.
        clearance = np.sqrt(size) / 2.0
        ax.annotate(
            label, (row["lon"], row["lat"]),
            xytext=(dx + np.sign(dx) * clearance, dy - (clearance if va == "top" else 0.0)),
            textcoords="offset points",
            ha=ha, va=va, fontsize=7.5, color="#2b343c", zorder=4,
        )

    if site_flags is not None and not site_flags.empty:
        for _, row in site_flags.iterrows():
            point = SITE_POINTS.get(row["site_id"])
            if point is None:
                continue
            lat, lon = point
            ax.scatter(
                [lon], [lat], marker="^", s=110,
                c=[FLAG_COLORS.get(row["flag"], "#888888")],
                edgecolor="#1c2126", linewidth=0.9, zorder=5,
            )
            dx, dy, ha, va = SITE_LABEL_OFFSETS.get(row["site_id"], _DEFAULT_LABEL_OFFSET)
            ax.annotate(
                row["site_id"], (lon, lat), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=7.5, style="italic", color="#1c2126", zorder=5,
            )

    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=color,
                   markeredgecolor="#33404a", label=flag)
        for flag, color in FLAG_COLORS.items()
    ]
    handles.append(
        plt.Line2D([], [], marker="^", ls="", markersize=8, markerfacecolor="#ffffff",
                   markeredgecolor="#1c2126", label="decision unit")
    )
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5, ncols=2)

    pad = 0.6
    ax.set_xlim(min_lon - pad, max_lon + pad)
    ax.set_ylim(min_lat - pad, max_lat + pad)
    ax.set_aspect(1 / np.cos(np.radians((min_lat + max_lat) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_title(
        "Where is high-magnitude flow capturable this week?\n"
        "circle area = expected capturable volume, colour = operational flag",
        loc="left", fontsize=10,
    )
    ax.text(
        0.0, -0.02,
        "Outline: US Census 2022 cartographic boundary (public domain), simplified.\n"
        "Basin and site markers are approximate placeholder locations, not gauge "
        "coordinates. All values are synthetic.",
        transform=ax.transAxes, fontsize=7, color="#6b757e", va="top",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
