"""Typed configuration objects loaded from config/*.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class EnvironmentalFlow:
    """Texas-style environmental flow standards, simplified.

    ``pulse_protection_fraction`` is the share of a high-flow pulse that must be
    allowed to pass to sustain channel and estuary function. It is the term that
    most often decides whether a legally available flood is actually capturable.
    """

    subsistence_cfs: float
    base_cfs: float
    pulse_protection_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.pulse_protection_fraction <= 1.0:
            raise ValueError("pulse_protection_fraction must be in [0, 1]")


@dataclass(frozen=True)
class WaterRights:
    unappropriated_fraction: float
    permitted_diversion_af_per_year: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.unappropriated_fraction <= 1.0:
            raise ValueError("unappropriated_fraction must be in [0, 1]")


@dataclass(frozen=True)
class Infrastructure:
    max_diversion_cfs: float
    conveyance_cfs: float
    treatment_mgd: float
    recharge_wells: int
    well_capacity_gpm: float
    storage_capacity_af: float


@dataclass(frozen=True)
class Basin:
    id: str
    name: str
    gauge_id: str
    drainage_area_km2: float
    mean_annual_precip_mm: float
    hmf_percentile: float
    eflow: EnvironmentalFlow
    water_rights: WaterRights
    usgs_site_no: str | None = None


@dataclass(frozen=True)
class Site:
    """A decision unit at one of the three nested scales."""

    id: str
    name: str
    scale: str
    partner: str
    basin_id: str | None
    provenance: str
    operational_threshold_af: float
    infrastructure: Infrastructure
    storage_initial_fraction: float

    @property
    def is_statewide(self) -> bool:
        return self.basin_id is None


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# Retained so existing internal callers keep working.
_read_yaml = read_yaml


def load_basins(path: Path | None = None) -> list[Basin]:
    payload = _read_yaml(path or CONFIG_DIR / "basins.yaml")
    basins = []
    for raw in payload["basins"]:
        basins.append(
            Basin(
                id=raw["id"],
                name=raw["name"],
                gauge_id=raw["gauge_id"],
                usgs_site_no=raw.get("usgs_site_no"),
                drainage_area_km2=float(raw["drainage_area_km2"]),
                mean_annual_precip_mm=float(raw["mean_annual_precip_mm"]),
                hmf_percentile=float(raw["hmf_percentile"]),
                eflow=EnvironmentalFlow(**raw["eflow"]),
                water_rights=WaterRights(**raw["water_rights"]),
            )
        )
    if not basins:
        raise ValueError("no basins defined")
    return basins


def load_sites(path: Path | None = None) -> list[Site]:
    payload = _read_yaml(path or CONFIG_DIR / "sites.yaml")
    known_scales = set(payload.get("scales", {}))
    sites = []
    for raw in payload["sites"]:
        scale = raw["scale"]
        if known_scales and scale not in known_scales:
            raise ValueError(f"site {raw['id']} has unknown scale {scale!r}")
        sites.append(
            Site(
                id=raw["id"],
                name=raw["name"],
                scale=scale,
                partner=raw["partner"],
                basin_id=raw.get("basin_id"),
                provenance=raw.get("provenance", "unknown"),
                operational_threshold_af=float(raw["operational_threshold_af"]),
                infrastructure=Infrastructure(**raw["infrastructure"]),
                storage_initial_fraction=float(raw["storage_initial_fraction"]),
            )
        )
    if not sites:
        raise ValueError("no sites defined")
    return sites


def basins_by_id(basins: list[Basin]) -> dict[str, Basin]:
    return {basin.id: basin for basin in basins}
