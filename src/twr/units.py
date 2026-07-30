"""Unit conversions.

Convention for the whole package:

* flow rates are cubic feet per second (cfs)
* volumes are acre-feet (AF)
* depths are millimetres (mm)
* areas are square kilometres (km2)

Everything here is an exact conversion, so these constants are the only place
where a factor-of-1000 mistake can hide.
"""

from __future__ import annotations

# 1 acre-foot = 43,560 ft3 = 325,851.4 US gallons
FT3_PER_AF = 43_560.0
GAL_PER_AF = 325_851.4

# One cfs sustained for one day.
CFS_DAY_TO_AF = 86_400.0 / FT3_PER_AF  # 1.98347 AF

# One million gallons per day, expressed as AF/day.
MGD_TO_AF_PER_DAY = 1_000_000.0 / GAL_PER_AF  # 3.06888 AF/day

# One gallon per minute sustained for one day.
GPM_DAY_TO_AF = 1_440.0 / GAL_PER_AF  # 0.00441915 AF

M3_TO_AF = 1.0 / 1_233.4818375475  # 1 acre-foot = 1233.48 m3

# One mm/day of runoff over one km2, expressed as cfs.
# 1 mm over 1 km2 = 1,000 m3 -> /86,400 s -> m3/s -> x35.3147 ft3/m3
MM_DAY_KM2_TO_CFS = (1_000.0 / 86_400.0) * 35.314666721  # 0.408734 cfs


def cfs_days_to_af(cfs: float, days: float = 1.0) -> float:
    """Convert a sustained flow rate and duration to a volume."""
    return cfs * days * CFS_DAY_TO_AF


def af_to_cfs(volume_af: float, days: float = 1.0) -> float:
    """Convert a volume delivered over ``days`` into an average flow rate."""
    if days <= 0:
        raise ValueError("days must be positive")
    return volume_af / (days * CFS_DAY_TO_AF)


def mm_to_cfs(depth_mm_per_day: float, area_km2: float) -> float:
    """Convert an areal runoff depth rate to a flow rate."""
    return depth_mm_per_day * area_km2 * MM_DAY_KM2_TO_CFS


def mm_over_area_to_af(depth_mm: float, area_km2: float) -> float:
    """Total volume of a rainfall depth over a catchment, in acre-feet."""
    cubic_metres = (depth_mm / 1_000.0) * (area_km2 * 1_000_000.0)
    return cubic_metres * M3_TO_AF
