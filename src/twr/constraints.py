"""Hydrologic, legal, and infrastructural constraints on capture.

This is the "physics and regulation embedded in the learning process" half of the
hybrid framework. The learner predicts how much high-magnitude water will show
up. This module decides how much of it a manager is actually allowed and able to
put underground, and reports *which* limit binds, because that is the
information that changes what an operator does next:

    binding = eflow_pulse       -> nothing to do, the river needs the pulse
    binding = water_rights      -> go talk to TCEQ, not to your pump crew
    binding = diversion_rate    -> a bigger intake would pay for itself
    binding = recharge_capacity -> drill or rehabilitate wells
    binding = storage_headroom  -> the aquifer is full, recover before recharging

The chain is deliberately applied in that order: hydrology, then law, then steel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EnvironmentalFlow, Infrastructure, WaterRights
from .units import CFS_DAY_TO_AF, GPM_DAY_TO_AF, MGD_TO_AF_PER_DAY, mm_over_area_to_af

BINDING_LABELS = (
    "none",
    "no_hmf",
    "hydrologic_availability",
    "eflow_pulse",
    "water_rights",
    "annual_permit",
    "diversion_rate",
    "conveyance",
    "treatment",
    "recharge_capacity",
    "storage_headroom",
)


@dataclass(frozen=True)
class CaptureLimits:
    """Every limit evaluated for one decision window, in acre-feet."""

    hydrologic_excess_af: float
    after_eflow_af: float
    water_rights_af: float
    annual_permit_af: float
    diversion_af: float
    conveyance_af: float
    treatment_af: float
    recharge_af: float
    storage_headroom_af: float

    def as_dict(self) -> dict[str, float]:
        return {
            "eflow_pulse": self.after_eflow_af,
            "water_rights": self.water_rights_af,
            "annual_permit": self.annual_permit_af,
            "diversion_rate": self.diversion_af,
            "conveyance": self.conveyance_af,
            "treatment": self.treatment_af,
            "recharge_capacity": self.recharge_af,
            "storage_headroom": self.storage_headroom_af,
        }


@dataclass(frozen=True)
class ConstraintResult:
    capturable_af: float
    binding: str
    limits: CaptureLimits
    passed_to_river_af: float
    detail: dict[str, float] = field(default_factory=dict)


def mass_balance_ceiling_af(
    antecedent_precip_mm: float, area_km2: float, max_runoff_fraction: float = 0.8
) -> float:
    """Upper bound on excess volume from water that has actually fallen.

    A statistical learner fitted in log space and sampled through an
    exponential can emit volumes that no amount of rain could produce. In one
    run this model predicted 1.3e8 AF of excess in a 7-day window on the Brazos,
    roughly two orders of magnitude beyond anything in its own training data,
    because the mean of an exponentiated heavy-tailed distribution is both
    unstable and unbounded above.

    This is the mass-balance term: runoff over a window cannot exceed the
    precipitation that fell on the catchment over the antecedent period, times a
    maximum runoff fraction. It is a hard physical bound on the learner's output,
    not a management constraint, so it is applied to the predictive samples
    before the decision chain rather than being reported as a binding limit an
    operator could act on.
    """
    if not 0.0 < max_runoff_fraction <= 1.0:
        raise ValueError("max_runoff_fraction must be in (0, 1]")
    available = mm_over_area_to_af(max(float(antecedent_precip_mm), 0.0), area_km2)
    return available * max_runoff_fraction


def infrastructure_capacity_af(infra: Infrastructure, days: float) -> dict[str, float]:
    """Volume each piece of hardware can handle over the decision window."""
    if days <= 0:
        raise ValueError("days must be positive")
    return {
        "diversion": infra.max_diversion_cfs * days * CFS_DAY_TO_AF,
        "conveyance": infra.conveyance_cfs * days * CFS_DAY_TO_AF,
        "treatment": infra.treatment_mgd * days * MGD_TO_AF_PER_DAY,
        "recharge": infra.recharge_wells * infra.well_capacity_gpm * days * GPM_DAY_TO_AF,
    }


def apply_constraints(
    excess_af: float,
    *,
    eflow: EnvironmentalFlow,
    rights: WaterRights,
    infra: Infrastructure,
    storage_headroom_af: float,
    days: float = 7.0,
    permit_remaining_af: float | None = None,
    materiality_af: float = 0.0,
) -> ConstraintResult:
    """Reduce a predicted HMF excess volume to a feasible capture volume.

    ``excess_af`` is already the volume *above* the high-flow threshold, so base
    and subsistence flows are protected by construction. The pulse-protection
    fraction then reserves part of the flood itself.

    ``materiality_af`` guards the binding-constraint label. The eflow and
    water-rights limits are *proportional* to availability, so on a quiet day
    they are always the smallest number in the chain and would be reported as
    binding. Telling an operator that water rights are blocking them when the
    river is simply low is worse than useless. Any excess at or below
    ``materiality_af`` is therefore attributed to hydrology, which is where it
    belongs. Callers normally pass the site's operational threshold.
    """
    excess_af = float(max(excess_af, 0.0))

    # 1. Hydrology and environmental flow.
    after_eflow = excess_af * (1.0 - eflow.pulse_protection_fraction)

    # 2. Water rights: only unappropriated flow, and only within the annual permit.
    rights_limit = after_eflow * rights.unappropriated_fraction
    if permit_remaining_af is None:
        permit_remaining_af = rights.permitted_diversion_af_per_year
    permit_limit = max(float(permit_remaining_af), 0.0)

    # 3. Infrastructure.
    caps = infrastructure_capacity_af(infra, days)

    # 4. Somewhere to put it.
    headroom = max(float(storage_headroom_af), 0.0)

    limits = CaptureLimits(
        hydrologic_excess_af=excess_af,
        after_eflow_af=after_eflow,
        water_rights_af=rights_limit,
        annual_permit_af=permit_limit,
        diversion_af=caps["diversion"],
        conveyance_af=caps["conveyance"],
        treatment_af=caps["treatment"],
        recharge_af=caps["recharge"],
        storage_headroom_af=headroom,
    )

    candidates = limits.as_dict()
    capturable = min(candidates.values())
    capturable = float(max(capturable, 0.0))

    if excess_af <= 0.0:
        binding = "no_hmf"
    elif excess_af <= max(float(materiality_af), 0.0):
        binding = "hydrologic_availability"
    else:
        # The binding constraint is the smallest limit; ties resolve toward the
        # earliest link in the chain, which is the one a manager cannot buy away.
        order = list(candidates)
        binding = min(order, key=lambda key: (candidates[key], order.index(key)))

    return ConstraintResult(
        capturable_af=capturable,
        binding=binding,
        limits=limits,
        passed_to_river_af=excess_af - capturable,
        detail={"days": float(days), **candidates},
    )


def apply_constraints_vectorised(
    excess_af: np.ndarray,
    *,
    eflow: EnvironmentalFlow,
    rights: WaterRights,
    infra: Infrastructure,
    storage_headroom_af: float,
    days: float = 7.0,
    permit_remaining_af: float | None = None,
) -> np.ndarray:
    """Constraint chain applied to an ensemble of predicted volumes.

    Used to push a whole predictive distribution through the feasibility map,
    which is what makes the Capture Index a probability over *feasible* volumes
    rather than over raw flood volumes.
    """
    excess = np.clip(np.asarray(excess_af, dtype=float), 0.0, None)
    after_eflow = excess * (1.0 - eflow.pulse_protection_fraction)
    rights_limit = after_eflow * rights.unappropriated_fraction

    caps = infrastructure_capacity_af(infra, days)
    permit = (
        rights.permitted_diversion_af_per_year
        if permit_remaining_af is None
        else permit_remaining_af
    )
    ceiling = min(
        max(float(permit), 0.0),
        caps["diversion"],
        caps["conveyance"],
        caps["treatment"],
        caps["recharge"],
        max(float(storage_headroom_af), 0.0),
    )
    return np.clip(np.minimum(rights_limit, ceiling), 0.0, None)


def check_mass_balance(
    excess_af: float, capturable_af: float, passed_af: float, tolerance: float = 1e-6
) -> None:
    """Guard rail: capture plus pass-through must equal the available volume."""
    if capturable_af < -tolerance:
        raise AssertionError("negative capture volume")
    if capturable_af > excess_af + tolerance:
        raise AssertionError("capture exceeds available excess volume")
    if abs((capturable_af + passed_af) - excess_af) > max(tolerance, 1e-6 * abs(excess_af)):
        raise AssertionError("mass balance violated")
