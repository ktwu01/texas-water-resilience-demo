"""Probabilistic Capture Index and the operational flag system.

The Capture Index is defined as

    CI = P( feasible capturable volume >= operational threshold )

estimated by pushing every bootstrap ensemble member through the full constraint
chain and counting how many survive above the site's operational threshold. Two
properties follow from that definition and both matter:

* It is a probability of an *action being worth taking*, not a probability of
  rain. A basin can be about to flood and still score zero because the aquifer
  is full or the water is fully appropriated.
* It is scale-aware. The same forecast yields different indices for statewide
  screening, a groundwater district, and a single ASR facility, because each has
  a different threshold and different hardware.

Flags are a coarsening of CI into the four states an operator can act on. The
coarsening is deliberate: a dashboard that shows 0.63 invites debate, a
dashboard that shows STANDBY tells a crew what to do this week.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import EnvironmentalFlow, Infrastructure, WaterRights
from .constraints import apply_constraints, apply_constraints_vectorised

# Ordered from least to most urgent.
FLAG_ORDER = ("NO_ACTION", "WATCH", "STANDBY", "CAPTURE")
FLAG_THRESHOLDS = {"WATCH": 0.20, "STANDBY": 0.45, "CAPTURE": 0.70}
FLAG_BLOCKED = "BLOCKED"

FLAG_COLORS = {
    "NO_ACTION": "#9aa4ad",
    "WATCH": "#2e7fd4",
    "STANDBY": "#e0a020",
    "CAPTURE": "#1e9e63",
    FLAG_BLOCKED: "#b23a48",
}

FLAG_ACTIONS = {
    "NO_ACTION": "No capture opportunity in the forecast window. Routine monitoring.",
    "WATCH": "Possible opportunity. Confirm aquifer headroom and check gauge trends daily.",
    "STANDBY": "Likely opportunity. Pre-position pumps, notify operators, confirm permit status.",
    "CAPTURE": "High capture potential. Execute diversion and recharge within the window.",
    FLAG_BLOCKED: "Water may be present but capture is not feasible. See binding constraint.",
}

# Constraints that no amount of favourable hydrology can overcome inside the
# forecast window. If one of these zeroes out the capture volume, the flag is
# BLOCKED rather than NO_ACTION, so the reason reaches the manager.
BLOCKING_CONSTRAINTS = ("storage_headroom", "annual_permit", "water_rights", "eflow_pulse")


@dataclass
class CaptureAssessment:
    """One site, one forecast window."""

    site_id: str
    basin_id: str | None
    date: str
    horizon_days: int
    capture_index: float
    flag: str
    action: str
    event_probability: float
    expected_capturable_af: float
    q10_capturable_af: float
    q50_capturable_af: float
    q90_capturable_af: float
    # Excess volume is reported as a median and an upper quantile, never a mean.
    # The predictive distribution is lognormal-ish and heavy tailed, so its mean
    # is dominated by a few extreme draws and is not a number to plan against.
    median_excess_af: float
    q90_excess_af: float
    mass_balance_ceiling_af: float
    # Share of predictive samples the mass-balance bound had to clip. A large
    # value is not a failure of the bound, it is the learner's log-space tail
    # reaching past what the catchment holds, and it belongs on the record.
    mass_balance_clipped_fraction: float
    # What limits the planning case. Drives the BLOCKED flag.
    binding_constraint: str
    # What would limit you if the opportunity materialises. Drives preparation.
    binding_if_captured: str
    storage_headroom_af: float
    operational_threshold_af: float
    ensemble_size: int
    limits_af: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def assign_flag(capture_index: float, binding: str) -> str:
    """Map a Capture Index and its binding constraint to an operational flag."""
    if not 0.0 <= capture_index <= 1.0:
        raise ValueError("capture_index must be in [0, 1]")
    if capture_index < FLAG_THRESHOLDS["WATCH"] and binding in BLOCKING_CONSTRAINTS:
        return FLAG_BLOCKED
    if capture_index >= FLAG_THRESHOLDS["CAPTURE"]:
        return "CAPTURE"
    if capture_index >= FLAG_THRESHOLDS["STANDBY"]:
        return "STANDBY"
    if capture_index >= FLAG_THRESHOLDS["WATCH"]:
        return "WATCH"
    return "NO_ACTION"


def assess(
    *,
    site_id: str,
    basin_id: str | None,
    date: str,
    volume_samples_af: np.ndarray,
    event_probability: float,
    eflow: EnvironmentalFlow,
    rights: WaterRights,
    infra: Infrastructure,
    storage_headroom_af: float,
    operational_threshold_af: float,
    horizon_days: int = 7,
    permit_remaining_af: float | None = None,
    mass_balance_ceiling_af: float | None = None,
) -> CaptureAssessment:
    """Turn an ensemble volume forecast into a Capture Index and a flag."""
    samples = np.asarray(volume_samples_af, dtype=float).ravel()
    if samples.size == 0:
        raise ValueError("volume_samples_af is empty")
    if operational_threshold_af <= 0:
        raise ValueError("operational_threshold_af must be positive")

    # Physical plausibility bound on the learner before any decision logic.
    ceiling = float("inf") if mass_balance_ceiling_af is None else float(mass_balance_ceiling_af)
    if ceiling < 0:
        raise ValueError("mass_balance_ceiling_af must be non-negative")
    clipped_fraction = float(np.mean(samples > ceiling)) if np.isfinite(ceiling) else 0.0
    samples = np.minimum(samples, ceiling)

    feasible = apply_constraints_vectorised(
        samples,
        eflow=eflow,
        rights=rights,
        infra=infra,
        storage_headroom_af=storage_headroom_af,
        days=horizon_days,
        permit_remaining_af=permit_remaining_af,
    )

    qualifying = feasible >= operational_threshold_af
    capture_index = float(np.mean(qualifying))

    # Two different questions, two fields. Conflating them is a mistake in both
    # directions, and this code has made both.
    #
    # `binding_constraint` answers "what limits the planning case", evaluated at
    # the unconditional median. It drives the BLOCKED flag, because BLOCKED must
    # mean "water is there and something is stopping you", and only a central
    # measure can distinguish that from "there is no water". Using the
    # conditional median here labelled 1701 of 2555 statewide days BLOCKED.
    #
    # `binding_if_captured` answers "if this opportunity does materialise, what
    # will stop me taking it", evaluated over the members that actually clear the
    # threshold. Using the unconditional median for this produced dashboard rows
    # reading "WATCH ... binding: no_hmf", since the median of a heavy-tailed
    # forecast is zero even when a quarter of the ensemble shows a real flood.
    median_excess = float(np.median(samples))
    chain_kwargs = {
        "eflow": eflow,
        "rights": rights,
        "infra": infra,
        "storage_headroom_af": storage_headroom_af,
        "days": horizon_days,
        "permit_remaining_af": permit_remaining_af,
        # Below the operational threshold there is nothing to allocate, so
        # hydrology owns the label rather than eflow or water rights.
        "materiality_af": operational_threshold_af,
    }
    median_result = apply_constraints(median_excess, **chain_kwargs)
    if qualifying.any():
        conditional = apply_constraints(float(np.median(samples[qualifying])), **chain_kwargs)
        binding_if_captured = conditional.binding
    else:
        binding_if_captured = median_result.binding

    flag = assign_flag(capture_index, median_result.binding)

    return CaptureAssessment(
        site_id=site_id,
        basin_id=basin_id,
        date=str(date),
        horizon_days=horizon_days,
        capture_index=capture_index,
        flag=flag,
        action=FLAG_ACTIONS[flag],
        event_probability=float(event_probability),
        expected_capturable_af=float(np.mean(feasible)),
        q10_capturable_af=float(np.quantile(feasible, 0.10)),
        q50_capturable_af=float(np.quantile(feasible, 0.50)),
        q90_capturable_af=float(np.quantile(feasible, 0.90)),
        median_excess_af=median_excess,
        q90_excess_af=float(np.quantile(samples, 0.90)),
        mass_balance_ceiling_af=ceiling if np.isfinite(ceiling) else float("nan"),
        mass_balance_clipped_fraction=clipped_fraction,
        binding_constraint=median_result.binding,
        binding_if_captured=binding_if_captured,
        storage_headroom_af=float(storage_headroom_af),
        operational_threshold_af=float(operational_threshold_af),
        ensemble_size=int(samples.size),
        limits_af=median_result.limits.as_dict(),
    )


def flag_rank(flag: str) -> int:
    """Sortable urgency rank. BLOCKED sorts just above NO_ACTION."""
    if flag == FLAG_BLOCKED:
        return 1
    if flag not in FLAG_ORDER:
        raise ValueError(f"unknown flag {flag!r}")
    return FLAG_ORDER.index(flag) * 2
