# Architecture

## The chain

```
NASA EO + gauges          twr/ingest.py       (synthetic in this demo)
        |
        v
conceptual hydrology      twr/synth.py        soil store, routing, sensor noise
        |
        v
causal features           twr/features.py     antecedent storage, API, anomalies
        |                                     + strictly forward-looking targets
        v
hybrid physics-AI         twr/model.py        two boosted heads, bootstrap
        |                                     ensemble, OOB residual sampling
        v
mass-balance bound        twr/constraints.py  runoff <= antecedent rainfall
        |
        v
constraint chain          twr/constraints.py  eflow -> rights -> hardware -> storage
        |
        v
Capture Index + flag      twr/capture_index.py
        |
        v
three nested scales       twr/pipeline.py     TWDB / GCD / utility
        |
        v
scenario evaluation       twr/scenarios.py    hold forecast, vary hardware
        |
        v
dashboard + report        dashboard/app.py, scripts/make_report.py
```

## Where the physics lives

The framework is called hybrid because constraints enter in three distinct
places, not because a neural network is bolted to a model.

1. **In the features.** `storage_deficit_index` is an antecedent storage capacity
   proxy, `api_decay` is an exponentially weighted antecedent precipitation
   index, `flow_ratio_threshold` normalises discharge by the basin's own
   high-flow threshold. These are conceptual hydrologic states, not raw pixels.

2. **As a bound on the learner.** `mass_balance_ceiling_af` caps predicted excess
   volume by the water that has actually fallen on the catchment. This was added
   because the unconstrained model emitted 1.3e8 AF of excess in a 7-day window
   on the Brazos, roughly two orders of magnitude past anything in its training
   data. Fitting in log space and sampling through an exponential produces a tail
   with no physical ceiling, and mass balance is the ceiling.

3. **As a feasibility map on the output.** Every ensemble member is pushed
   through the full constraint chain before the index is computed, so the index
   is a probability over *feasible actions*, never over raw flood volume.

## The Capture Index

```
CI = P( feasible capturable volume >= the site's operational threshold )
```

Estimated as the fraction of predictive samples that survive the constraint
chain above the threshold. Two consequences worth stating plainly:

- A basin can be about to flood and still score zero, because the aquifer is
  full or the water is fully appropriated. That is the point. Hydrology alone
  does not make an opportunity.
- The same forecast produces different indices at the three scales, because each
  has a different threshold and different hardware.

## The constraint chain

Applied in order: hydrology, then law, then steel. The order matters because the
*binding* constraint is what an operator can act on, and ties resolve toward the
earliest link, which is the one they cannot buy their way past.

| Binding label | Meaning | What a manager does |
| --- | --- | --- |
| `no_hmf` | no excess flow at all | nothing |
| `hydrologic_availability` | excess below the operational threshold | nothing |
| `eflow_pulse` | the river needs the pulse | nothing; this is by design |
| `water_rights` | flow is appropriated | talk to TCEQ, not the pump crew |
| `annual_permit` | permit volume spent for the year | permit amendment |
| `diversion_rate` | intake too small | capital project |
| `conveyance` | pipeline too small | capital project |
| `treatment` | treatment throughput | capital project |
| `recharge_capacity` | well field injection limit | drill or rehabilitate wells |
| `storage_headroom` | aquifer full | recover before recharging |

`hydrologic_availability` exists because of a real failure mode. The eflow and
water-rights limits are *proportional* to availability, so on a quiet day they
are always the smallest number in the chain and get reported as binding. Telling
an operator that water rights are blocking them when the river is simply low is
worse than useless, so any excess below the operational threshold is attributed
to hydrology instead.

## Uncertainty

Bootstrap ensembles give **epistemic** uncertainty: how much the fitted function
moves when the training sample is perturbed. On this problem that alone produced
80% intervals covering about 4% of observations, because the members agree
closely on a target that is mostly zero.

So the ensemble also stores **out-of-bag residuals binned by predicted value**,
giving the aleatoric term. Binning matters: the scatter is strongly
heteroscedastic, quiet days are predicted almost exactly and floods are not, and
a pooled residual would manufacture phantom floods on dry days. With both terms,
observed coverage of the nominal 80% interval lands at 0.86 to 0.94 across
leave-one-basin-out folds.

The binding constraint is reported twice, because there are two questions.
`binding_constraint` is evaluated at the unconditional median and answers "what
limits the planning case"; it drives the `BLOCKED` flag, which must distinguish
"water is there and something stops you" from "there is no water".
`binding_if_captured` is evaluated over the ensemble members that actually clear
the threshold and answers "if this materialises, what stops me". Collapsing them
in either direction produced a visible defect: the unconditional view alone gives
rows reading "WATCH, binding: no_hmf", and the conditional view alone labelled
1701 of 2555 statewide days `BLOCKED`.

Evaluation is **leave-one-basin-out**, never random k-fold. Adjacent days in a
daily hydrologic record are near-duplicates, so a random split reports skill that
evaporates the moment the model sees a new basin.

## Leakage discipline

Two rules are enforced in `features.py`, because breaking either produces a model
that looks excellent and is useless.

- **Temporal.** Features at time t use sensor data at times <= t. Targets use
  t+1 to t+horizon. ECOSTRESS gaps are forward-filled only. Tests assert the
  target at index i equals the realised excess over i+1 to i+7.
- **Climatological.** Percentiles, anomalies, and the HMF threshold itself come
  from a fixed baseline period, not the full record. An operator in 2026 does not
  know the 2027 distribution.

## Deliberate simplifications

Every one of these is a place where a real deployment substitutes something
better behind an unchanged interface.

| Demo | Real deployment |
| --- | --- |
| synthetic sensors (`synth.py`) | `ingest.load_observed()`, see DATA_SOURCES.md |
| lumped storage bucket (`aquifer.py`) | the district's calibrated groundwater model |
| ridge-on-patches super-resolution | a CNN; same `fit`/`predict` interface |
| percentile HMF threshold | basin-specific standards, TCEQ pulse definitions |
| single pulse-protection fraction | full HEFR-style seasonal eflow standards |
| calendar-year permit accounting | actual water-right accounting and priority calls |
| illustrative capacities | permit and design-document values |
