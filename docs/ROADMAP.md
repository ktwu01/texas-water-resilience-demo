# From this scaffold to a decision-support tool

The proposal frames the goal as movement from analytical integration (ARL 2) to
decision-support validation in a relevant operational environment (ARL 6). This
repository sits at the bottom of that path: the chain closes end to end, under
test, on synthetic data. Here is what each step up actually requires.

## Step 1: real observations behind the same interface

Implement `twr.ingest.load_observed()` so it returns the columns in
`synth.SENSOR_COLUMNS`. Nothing downstream changes. Order of work:

1. Public gauge discharge first, because it is open, low latency, and defines the
   target. Everything else is optional until the target is real.
2. IMERG Late Run, aggregated to daily over basin polygons.
3. SMAP `SPL3SMP_E`, basin-mean, with a real gap-handling policy.
4. ECOSTRESS ET, accepting that gaps correlate with the storms that matter.
5. SWOT last, and as event confirmation rather than a forecast input. Its revisit
   cannot serve a 7-day decision.

The moment real gauges arrive, re-run `leave_one_basin_out`. Expect the skill to
drop substantially. That drop is information, not a bug.

## Step 2: an honest forecast

The current model is antecedent-conditions-only, which is a narrower claim than
"flood forecast" (see LIMITATIONS.md section 2). Adding NWP precipitation
forecasts as features is the single largest available skill gain. It also opens
the largest available leakage hole: forecast fields are indexed by valid time,
and using a valid time inside the target window silently inflates skill. Extend
the leakage tests in `tests/test_features_and_model.py` before adding the
features, not after.

## Step 3: a real regulatory model

Replace the multiplicative caricature in `constraints.py` with:

- Adopted environmental flow standards per basin, seasonal, with subsistence
  / base / pulse tiers and pulse-counting logic;
- water-right accounting with priority dates, so priority calls are representable;
- the surface-water-versus-groundwater ownership distinction, which is the legal
  crux of MAR and ASR in Texas.

Keep the binding-constraint labels. They are the part of the current design that
survives, because "which kind of limit is in the way" is what a manager acts on.

## Step 4: a real aquifer

Swap the lumped bucket for the district's groundwater availability model behind
the same `simulate_storage` interface. Then confront the thing a bucket cannot
represent: recoverability is not storage. Travel time, clogging, water quality,
and well interference decide ASR feasibility in practice.

## Step 5: co-production, which is not a software task

ARL 6 is a claim about *validated decision support with stakeholders*, and no
amount of code earns it. What earns it:

- **Thresholds set by the operators, not the modellers.** The flag boundaries
  (0.20 / 0.45 / 0.70) and every `operational_threshold_af` in
  `config/sites.yaml` are placeholders. Their real values are a judgement about
  cost of mobilisation versus cost of a missed flood, and only the utility can
  make it.
- **A retrospective replay the partners recognise.** Run the flag history over
  historical floods those operators lived through and ask whether the flags match
  what they did and what they wish they had done. `pipeline.run()` already
  supports this via `as_of` and `history_days`; the missing ingredient is the real
  record.
- **An agreed definition of a false alarm and its cost.** A STANDBY that
  mobilises a crew for nothing has a price. Until that price is written down, the
  flag thresholds cannot be tuned and "skill" has no operational meaning.
- **Failure-mode agreement.** Which is worse here, a missed capture or a wasted
  mobilisation? That answer differs between statewide screening and a facility
  operator, and it should change the thresholds per scale.

## Step 6: things that will bite

- **Non-stationarity.** The out-of-bag residual model assumes the future residual
  distribution resembles the past. Compound extremes are precisely the regime
  where it will not. Monitor coverage in operation, not just at fit time.
- **Train/serve skew.** Train on the latency-matched product you will operate on
  (near-real-time precipitation, provisional gauge data), not the research-grade version.
- **Provisional data revision.** Ratings change after floods. A retrospective
  catalogue rebuilt a year later will not match the real-time one, so archive what
  the system actually saw at decision time.
- **Automation bias.** A green flag that is wrong once, expensively, ends the
  tool's credibility. The binding-constraint field and the interval exist so the
  dashboard argues its case instead of asserting a number.
