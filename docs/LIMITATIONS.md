# Limitations

Read this before showing anyone a number from this repository.

## 1. The data is synthetic

There are no observations here. `twr/synth.py` runs a small conceptual
rainfall-runoff model and degrades its internal states into sensor-like
variables. The relationships an ML model recovers from it are relationships the
simulator was written to contain.

Consequences:

- **Reported skill is an upper bound, not an estimate.** The synthetic world has
  no ungauged inflows, no reservoir operations, no rating-curve error, no
  regulation, no karst. Real Texas basins have all of these. The Guadalupe and
  San Antonio in particular are strongly spring-fed, which this simulator does
  not represent at all.
- **Event magnitudes are plausible in order, not in detail.** Nothing has been
  validated against a real flood.
- Anything that looks like a finding about Texas hydrology is an artefact of the
  generator.

## 2. The forecast is not a forecast

The model predicts the next 7 days from *observed antecedent conditions only*. It
has no meteorological forecast input. In practice most of the skill on a 7-day
horizon for a rainfall-driven flood comes from a numerical weather prediction
forecast, which is absent here.

What the model actually does is estimate how much high-magnitude flow is likely
to emerge from water already in the catchment. That is a real and useful
quantity, and it is a narrower one than "flood forecast". A deployment would add
NWP precipitation forecasts as features, at which point the leakage discipline in
`features.py` needs re-examining, because forecast fields are indexed by valid
time and it is easy to accidentally use a valid time inside the target window.

## 3. Uncertainty is calibrated in-sample-of-basins

Leave-one-basin-out coverage of the nominal 80% interval lands at 0.86 to 0.94,
which is calibrated to somewhat conservative. But:

- Seven synthetic basins drawn from one generator are not seven independent
  hydrologic regimes. Cross-basin transfer looks better here than it would in
  reality.
- The aleatoric term is a resampled out-of-bag residual, which assumes the future
  residual distribution resembles the training one. It will not, under
  non-stationarity, which is the exact regime the proposal is about.
- The mass-balance bound clips a meaningful share of the upper tail on wet days.
  That share is reported (`mass_balance_clipped_fraction`) rather than hidden, and
  a large value is a signal that the log-space model is poorly behaved in the
  tail, not that the bound is wrong.

## 4. The regulatory model is a caricature

Real Texas water law is not a multiplication. The demo reduces it to a single
unappropriated fraction, a single pulse-protection fraction, and a calendar-year
permit volume. Missing entirely:

- prior appropriation and priority dates, so no priority calls;
- seasonal and site-specific environmental flow standards, including separate
  subsistence, base, and multiple pulse tiers;
- interstate compacts, which dominate on the Rio Grande;
- the distinction between surface water rights and groundwater ownership, which
  is the legal crux of MAR and ASR in Texas;
- reservoir system operations and existing contractual obligations.

The binding-constraint labels are the honest part of this module: they tell an
operator *which kind* of limit is in the way. The magnitudes are not defensible.

## 5. Infrastructure numbers are invented

`config/sites.yaml` carries `provenance: illustrative` on every site. The numbers
were chosen so the flag ladder is reachable, and they were wrong in a specific
and instructive way on the first attempt: two sites had operational thresholds
their own well fields could not deliver within the forecast window, which pinned
their Capture Index at zero forever and looked like a hydrologic finding rather
than a configuration error. `pipeline.validate_site_feasibility()` now refuses to
run such a configuration. That guard is worth keeping when real numbers go in.

## 6. The map is not a delineation

`config/geography.yaml` is the one config file whose numbers are real rather than
synthetic: the coordinates are approximate public geography (river mouths, basin
midpoints, and the towns the partner organisations sit in), rounded to about 0.1
degrees. That makes the dashboard map roughly correct in position, and it is
still not a geospatial product.

- **There are no basin boundary polygons.** Basins are drawn as scaled point
  markers, not filled watersheds, because this repository has never delineated a
  watershed. A filled outline would look authoritative and would be invented.
  Replace with Watershed Boundary Dataset HUC polygons and the state's major
  river basin shapefiles before showing this to anyone who works in GIS.
- **The one real polygon is the state outline** in `data/geo/texas_state.geojson`,
  a generalised public-domain national cartographic boundary (~150 vertices). It
  is there so the map reads as Texas even when basemap tiles do not load. Its Gulf
  coastline is simplified and nothing should be measured from it.
- **`centroid` is a hand-placed visual anchor**, not a computed area centroid,
  and it is not the gauge location either. Nothing in the model uses it; it only
  decides where a marker lands.
- **The marker area encodes the Capture Index, not basin size.** A large circle
  means a high index in a possibly small basin. Anyone reading it as drainage
  area will read it backwards.
- The site markers are placed at the partner's town. The infrastructure those
  markers represent is still illustrative, per section 5.

## 7. Super-resolution is a stand-in

`twr/downscale.py` is a ridge regression on patch features, not a CNN. It beats
bilinear interpolation on synthetic fields whose fine structure is a
deterministic function of coarse fields and static terrain. That is a
demonstration of the *problem shape*, not evidence about deep-learning
downscaling of real precipitation, where fine structure is far less predictable
and the evaluation must be done on held-out storms and extremes rather than
pooled RMSE.

## 8. The aquifer is one bucket

A lumped storage bucket with seasonal recovery. No spatial distribution of
recharge, no travel time from a recharge basin to a recovery well, no water
quality, no clogging, no aquifer heterogeneity, no interference between wells.
Recoverability is not the same as storage, and this module does not distinguish
them. ASR feasibility in practice often turns on exactly those omissions.

## 9. ARL claims

The proposal targets movement from ARL 2 to ARL 6, where ARL 6 means validation
in a relevant operational environment with stakeholders. **This repository is
scaffolding at the low end of that range.** It demonstrates that the analysis
chain closes end to end, with tests, on synthetic data. It contains no
stakeholder validation, no real observations, and no operational deployment.
Presenting it as evidence of ARL 6 would be false.
