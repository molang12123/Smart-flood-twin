# Theoretical PINN improvement audit

## Outcome

The published v4 checkpoint remains the default model. No expanded 0.5–700 mm/hr experiment passed all strict physics gates, so no experimental checkpoint replaced it.

## Experiments

| formulation | rain range (mm/hr) | test mean PDE | worst PDE | test mean mass | worst mass | result |
|---|---:|---:|---:|---:|---:|---|
| strong, single model, linear conditions | 0.5–700 | 0.17265 | 0.71334 | 0.000896 | 0.003181 | failed PDE |
| strong, single model, log conditions | 0.5–700 | 0.14293 | 0.63812 | 0.002038 | 0.007679 | failed PDE |
| finite-volume expert, low | 0.5–30 | 0.05187 | 0.15535 | 0.002125 | 0.010499 | failed PDE |
| finite-volume expert, medium | 20–200 | 0.15683 | 0.28292 | 0.001284 | 0.003277 | failed PDE |
| finite-volume expert, high | 150–700 | 0.21784 | 0.35867 | 0.000637 | 0.001996 | failed PDE |

Gates: test mean PDE ≤ 0.02, worst PDE ≤ 0.1, test mean mass ≤ 0.01 and worst mass ≤ 0.05.

## What was improved

- Deterministic design scenarios cover 0.5–700 mm/hr and 1–37 hours with complete held-out combinations.
- Logarithmic condition encoding preserves resolution across a 1400× rainfall range.
- Slope-biased and high-residual sampling expose errors hidden by uniform sampling.
- Multi-time relative water balance prevents a large depth scale from hiding volume error.
- A local finite-volume loss directly checks storage change, four face fluxes, rainfall, momentum and friction.
- Flat uniform rain is an automated exact benchmark.

## Identified limitation

The experimental finite-volume momentum residual uses midpoint face fluxes and a centre bed-slope source. On steep terrain and wet/dry fronts this is not a well-balanced discretization: pressure flux and terrain source do not cancel exactly for lake-at-rest states. That creates an irreducible residual which longer optimization cannot fix safely.

The next numerical formulation must implement hydrostatic reconstruction or an equivalent well-balanced conservative weak form, test lake-at-rest with wet/dry shorelines, and only then retrain rain-band experts. Thresholds must not be relaxed merely to publish a 700 mm/hr range.
