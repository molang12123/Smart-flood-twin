# 0.5–700 mm/hr expansion status

> Historical result: this file records the rejected pre-v6 single-model experiments. The current code defines a new 0–800 mm/hr, 1–24 hour, 36-hour-horizon three-expert v6 workflow with well-balanced finite-volume loss and strict overlap auditing. v6 is not approved until all three expert physics gates and both overlap gates pass; until then the GUI continues to use the v4 fallback.

The requested parameter domain was represented by 108 deterministic design scenarios (12 intensities × 9 durations) in addition to the 151 C0M760 events. Domain edges belong to training; interior combinations include complete held-out validation and test scenarios.

Two full-range models were trained. Neither is published as the production checkpoint because the stricter physics gates rejected them.

| experiment | test mean PDE loss | worst PDE loss | test mean mass loss | worst mass loss | result |
|---|---:|---:|---:|---:|---|
| linear condition encoding | 0.17265 | 0.71334 | 0.000896 | 0.003181 | failed PDE gates |
| logarithmic condition encoding | 0.14293 | 0.63812 | 0.002038 | 0.007679 | failed PDE gates |

Required gates are test mean PDE ≤ 0.02, worst PDE ≤ 0.1, test mean mass ≤ 0.01 and worst mass ≤ 0.05. Both experiments passed water balance and failed momentum/physics residual gates. The official checkpoint therefore remains the validated 0.5–23 mm/hr model.

The implementation now supports deterministic full-range scenario generation, logarithmic rain/duration conditioning, slope-biased collocation, extreme-event residual sampling and worst-event gates. A future implementation should use rain-band experts or a conservative weak-form/domain-decomposed PINN rather than merely increasing the declared checkpoint range.

That follow-up formulation is now implemented in code, including Audusse hydrostatic reconstruction, HLL flux, fixed terrain/time features, hard storage projection, exact resumable training state and overlap auditing. This does not change the failed results above and does not assert that the new training run has passed.
