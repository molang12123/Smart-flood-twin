# 0.5–700 mm/hr expansion status

The requested parameter domain was represented by 108 deterministic design scenarios (12 intensities × 9 durations) in addition to the 151 C0M760 events. Domain edges belong to training; interior combinations include complete held-out validation and test scenarios.

Two full-range models were trained. Neither is published as the production checkpoint because the stricter physics gates rejected them.

| experiment | test mean PDE loss | worst PDE loss | test mean mass loss | worst mass loss | result |
|---|---:|---:|---:|---:|---|
| linear condition encoding | 0.17265 | 0.71334 | 0.000896 | 0.003181 | failed PDE gates |
| logarithmic condition encoding | 0.14293 | 0.63812 | 0.002038 | 0.007679 | failed PDE gates |

Required gates are test mean PDE ≤ 0.02, worst PDE ≤ 0.1, test mean mass ≤ 0.01 and worst mass ≤ 0.05. Both experiments passed water balance and failed momentum/physics residual gates. The official checkpoint therefore remains the validated 0.5–23 mm/hr model.

The implementation now supports deterministic full-range scenario generation, logarithmic rain/duration conditioning, slope-biased collocation, extreme-event residual sampling and worst-event gates. A future implementation should use rain-band experts or a conservative weak-form/domain-decomposed PINN rather than merely increasing the declared checkpoint range.
