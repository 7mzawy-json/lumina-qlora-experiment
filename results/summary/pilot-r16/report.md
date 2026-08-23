# Lumina Paired Evaluation Report

Evidence state: `8b_pilot_measured`

> Directional 8B pilot measurement; this report does not claim completion of the full frozen benchmark.

## Primary metrics

| Metric | Adapter minus base (percentage points) | 95% paired bootstrap CI |
|---|---:|---:|
| instruction_following | -33.33 | [-33.33, -33.33] |
| json_schema_validity | +0.00 | [+0.00, +0.00] |

## Secondary metrics

| Metric | Adapter minus base (percentage points) | 95% paired bootstrap CI |
|---|---:|---:|
| knowledge | -33.33 | [-100.00, +0.00] |
| programming | -33.33 | [-100.00, +0.00] |
| reasoning | +0.00 | [+0.00, +0.00] |
| summarization | -33.73 | [-100.00, +10.53] |

## Concealed writing review

Adapter wins: 0; base wins: 3; ties: 0; adapter non-tied win rate: 0.0%.

## Full-benchmark success gate

Decision: **NOT EVALUATED**

The bounded pilot is reported descriptively; run the full frozen benchmark before applying the pre-registered production decision gate.
