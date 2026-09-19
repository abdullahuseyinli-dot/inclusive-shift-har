# HERA posture semantic-gauge v1 results

Run completed 2026-09-16 as reused source-development evidence. The runner applied a
fixed labelled semantic gauge to frozen HERA-v1 strict probabilities. It did not retrain
HERA, load target records, or fit a temporal model.

## Result

The full frozen HERA reference is 0.8675497019 mean participant macro-F1 (86.755%).
The active one-query hard gauge used exactly one selected label per source participant,
swapped the sitting/standing columns for participant 10, and retained identity for the
other nine participants. Because queried windows are excluded by protocol, the matched
comparison is on the same 715 remaining windows:

| Path | Queries | Mean participant macro-F1 | Delta vs same remaining windows | 95% participant bootstrap | Wins / harms / ties |
|---|---:|---:|---:|---:|---:|
| Full frozen HERA-v1 strict | 0 | 0.867550 | — | — | — |
| Active hard, budget 1 | 10 | 0.872558 | +0.611 pp | [0.000, +1.833] pp | 1 / 0 / 9 |
| Active soft, budget 1 | 10 | 0.871067 | +0.462 pp | [−0.447, +1.833] pp | 1 / 1 / 8 |
| Random hard, budget 1 | 10 | 0.873285 | +0.565 pp | [0.000, +1.696] pp | 1 / 0 / 9 |
| Random soft, budget 1 | 10 | 0.830456 | −3.718 pp | [−12.557, +1.550] pp | 1 / 2 / 7 |
| Fixed SAR hard (1+1 labels) | 20 | 0.821417 | −5.041 pp | [−15.123, 0.000] pp | 0 / 1 / 9 |

Budgets 2 and 3 produced the same active decisions and result as budget 1: the fixed
threshold was reached after one query for every participant. Random budgets used 10,
14 and 16 queries respectively; all budgets are retained in `result.json` without
best-budget selection.

The active hard path reduced the matched remaining posture exchanges from 73 to 69.
Sitting recall increased from 77.974% to 81.498%, standing recall changed from 84.348%
to 82.609%, and mobility recall stayed at 97.287%. Participant 10 improved by 6.110 pp
macro-F1; there were no hard-gauge participant harms. The bootstrap lower bound is zero,
so this is a promising diagnostic rather than a promoted HERA successor. The soft path
was less stable, and its random control caused a large participant-7 harm.

## Interpretation and boundary

The run supports the semantic-gauge hypothesis: one label can identify the P10-like
posture-column ambiguity. It does not show that the physical sitting/standing signal is
more separable, and it is not a zero-shot gain because labels are queried and removed
from evaluation. The 10-person source cohort is already used for development; an
independent ability-relevant cohort is required for a generalization claim.

The proposed temporal part was explicitly blocked. The released artifact has stable
window IDs but no authoritative contiguous session/trial/timestamp order, so constructing
sequences from file order would be invalid. No temporal fit was launched.
The blocker is recorded at `.audit/hera_posture_gauge/TEMPORAL_STAGE_BLOCKER_20260916.json`.

## Evidence and validation

Implementation: `src/inclusive_shift_har/experiments/hera_posture_gauge.py`.
Protocol/config: `docs/research/HERA_POSTURE_GAUGE_V1_PROTOCOL.md` and
`configs/experiments/hera_posture_gauge_v1.yaml`.

Complete run artifacts:

- `.audit/hera_posture_gauge/hera-posture-gauge-20260916-005/result.json`
- `.audit/hera_posture_gauge/hera-posture-gauge-20260916-005/predictions.npz`
- `.audit/hera_posture_gauge/hera-posture-gauge-20260916-005/manifest.json`
- `.audit/hera_posture_gauge/HERA_POSTURE_GAUGE_VALIDATION_20260916.json` (44/44 checks passed)

Earlier failed attempts remain visible and were not used: `...-001/FAILURE.json` records
the SAR indexing failure; `...-002/FAILURE.json` records the binary-mode serialization
failure and its partial prediction artifact. Attempts `...-003` and `...-004` are retained
as superseded byte-identical replays after implementation cleanup; they are not independent
seeds. `ATTEMPT_LEDGER.json` records all five attempts. The final run took 0.303 seconds
after validation and produced no target access.
