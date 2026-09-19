# HERA posture semantic-gauge protocol v1

Status: locked before the source-development run on 2026-09-16.

This protocol applies the previously qualified labelled semantic-gauge idea to the
frozen HERA-v1 strict source predictions. It is a diagnostic of participant-specific
sitting/standing semantics. It does not retrain HERA, change its mobility head, use
target participants, or claim zero-shot improvement.

## Question

HERA's source error is concentrated in sitting-versus-standing exchanges. The run
tests whether a very small number of labels can resolve a participant-specific
identity-versus-sitting/standing-swap gauge while preserving HERA's mobility mass.
The temporal decoder from the architecture proposal is not fitted because the released
source prediction artifact has no authoritative contiguous session, trial, or timestamp
ordering. Arbitrary window-file order is not a valid temporal sequence.

## Frozen input and splits

The input is the previously sealed source-development artifact
`.audit/hera_compact_evidence_integration/hera-compact-evidence-integration-20260916-001/predictions.npz`.
The `hera_v1_strict_probabilities`, labels, participant IDs and window IDs must match
exactly. Participants 1--10 are the exhausted source-development cohort. Labels are
used only to simulate a query response after a window has been selected from prediction
values and stable window IDs; they are never used to rank or select a query.

## Active query and hard gauge

For each participant, stationary mass is `p_sit+p_stand`. Candidate windows have mass
at least 0.5, with lower-mass windows as a deterministic fallback. The active utility is
stationary mass times the absolute sitting/standing log odds. Ties use the lexical window
ID. A SHA-256 hash order of `seed|window_id` under the same filter is the random control.

At most 1, 2, or 3 queries per participant are allowed. After each selected window the
label is read, and the swap-over-identity log Bayes increment is
`log(p_other)-log(p_observed)` for a posture label. Querying stops when the absolute
accumulated factor reaches `log(3)`. A positive factor at least `log(3)` swaps sitting
and standing columns; otherwise identity is retained. Every queried window is excluded
from both the adapted score and its same-remaining-window no-adaptation control.

## Soft gauge candidate

The active and random query masks are shared with their hard counterparts. The soft
candidate uses a fixed prior swap probability of 0.5, evidence temperature 1.0 and
maximum swap weight 1.0. After the selected labels, the posterior swap probability is
`sigmoid(log_bayes_factor)`. The adjusted distribution is the posterior mixture of the
identity and swapped posture columns; mobility is copied unchanged and stationary mass
is renormalized. These values were fixed before reading this run's outcomes.

## SAR control

The fixed SAR control selects one hash-ranked label from each of the sitting and standing
classes for every participant. It uses two labels per participant, excludes both windows
from evaluation, and applies the same identity-versus-swap likelihood rule. It is a
label-cost comparator, not a zero-shot method.

## Metrics and evidence boundary

Report full HERA, matched same-remaining HERA, active hard and soft, random hard and
soft, and SAR reports for every budget. Each report contains fixed-three-class macro-F1,
per-class recall, confusion matrix, proper scores, every participant score, query count,
decision count, and a 10,000-resample participant bootstrap for adapted-minus-matched
baseline differences (seed 1729). Report all budgets without selecting a best budget.

The run is `reused_source_development_not_independent`, `labelled_personalization_only`,
and `human_performance_claim: false`. A positive result supports only sparse labelled
participant personalization. An independent ability-relevant cohort is required for any
generalization or deployment claim.
