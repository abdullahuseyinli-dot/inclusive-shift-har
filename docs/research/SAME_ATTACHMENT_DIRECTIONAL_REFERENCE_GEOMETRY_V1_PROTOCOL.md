# Same-attachment directional-reference geometry v1

Status: zero-fit software qualification. This protocol implements only the algebra and
deterministic fixtures authorized by the post-diagnostic roadmap. It does not alter the frozen
physical information pilot, use human outcomes, train an estimator, or authorize an architecture
performance experiment.

Normative configuration:
`configs/experiments/same_attachment_directional_reference_geometry_v1.yaml`.

Governing authority:
`NEXT_EXPERIMENT_SPEC.md`, section "After this diagnostic", and the retained physical information
identifiability protocol. The existing physical signal analyzer is complete and is not rebuilt here.

## Information contract

The reference contains exactly four separately qualified support-bout summaries from one unchanged
attachment: two sitting and two standing. Every bout contributes one unit gravity resultant, one
nonnegative directional concentration, and one measured angular dispersion. The two mixture
components within each class have exactly equal weight. Concentration is an explicit input; its
future mapping from support spread must be trained and frozen on outer-training participants. This
module does not estimate or silently clip it.

Every query supplies a gravity direction, two compact dynamics values, an explicit attachment ID,
an explicit validity flag, and a matched baseline probability vector stored as a native float64
NumPy array. Requiring that dtype is the precondition for bit-preserving fallback; other baseline
dtypes fail validation instead of being silently converted. The API accepts no activity
label, motion-stratum label, participant identity feature, or inferred remounting threshold.
Attachment IDs are equality metadata only. Missing, invalid, stale, or attachment-mismatched support
returns the baseline row byte-for-byte.

## Directional algebra

For a unit query `u`, support component `s_j`, and concentration `kappa_j`, the normalized
three-dimensional von Mises-Fisher log density is

`log f(u|s_j,kappa_j) = log C3(kappa_j) + kappa_j * dot(u,s_j)`,

where `C3(kappa) = kappa / (4*pi*sinh(kappa))`. The `kappa=0` branch is exactly the uniform sphere
density `1/(4*pi)`. Small and large concentrations use stable log-space formulas. Each class
likelihood is the equal-weight log-sum-exp of its two bout components.

The compact evidence is:

- `r = log L_sit - log L_stand`, signed posture preference;
- `e = log((L_sit + L_stand)/2) - log(1/(4*pi))`, absolute reference compatibility;
- `h`, the arithmetic mean of all four measured bout dispersions, a class-symmetric reliability
  feature.

The full joint head is exercised with fixed fixture coefficients only:

`z_motion = a0 + a'x + b_e*e + b_h*h + e*d'x`

`z_posture = r*(c_r + c_h*h)`

`p = [sigmoid(z_motion), (1-sigmoid(z_motion))*sigmoid(z_posture), remainder]`

The posture branch has no free intercept, so exchanging sitting and standing negates its logit with
unchanged coefficients. A later protocol may define a transformed intercept/prior under relabeling,
but it may not call an unchanged nonzero intercept label-swap equivariant.

## Symmetry and degeneracy rules

A shared proper rotation applied to query and support directions must leave `r`, `e`, `h`, and final
active probabilities unchanged. Swapping both posture support classes swaps their log likelihoods,
negates `r`, preserves `e` and `h`, and swaps sitting/standing probabilities while leaving mobility
unchanged.

Identical full sitting and standing mixtures (directions, concentrations, and weights) are valid but
nondiscriminative (`r=0`). Equal directions with different concentrations need not have `r=0`.
Antipodal class anchors remain valid and can be highly discriminative even though they cannot define a well-conditioned full
three-dimensional frame. The physical pilot's inverse-sine full-frame gate must not reject them from
this likelihood. A support bout whose mean resultant norm is at or below epsilon is invalid; this is
different from a valid component with `kappa=0`, which represents broad uncertainty.

The uniform four-component limit has `r=e=0`. Equal class odds alone do not imply neutral
compatibility: two equally improbable concentrated classes have `r=0` and negative `e`. This
distinction is required before the reference is allowed to interact with the motion path.

## Qualification gate

All configured deterministic fixtures must pass in one zero-fit execution. The report must retain
the inputs or their complete deterministic construction, output values, maximum symmetry errors,
fallback byte equality, probability bounds/sums, source/config/protocol hashes, repository state,
runtime, and zero-fit/zero-worker receipts. Independent validation recomputes the fixtures and checks
the artifact inventory. Any failed fixture leaves a visible failed report and blocks software
qualification.

Passing means only that the algebra is implemented consistently. Physical identifiability,
reattachment robustness, population performance, model gain, and novelty remain false. No
supervised fit may follow automatically.

## External gate retained

The next physical evidence remains the frozen six-wearer, two-visit, two-reattachment-per-visit
pilot: 24 blocks and 264 separate bouts. Its real-recording gate requires every complete wearer to
have strictly positive fresh wearer-mean signed posture margin separately for quiet and
upper-body-motion queries. Device qualification, lawful human-study authorization, consent,
canonical recordings, and adjudicated annotations are still absent. Synthetic fixtures cannot
substitute for them.
