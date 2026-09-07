# Physical information identifiability pilot v1 runbook

This runbook operationalizes the frozen protocol. It does not authorize a human study.

## Before scheduling participants

1. Obtain the applicable ethics, consent, privacy, accessibility, and data-retention
   approvals. Record their identifiers without storing direct participant identity in the
   repository.
2. Qualify one actual device interface. Record device/model/firmware, logger version,
   placement and attachment instructions, axis convention, channels, units, clock domain,
   native-gravity status, measured sampling intervals, dropouts, and bench uncertainty.
   Hash both the placement instruction and bench-qualification files.
3. Freeze the device-specific sampling, discontinuity, gravity, and resampling settings
   before any human outcome, including both window length and stride. Set the
   frame-conditioning limit from bench uncertainty or leave frame eligibility unresolved.
4. Generate and validate the plan from a clean committed worktree:

   ~~~powershell
   $pilotArgs = @(
     "prepare",
     "--repository-root", "<worktree>",
     "--evidence-root", "<evidence-root>",
     "--config", "<worktree>/configs/experiments/physical_information_identifiability_v1.yaml",
     "--protocol", "<worktree>/docs/research/PHYSICAL_INFORMATION_IDENTIFIABILITY_V1_PROTOCOL.md",
     "--schema", "<worktree>/configs/schema/physical_information_pilot.schema.json",
     "--output", "<evidence-root>/.audit/physical_information_pilot/<run-id>",
     "--code-commit", "<HEAD>"
   )
   python -m inclusive_shift_har.experiments.physical_information_pilot @pilotArgs
   ~~~

## At each visit

Use only the assigned pseudonymous wearer slot. Record actual UTC visit order. For each
attachment block, fully remove and reattach the device using the frozen instruction;
photographic or placement evidence follows the approved privacy protocol.

Start one continuous raw recording before the first bout and retain transitions. Run the
four support prompts in the generated order, then the seven query prompts. Sitting and
standing are posture instructions, never definitions based on low motion. Upper-body
motion should be comfortable and adapted to the wearer without changing the intended
condition. Record assistance, deviation, interruption, failure, or withdrawal as it occurs.

Do not repeat or substitute a bout based on visible signal quality or a model response.
Safety and participant choice take priority; mark incomplete slots explicitly.

## Import and adjudication

For every raw object, record a safe relative path, SHA-256, byte size, sample count,
recording/device/wearer/visit/attachment identifiers, units, channel map, clock, and
observed start/end. `started_at` and `ended_at` are seconds on one common device clock
relative to `visit_started_at_utc`; preserve raw clocks and document any reset mapping.
Stable sample IDs are '<recording-id>:<zero-padded-index>'.

Annotate actual bout start/end on the continuous recording. Keep intended prompt separate
from adjudicated activity/motion, adjudication status, adjudicator, and timestamp. Check
that support and query sample ranges do not overlap and that every reference precedes its
query. Retain invalid and missing samples; do not silently shorten the denominator.

## Analysis release gate

Run signal processing over complete continuous attachment recordings, then project fixed
five-second-trim interiors. Use identical query IDs and validity masks in fresh/stale
comparisons. Produce per-bout, per-condition, and per-wearer denominators before aggregate
statistics.

Architecture work remains closed unless the protocol's physical-information rule is met.
A positive feasibility decision permits only drafting a separate frozen architecture
protocol.

The collection checker requires the same frozen schema used by the plan package:

~~~powershell
$checkArgs = @(
  "check-collection",
  "--plan", "<plan.json>",
  "--config", "<config.yaml>",
  "--schema", "<schema.json>",
  "--equipment", "<equipment.json>",
  "--recordings", "<recordings.json>",
  "--annotations", "<annotations.json>",
  "--raw-root", "<evidence-root>",
  "--output", "<new-validation.json>"
)
python -m inclusive_shift_har.experiments.physical_information_pilot @checkArgs
~~~

A nonzero exit means the create-only report is invalid. Metadata-complete records remain
signal-ineligible until the separately frozen continuous-record processing and fixed-trim
checks run.
