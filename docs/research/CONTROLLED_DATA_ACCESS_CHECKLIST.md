# Controlled clinical data access checklist

## WearGait-PD — sealed confirmation

Current gate: **blocked; no authorized local data or access receipt is present**.

The dataset is hosted as Synapse project `syn52540892` and requires registration and
acceptance of its data-use conditions by an authorized human account holder. The repository
must not automate acceptance, borrow credentials, or replace the cohort with synthetic
records.

Before the single confirmatory opening:

1. The account holder obtains access and records the approved Synapse project/version,
   approval date, applicable terms, and permitted retention/publication conditions.
2. A source receipt records every downloaded artifact's provider ID, size, and digest.
3. The L4/L5 acceleration, free-acceleration, gyroscope, timestamp, trial, session, site,
   participant, and framewise label fields pass a schema and unit/frame audit.
4. The method code, derived-gravity rule (`gravity = acceleration - free acceleration` only
   after the frame/unit audit), model checkpoints, route thresholds, primary endpoint, and
   statistical analysis are frozen and hashed before any target label is opened.
5. Exactly one confirmatory scoring run is made. Failures and manual interventions remain in
   its run record. No tuning or rerun is disguised as confirmation.

## Parkinson at Home — longitudinal extension

Current gate: **blocked; steward approval has not been supplied**.

The authorized researcher should contact the data steward through the process described by
the study publication. The request should identify the institution and investigator,
research purpose, requested lower-back/phone/watch modalities and annotations, security and
retention plan, intended derived outputs, and publication/re-identification safeguards.
Approval, terms, and artifact receipts must be stored before any adapter is allowed to read
the data. This extension is not a substitute for the frozen WearGait-PD confirmation.

## Fail-closed rule

Absence of either access package is a visible `BLOCKED_NO_AUTHORIZED_ACCESS` outcome, not a
failed model result. No code path may emit a performance metric for an inaccessible cohort.
