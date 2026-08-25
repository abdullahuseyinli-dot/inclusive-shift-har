# Release evidence

`final_release_gate_report.json` is a tracked, create-only precommit record. It
is intentionally pending: the commit containing that record and its successful
remote CI run cannot be attested from inside the record itself.

Exact candidate evidence belongs outside Git under
`.audit/release-attestations/<candidate_commit>/`. The final evidence inventory
belongs under `.audit/release-assets/<candidate_commit>/` and is attached as a
release asset, not committed. Neither external directory may contain raw data,
checkpoints, secrets, Gitleaks raw findings, or downloaded scanner binaries in
the uploaded evidence bundle.

Sanitized release failures are tracked under `failures/`. The
[`benchmark-v0.1.5` bundle-scan record](failures/benchmark-v0.1.5-bundle-gitleaks.json)
binds the candidate, tag, successful CI run, bundle, and raw-report identities
without recording a secret value or an absolute quarantine path. The bundle,
raw report, and supporting evidence remain quarantined outside Git. No release
was created and no assets were uploaded.

The v0.1.6 successor prevents recurrence by filtering the authenticated
repository response before it is written, rejecting any retained prohibited
field during both capture and offline validation, and limiting the Gitleaks
allowance to the two validated SHA-256 evidence fields in `ci.json`.

The
[`benchmark-v0.1.6` draft-validation record](failures/benchmark-v0.1.6-draft-utf8-validation.json)
preserves the later Windows PowerShell 5 UTF-8 decoding false negative. All
candidate, CI, inventory, bundle, and security gates passed and the two draft
assets were uploaded, but the draft was not published. The v0.1.7 successor
uses strict UTF-8 decoding for saved GitHub JSON; the earlier draft and external
evidence remain unchanged.

See `docs/RELEASE_EVIDENCE_GATE.md` for the create-only commands and circularity
rules.
