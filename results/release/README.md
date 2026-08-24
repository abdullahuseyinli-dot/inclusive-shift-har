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

See `docs/RELEASE_EVIDENCE_GATE.md` for the create-only commands and circularity
rules.
