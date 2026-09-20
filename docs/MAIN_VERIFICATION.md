# Main branch verification

The current `main` branch integrates the reviewed source-development and external
diagnostic work. Its scientific entry points are the [research report](RESEARCH_REPORT.md)
and [evidence index](EVIDENCE_INDEX.md). A branch update does not create a software
release, change repository visibility, or establish new scientific confirmation.

## Immutable tags and current commits

The [repository security policy](../configs/release/release_gate_policy_v1.json)
pins all eleven existing remote annotated tags by object ID, target commit,
message, and tagger. In particular, `benchmark-v0.1.7` remains at
`2c3d2262d9698b60f06a1291173389392feab50b`; it is no longer a candidate tag expected
to follow each new `main` commit. Unknown, altered, missing, or lightweight tags
remain failures. The current main ref must identify the candidate being checked.

The [historical release runbook](RELEASE_EVIDENCE_GATE.md) describes the earlier
tagged candidate. Reproduce that procedure from its historical revision. Do not
reuse its tag-creation commands for a current branch update. A future versioned
release requires a separately reviewed new tag and corresponding policy update.

## Required checks

Every `main` push runs the complete synthetic collection, lint, formatting,
types, manifest checks, split audit, artifact validation, configuration parsing,
dependency-lock verification, and Git integrity checks on Linux and Windows.
After those pass, the complete-history main security job requires:

- the candidate tree and all reachable Git history to pass content and ref checks;
- the pinned Gitleaks scanner to pass the complete commit-history scan;
- the installed dependency inventory to pass the reviewed licence policy;
- repeated quality gates and a candidate-bound CI evidence bundle.

Failed security steps retain evidence and cause the final bundle check to fail.
The historical tracked precommit report is checked as a preserved pending record;
it is not used as a passing attestation for the current commit. Current evidence
is uploaded by the workflow under its exact commit identifier. Final verification
also requires the completed GitHub Actions run to succeed.

Local verification of a proposed push uses a clean, complete clone of the remote
refs and the candidate main commit. Additional local-only research tags remain
preserved in the research worktrees and are not automatically published or
silently added to the distribution policy. No remote branch or tag is omitted
from the clone's history scan.
