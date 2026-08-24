# Final release evidence gate

The checked-in Stage 9 report proves the gate that preceded the one-time target
opening. It does not cover later code, documentation, post-confirmatory work, or
the current dirty worktree. A new final release gate must be run from the exact
candidate commit. It must not reopen the target.

## Release-blocking checks

From a CUDA environment at the candidate commit, record stdout/stderr, tool
versions, exit codes, UTC time, machine record, and output hashes for:

```powershell
uv sync --locked --extra training-cuda --group research
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
uv run inclusive-shift-har validate-manifests --json
uv run inclusive-shift-har audit-splits `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --json
uv run inclusive-shift-har validate-artifacts --artifact-root results --require-artifacts --json
git status --short
git ls-files
```

The release record must additionally include a versioned secret-scanner result,
a tracked-file/large-file scan proving that raw datasets, the coursework ZIP,
secrets, caches, and checkpoints are absent from normal Git history, a
third-party license/notice audit, and a successful CI run on the candidate
commit. A hand-written grep is not sufficient evidence for the secret gate.

Every primary result file linked by the locked index must hash correctly, every
checkpoint/calibrator link in the final freeze must validate, and all preserved
failure/deviation records referenced by public claims must remain reachable.
Post-confirmatory operations may be either complete and validated or explicitly
listed as not run; incompleteness must not be hidden by omitting a role.

## Evidence inventory

`configs/schema/release_evidence_inventory.schema.json` defines the version
1.1 external release-asset inventory. Generate the actual inventory only after
the candidate commit and files are stable. The inventory must use
`delivery_mode: external_release_asset` and `tracked: false`; place it at
`.audit/release-assets/<candidate>/final_release_evidence_inventory.json` and
attach that exact file to the eventual GitHub release. This avoids an
impossible commit self-reference. Record the inventory SHA-256 in the annotated
release tag message. For each evidence item record its
repository-relative path, exact byte size, file SHA-256, embedded record SHA-256
where present, evidence status, validation status, claim scope, and whether it
is release-required.

The required-role list deliberately covers legacy audit/verification,
literature, dataset provenance/cards/audit, ontology and protocol locks, splits,
source freeze, consumed opening receipt, locked target index, participant
statistics, publication report, cards, baseline omissions, runbook, paper
outline, and the new release report. The inventory must preserve
`opening_count = 1` and `target_rerun_permitted = false`. It is not valid merely
because files exist; hashes, link targets, status semantics, and combinatorial
coverage must validate.

`status: ready` requires every required role, no open blocker, every release gate
at `pass`, a clean worktree, and a commit matching all recorded outputs. The
schema is the interchange contract;
`inclusive_shift_har.artifacts.release_inventory` is the create-only generator
and deterministic validator that enforces role/path uniqueness, required-role
coverage, declared embedded self-hashes, and the one-opening/no-rerun state.

## Inventory generation runbook

Prepare a reviewed strict-JSON generation spec. The spec remains version 1.0
while its generated inventory contract is version 1.1. It has exactly these top-level
keys: `schema_version`, `spec_kind`, `requested_status`, `created_at_utc`,
`protocol_tag`, `remote`, `confirmatory_state`, `artifacts`, `gates`,
`postconfirmatory_statuses`, and `blockers`. Use schema version `1.0.0` and spec
kind `release_evidence_inventory_spec`. The synthetic fixture in
`tests/test_release_inventory.py` is the executable field-level example.

Artifact entries point to regular files in the candidate Git commit and pin
their byte SHA-256. They cannot point at raw data, caches, archives, model
checkpoints, or untracked working-tree files. Each role and path is unique. The
opening receipt and locked target index must declare
`embedded_self_hash_field: record_sha256`. All ten post-confirmatory tracks must
be present explicitly, including tracks that are `incomplete`, `not_run`,
`blocked`, or `not_applicable`; every one remains
`primary_claim_eligible: false`.

Every `pass` or `fail` gate cites pinned evidence from either the candidate
commit (`evidence_source: repository_commit`) or the directory containing the
spec (`evidence_source: spec_root`). `not_run` and `not_applicable` gates cite no
evidence. A CI `pass` additionally requires a self-hashed JSON record with
`record_kind: ci_run_evidence`, `status: pass`, `conclusion: success`, and the
exact candidate commit. A non-absent GitHub remote requires a pinned,
self-hashed `github_remote_evidence` JSON record with the exact commit,
visibility, and canonical HTTPS URL. This external attestation bundle avoids a
commit self-reference while recording only scoped relative paths in the
inventory; no absolute workstation path is serialized.

Run from the repository root only after the candidate commit is clean. The
destination is relative to `--allowed-output-root`; it must not already exist.

```powershell
$candidate = (git rev-parse HEAD).Trim()
$dirty = git status --porcelain=v1 --untracked-files=all
if ($dirty) { throw "Release candidate worktree is not clean" }
$spec = "C:\path\to\reviewed-release-attestations\release-spec.json"

uv run python -m inclusive_shift_har.artifacts.release_inventory generate `
  --spec $spec `
  --repository-root . `
  --candidate-commit $candidate `
  --require-clean-worktree `
  --allowed-output-root .audit `
  --destination release-assets/$candidate/final_release_evidence_inventory.json

uv run python -m inclusive_shift_har.artifacts.release_inventory validate `
  --inventory .audit/release-assets/$candidate/final_release_evidence_inventory.json `
  --spec $spec `
  --repository-root . `
  --candidate-commit $candidate
```

Both commands return nonzero on validation failure. Generation is create-only;
do not edit or overwrite the output. Validation recomputes the complete
inventory from the pinned candidate Git blobs and the supplied attestation
bundle, so changing and merely re-self-hashing the inventory is insufficient.
Neither operation loads sensor arrays or raw target recordings.

## Final security/provenance report

The tracked `results/release/final_release_gate_report.json` is deliberately a
`tracked_precommit_report`: it binds its parent commit, leaves
`content_commit` null, records every gate as `not_run`, and says that the exact
candidate attestation is pending. It must never be edited to claim that the
commit containing itself has passed. After that report is committed, create
the exact, ignored attestation under
`.audit/release-attestations/<candidate>/final_release_gate_report.json`.

The release-security commands are create-only:

```powershell
$candidate = (git rev-parse HEAD).Trim()
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$evidence = ".audit/release-attestations/$candidate"

uv run python -m inclusive_shift_har.artifacts.release_gate scan-repository `
  --repository-root . --candidate-commit $candidate `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output "$evidence/repository_scan.json"

uv run python -m inclusive_shift_har.artifacts.release_gate validate-secret-scan `
  --repository-root . --report "$evidence/gitleaks.json" `
  --candidate-commit $candidate --gitleaks-version 8.30.1 `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output "$evidence/secret_scan.json"

uv run python -m inclusive_shift_har.artifacts.release_gate audit-licenses `
  --repository-root . --inventory "$evidence/python_licenses.json" `
  --candidate-commit $candidate --pip-licenses-version 5.5.5 `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output "$evidence/license_audit.json"

uv run python -m inclusive_shift_har.artifacts.release_gate assemble `
  --repository-root . --policy configs/release/release_gate_policy_v1.json `
  --mode exact_candidate_attestation --created-at-utc $now `
  --spec "$evidence/attestation_spec.json" `
  --output "$evidence/final_release_gate_report.json"

uv run python -m inclusive_shift_har.artifacts.release_gate validate `
  --report "$evidence/final_release_gate_report.json" --repository-root .
```

The repository scanner reads candidate and historical Git objects, rejects
protected/raw/archive/checkpoint paths and suffixes (including ONNX), blobs over
16 MiB, symlinks, submodules, case collisions, and disguised binary payloads.
It does not walk or open raw worktree paths. Gitleaks is pinned to 8.30.1; the
official Linux x64 archive SHA-256 is
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`.
The release dependency group locks `pip-licenses==5.5.5`.

CI assembles and validates a candidate-scoped attestation only after the
synthetic matrix and its substantive security/quality steps succeed. Its
sanitized ignored JSON evidence is uploaded even when a preceding gate fails.
The external inventory is generated only after successful remote CI evidence
can be pinned; generating it inside the still-running CI job would overstate
that circular state.

## Private GitHub sequence

No remote operation belongs in the evidence gate until the local candidate is
ready. Then:

1. Run `gh auth status` and obtain the current login; do not assume an owner.
2. Check `OWNER/inclusive-shift-har` without creating or modifying it.
3. If it exists, inspect and stop—never overwrite or force-push.
4. If absent, create a private repository from the validated local candidate and
   push only lawful, non-sensitive tracked files.
5. Require CI to pass on the pushed commit before a release-status inventory.
6. Add a benchmark tag only after its target commit and inventory validate.
7. Keep the repository private until result lineage, licenses, documentation,
   claims, and tracked-file scans pass. Do not mint a DOI for a private or
   unstable release.

Existing `legacy-audit-v0.1.0`, `protocol-v1.0.0`, and `protocol-v1.2.0` tags are
historical anchors. They are not substitutes for a future benchmark release tag.
