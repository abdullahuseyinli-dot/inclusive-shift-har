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

`configs/schema/release_evidence_inventory.schema.json` defines a draft
machine-readable inventory. Generate the actual inventory only after the
candidate commit and files are stable. For each evidence item record its
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

Prepare a reviewed strict-JSON generation spec. It has exactly these top-level
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
  --allowed-output-root results `
  --destination release/final_release_evidence_inventory.json

uv run python -m inclusive_shift_har.artifacts.release_inventory validate `
  --inventory results/release/final_release_evidence_inventory.json `
  --spec $spec `
  --repository-root . `
  --candidate-commit $candidate
```

Both commands return nonzero on validation failure. Generation is create-only;
do not edit or overwrite the output. Validation recomputes the complete
inventory from the pinned candidate Git blobs and the supplied attestation
bundle, so changing and merely re-self-hashing the inventory is insufficient.
Neither operation loads sensor arrays or raw target recordings.

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
