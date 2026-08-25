# Final release evidence gate

This runbook releases one immutable candidate without reopening the consumed
confirmatory target. The tracked
`results/release/final_release_gate_report.json` is intentionally a truthful
`tracked_precommit_report`: it remains `pending`, names no content commit, and
does not pretend that the commit containing it has already passed. Completed
candidate evidence lives outside normal Git history.

The version 1.1 release inventory has only three states: `draft`, `blocked`, and
`ready`. It has no `released` state and makes no post-publication claim.

## Non-negotiable invariants

- Never invoke the confirmatory evaluator or materialize target predictions
  again. Opening count remains one and `target_rerun_permitted` remains false.
- Run neural/environment checks from the locked CUDA environment. Native
  Random Forest, SVM, and logistic-regression implementations remain CPU
  methods by design.
- Never commit or upload raw sensor data, the coursework ZIP, checkpoints,
  secrets, caches, raw Gitleaks reports, or unrestricted logs.
- Do not amend the candidate, force-push, move a tag, overwrite create-only
  evidence, or edit release metadata in the candidate after it is pushed.
- The exact release candidate, remote `main`, CI `head_sha`, benchmark-tag
  target, and every external record must identify the same 40-character commit.

## 1. Local CUDA and repository gates

Run these before staging. Preserve command, UTC start/completion, exit code,
tool version, machine/CUDA record, complete sanitized log, byte size, and
SHA-256 under a new ignored `.audit/` directory.

```powershell
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Assert-NativeSuccess([string]$Label) {
  if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
uv lock --check
Assert-NativeSuccess 'uv lock check'
uv sync --locked --extra training-cuda --group research --group release
Assert-NativeSuccess 'locked CUDA environment sync'

uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
Assert-NativeSuccess 'CUDA allocation gate'
uv run pytest -q
Assert-NativeSuccess 'pytest'
uv run ruff check src tests
Assert-NativeSuccess 'ruff lint'
uv run ruff format --check src tests
Assert-NativeSuccess 'ruff format check'
uv run mypy src tests
Assert-NativeSuccess 'mypy'
uv run inclusive-shift-har validate-manifests --json
Assert-NativeSuccess 'manifest validation'
uv run inclusive-shift-har audit-splits `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --json
Assert-NativeSuccess 'split audit'
uv run pytest -q tests/test_few_person_protocol.py tests/test_protocol_splits.py `
  tests/test_schema_documents.py tests/test_uci_source_protocol.py
Assert-NativeSuccess 'protocol regression tests'
uv run inclusive-shift-har validate-artifacts `
  --artifact-root results --require-artifacts --json
Assert-NativeSuccess 'artifact validation'
git fsck --full
Assert-NativeSuccess 'git fsck'
git diff --check
Assert-NativeSuccess 'worktree whitespace check'
```

Also parse `CITATION.cff`, `.zenodo.json`, every JSON schema, and every tracked
self-hashed result. These are read-only validation operations; they do not
authorize another model evaluation.

## 2. Review the exact staged candidate

Stage only reviewed paths. Require no unstaged tracked differences and no
untracked, non-ignored files. The index scanner reads Git index objects without
opening raw worktree paths or writing Git objects.

```powershell
$base = (git rev-parse HEAD).Trim()
Assert-NativeSuccess 'resolve candidate parent'
$review = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

# Regenerate the tracked pending report against the current parent and policy.
$trackedDraft = ".audit/release-precommit/$review/tracked_precommit_report.json"
uv run python -m inclusive_shift_har.artifacts.release_gate assemble `
  --repository-root . `
  --policy configs/release/release_gate_policy_v1.json `
  --mode tracked_precommit_report --created-at-utc $now `
  --output $trackedDraft
Assert-NativeSuccess 'tracked pending-report assembly'
Copy-Item -LiteralPath $trackedDraft `
  -Destination results/release/final_release_gate_report.json -Force

# Repository-bound validation is intentionally deferred until the direct child
# candidate exists; before that commit, the new policy/report blobs have no
# reconstructible Git lineage.

git add -- <explicit-reviewed-paths>
git diff --cached --check
Assert-NativeSuccess 'cached whitespace check'
git diff --cached --name-status
Assert-NativeSuccess 'cached path review'
if (git diff --name-only) { throw "Tracked worktree differs from reviewed index" }
if (git status --porcelain=v1 --untracked-files=all | Select-String '^\?\?') {
  throw "Unreviewed untracked file present"
}

uv run python -m inclusive_shift_har.artifacts.release_gate scan-index `
  --repository-root . `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now `
  --output ".audit/release-precommit/$review/staged_index_scan.json"
if ($LASTEXITCODE -ne 0) { throw "Exact staged-index scan failed" }
```

Run the checksum-verified Gitleaks 8.30.1 executable against the staged index as
a second, independent gate. Preserve its actual exit code and report outside
Git. The staged-index scanner is not a secret scanner.

Provision the Windows x64 binary only in a fresh ignored tool directory and
verify both the official archive and extracted executable before use:

```powershell
$toolRoot = ".audit/tools/gitleaks-8.30.1-windows-x64"
if (Test-Path -LiteralPath $toolRoot) { throw "$toolRoot already exists; inspect and stop" }
New-Item -ItemType Directory -Path $toolRoot -ErrorAction Stop | Out-Null
gh release download v8.30.1 --repo gitleaks/gitleaks `
  --pattern gitleaks_8.30.1_windows_x64.zip --dir $toolRoot
Assert-NativeSuccess 'Gitleaks archive download'
$archive = Join-Path $toolRoot 'gitleaks_8.30.1_windows_x64.zip'
if ((Get-Item -LiteralPath $archive).Length -ne 8438883 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -ne
      'd29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e') {
  throw 'Gitleaks Windows archive pin differs'
}
$expanded = Join-Path $toolRoot 'expanded'
New-Item -ItemType Directory -Path $expanded -ErrorAction Stop | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $expanded -ErrorAction Stop
$gitleaks = Join-Path $expanded 'gitleaks.exe'
if ((Get-Item -LiteralPath $gitleaks).Length -ne 22575104 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $gitleaks).Hash.ToLowerInvariant() -ne
      '17157e2ee8b76fc8b1d8bee607a250e34b8a8023c8bc81822d4b5ee4d78fcb7c') {
  throw 'Gitleaks Windows executable pin differs'
}
if ((& $gitleaks version).Trim() -ne '8.30.1') { throw 'Gitleaks version differs' }
```

Immediately recheck the cached diff, then commit once:

```powershell
$stagedSecretReport = ".audit/release-precommit/$review/staged-gitleaks.json"
& $gitleaks git --staged --no-banner --config .gitleaks.toml --exit-code 1 `
  --report-format json --report-path $stagedSecretReport
$stagedGitleaksExit = $LASTEXITCODE
if ($stagedGitleaksExit -ne 0) {
  throw "Staged Gitleaks gate failed with exit code $stagedGitleaksExit"
}

# Reconstruct the index record immediately before committing and compare it.
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$recheckPath = ".audit/release-precommit/$review/staged_index_scan_recheck.json"
uv run python -m inclusive_shift_har.artifacts.release_gate scan-index `
  --repository-root . `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output $recheckPath
Assert-NativeSuccess 'staged-index recheck'
$firstIndex = Get-Content -Raw ".audit/release-precommit/$review/staged_index_scan.json" |
  ConvertFrom-Json
$secondIndex = Get-Content -Raw $recheckPath | ConvertFrom-Json
foreach ($field in @('head_commit','index_entry_count','staged_change_count',
    'unique_blob_count','index_entry_index_sha256')) {
  if ($firstIndex.$field -ne $secondIndex.$field) {
    throw "Reviewed index changed before commit: $field"
  }
}
git diff --cached --check
Assert-NativeSuccess 'final cached whitespace check'
if (git diff --name-only) { throw "Tracked worktree changed after index review" }

git commit -m "Prepare benchmark release candidate"
Assert-NativeSuccess 'release-candidate commit'
$candidate = (git rev-parse HEAD).Trim()
if ((git rev-parse "$candidate^").Trim() -ne $base) { throw "Candidate parent changed" }
if (git status --porcelain=v1 --untracked-files=all) { throw "Candidate is not clean" }
if ((git branch --show-current).Trim() -ne 'main') { throw "Candidate is not on main" }
uv run python -m inclusive_shift_har.artifacts.release_gate validate `
  --report results/release/final_release_gate_report.json --repository-root .
Assert-NativeSuccess 'candidate-bound tracked pending-report validation'
```

Rerun all Section 1 gates against this clean candidate. Do not create another
content commit afterward.

## 3. Create and validate the benchmark tag before release CI

The release policy byte-pins all five historical annotated tags. This includes
the immutable `benchmark-v0.1.0` and `benchmark-v0.1.1` candidates whose
Actions runs `32799146947` and `32801378375` failed before tests because of,
respectively, conflicting frozen/locked options and a hosted-runner rejection
of an empty boolish override. It also retains the preserved `protocol-v1.0.0`
name/message mismatch. The policy permits exactly one successor candidate tag
with fixed metadata. Create it only after the pre-tag local gates pass:

```powershell
git show-ref --verify --quiet refs/tags/benchmark-v0.1.2
$tagProbe = $LASTEXITCODE
if ($tagProbe -eq 0) {
  throw "benchmark-v0.1.2 already exists; inspect it and stop"
}
if ($tagProbe -ne 1) { throw "Unable to determine benchmark tag state" }
if ((git config --get user.name).Trim() -ne 'Abdulla Huseyinli' -or
    (git config --get user.email).Trim() -ne 'abdullahuseyinli@gmail.com') {
  throw 'Git tagger identity differs from the release policy'
}
git tag -a benchmark-v0.1.2 $candidate `
  -m "InclusiveShift-HAR benchmark v0.1.2"
Assert-NativeSuccess 'benchmark tag creation'
if ((git cat-file -t benchmark-v0.1.2).Trim() -ne 'tag') { throw "Tag is not annotated" }
if ((git rev-parse 'benchmark-v0.1.2^{commit}').Trim() -ne $candidate) {
  throw "Benchmark tag targets another commit"
}
```

The policy validates the raw UTF-8 tag object: direct `type commit`, exact tag
name, exact one-line message, exact tagger name/email, canonical headers, and
candidate target. Gitleaks does not inspect annotated-tag payloads, so this
independent byte-level policy is mandatory. For a later release, move this tag
to the pinned historical set in a new policy and declare a new candidate tag;
never retarget `benchmark-v0.1.0`, `benchmark-v0.1.1`, or `benchmark-v0.1.2`.

Now create candidate-bound local repository, secret, and licence evidence in a
fresh path. The repository scanner requires all policy-declared tags to exist,
rejects non-commit refs, lightweight/unreviewed/nested tags, tag-only blobs,
replace refs, grafts, shallow history, raw/checkpoint/archive paths, symlinks,
submodules, unsupported binaries, and blobs over 16 MiB.

```powershell
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$local = ".audit/local-candidate/$candidate"
if (Test-Path -LiteralPath $local) { throw "$local already exists; inspect and stop" }

uv run python -m inclusive_shift_har.artifacts.release_gate scan-repository `
  --repository-root . --candidate-commit $candidate `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output "$local/repository_scan.json"
Assert-NativeSuccess 'complete repository scan'

# Run the pinned Gitleaks binary first; preserve its real exit code.
& $gitleaks git --no-banner --config .gitleaks.toml --exit-code 1 `
  --log-opts=--all --report-format json --report-path "$local/gitleaks.json"
$gitleaksExit = $LASTEXITCODE
uv run python -m inclusive_shift_har.artifacts.release_gate validate-secret-scan `
  --repository-root . --report "$local/gitleaks.json" `
  --candidate-commit $candidate `
  --policy configs/release/release_gate_policy_v1.json `
  --config .gitleaks.toml --ignore .gitleaksignore `
  --gitleaks-version 8.30.1 --gitleaks-exit-code $gitleaksExit `
  --created-at-utc $now --output "$local/secret_scan.json"
if ($gitleaksExit -ne 0 -or $LASTEXITCODE -ne 0) { throw "Secret gate failed" }

uv run pip-licenses --format=json --with-urls `
  --output-file="$local/python_licenses.json"
Assert-NativeSuccess 'Python licence inventory'
uv run python -m inclusive_shift_har.artifacts.release_gate audit-licenses `
  --repository-root . --inventory "$local/python_licenses.json" `
  --candidate-commit $candidate --pip-licenses-version 5.5.5 `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output "$local/license_audit.json"
Assert-NativeSuccess 'licence audit'
```

The official Gitleaks 8.30.1 Linux x64 archive SHA-256 is
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`.
The Windows archive and executable pins used above are also recorded in the
versioned release policy.

## 4. Create the private remote and push atomically

Check authentication and require a genuine 404 before creating anything. If
the repository exists, stop and inspect it; never overwrite it.

```powershell
gh auth status
Assert-NativeSuccess 'GitHub authentication'
$owner = (gh api user --jq .login).Trim()
Assert-NativeSuccess 'authenticated GitHub owner lookup'
$repository = "$owner/inclusive-shift-har"
$probe = gh api "repos/$repository" 2>&1
if ($LASTEXITCODE -eq 0) { throw "$repository already exists; stop for inspection" }
if ($probe -notmatch 'HTTP 404|Not Found') { throw "Absence was not proven by HTTP 404" }

gh repo create $repository --private --source . --remote origin
Assert-NativeSuccess 'private GitHub repository creation'
$remoteState = gh api "repos/$repository" | ConvertFrom-Json
Assert-NativeSuccess 'created repository lookup'
if (-not $remoteState.private -or $remoteState.full_name -ne $repository) {
  throw "Created remote is not the intended private repository"
}
```

Push `main` and every required tag in one atomic update so the main-push CI job
can fetch and validate the complete tag policy:

```powershell
git push --atomic -u origin `
  main `
  refs/tags/legacy-audit-v0.1.0 `
  refs/tags/protocol-v1.0.0 `
  refs/tags/protocol-v1.2.0 `
  refs/tags/benchmark-v0.1.0 `
  refs/tags/benchmark-v0.1.1 `
  refs/tags/benchmark-v0.1.2
Assert-NativeSuccess 'atomic main and tag push'
$remoteRefLines = @(git ls-remote origin `
  refs/heads/main `
  refs/tags/legacy-audit-v0.1.0 'refs/tags/legacy-audit-v0.1.0^{}' `
  refs/tags/protocol-v1.0.0 'refs/tags/protocol-v1.0.0^{}' `
  refs/tags/protocol-v1.2.0 'refs/tags/protocol-v1.2.0^{}' `
  refs/tags/benchmark-v0.1.0 'refs/tags/benchmark-v0.1.0^{}' `
  refs/tags/benchmark-v0.1.1 'refs/tags/benchmark-v0.1.1^{}' `
  refs/tags/benchmark-v0.1.2 'refs/tags/benchmark-v0.1.2^{}' |
  Tee-Object -FilePath ".audit/local-candidate/$candidate/remote-refs.txt")
Assert-NativeSuccess 'remote ref verification'
$remoteRefs = @{}
foreach ($line in $remoteRefLines) {
  if ($line -notmatch '^([0-9a-f]{40})\s+(.+)$' -or $remoteRefs.ContainsKey($Matches[2])) {
    throw "Malformed or duplicate remote-ref row: $line"
  }
  $remoteRefs[$Matches[2]] = $Matches[1]
}
$expectedRemoteRefs = @{
  'refs/heads/main' = $candidate
}
foreach ($tag in @(
  'legacy-audit-v0.1.0', 'protocol-v1.0.0', 'protocol-v1.2.0',
  'benchmark-v0.1.0', 'benchmark-v0.1.1', 'benchmark-v0.1.2'
)) {
  $tagObject = (git rev-parse "refs/tags/$tag").Trim()
  Assert-NativeSuccess "resolve local tag object $tag"
  $tagTarget = (git rev-parse "refs/tags/$tag^{}").Trim()
  Assert-NativeSuccess "resolve local peeled tag target $tag"
  $expectedRemoteRefs["refs/tags/$tag"] = $tagObject
  $expectedRemoteRefs["refs/tags/$tag^{}"] = $tagTarget
}
if ($remoteRefs.Count -ne $expectedRemoteRefs.Count) {
  throw 'Remote ref count differs from the exact expected set'
}
foreach ($name in $expectedRemoteRefs.Keys) {
  if (-not $remoteRefs.ContainsKey($name) -or
      $remoteRefs[$name] -ne $expectedRemoteRefs[$name]) {
    throw "Remote ref differs: $name"
  }
}
```

Verify remote `main` and all four tag objects/peeled targets with `git
ls-remote`; preserve the output. The successful exact-main CI repository scan
then independently validates the policy-pinned tag objects fetched from the
remote and binds their normalized ref digest. The `release-security` job runs only for a `push` to
`refs/heads/main`; pull requests and manual runs execute the synthetic matrix
but do not claim release evidence.

The workflow and release policy pin the verified Node 24 implementations of
checkout v7.0.1, setup-python v6.2.0, setup-uv v10.0.1, and upload-artifact
v7.0.1 by full commit SHA. Re-audit those immutable pins when preparing a later
candidate; never replace them with floating major tags.

## 5. Require completed remote CI and capture its evidence

Select only the CI run with event `push`, branch `main`, exact candidate
`headSha`, status `completed`, and conclusion `success`. A PR, manual, tag,
older-head, or running job is not evidence. Require exactly one unexpired
artifact named `release-security-$candidate`.

The CI artifact contains only:

- `repository_scan.json`, `secret_scan.json`, and `license_audit.json`;
- `quality_gates.json`, whose `ci_quality_proxy` means only that preceding job
  steps passed;
- `ci_bundle_validation.json`;
- reviewed quality status/exit/timestamp/log files.

CI does **not** produce a final exact report, a completed-workflow record, or an
inventory spec. Download the artifact promptly; retention is 30 days.

Save the exact repository, main-ref, workflow-run, and run-artifacts API
responses. The run-artifacts response must have `total_count: 1` and exactly
one matching item. Download the artifact ZIP without PowerShell 5 text
redirection. The following helper redirects the native process handle directly
to a create-only file and checks its exit status:
Then use the create-only capture module. It validates API identity, artifact
digest, ZIP paths, duplicates, encryption, symlinks, size caps, exact file set,
run/attempt/event/branch/SHA, and mirrors only sanitized files:

```powershell
$external = "C:\path\outside-repository\inclusive-shift-har-$candidate"
if (Test-Path -LiteralPath $external) { throw "$external already exists; inspect and stop" }
New-Item -ItemType Directory -Path $external -ErrorAction Stop | Out-Null
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

function Save-GhApiResponse([string]$Endpoint, [string]$Destination) {
  if (Test-Path -LiteralPath $Destination) {
    throw "Refusing to overwrite API evidence: $Destination"
  }
  $stderr = "$Destination.stderr"
  $process = Start-Process -FilePath (Get-Command gh).Source `
    -ArgumentList @('api', $Endpoint) -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $Destination -RedirectStandardError $stderr
  if ($process.ExitCode -ne 0 -or (Get-Item -LiteralPath $Destination).Length -eq 0) {
    throw "gh api failed for $Endpoint; preserve $stderr"
  }
}

$runs = gh run list --repo $repository --workflow ci.yml --branch main `
  --event push --commit $candidate --limit 20 `
  --json databaseId,status,conclusion,headSha,event,headBranch | ConvertFrom-Json
Assert-NativeSuccess 'candidate workflow-run selection'
$matches = @($runs | Where-Object {
  $_.status -eq 'completed' -and $_.conclusion -eq 'success' -and
  $_.headSha -eq $candidate -and $_.event -eq 'push' -and $_.headBranch -eq 'main'
})
if ($matches.Count -ne 1) { throw 'Exactly one completed successful candidate run is required' }
$runId = $matches[0].databaseId

Save-GhApiResponse "repos/$repository" "$external/repository-api.json"
Save-GhApiResponse "repos/$repository/git/refs/heads/main" `
  "$external/main-ref-api.json"
Save-GhApiResponse "repos/$repository/actions/runs/$runId" `
  "$external/workflow-run-api.json"
Save-GhApiResponse `
  "repos/$repository/actions/runs/$runId/artifacts?name=release-security-$candidate" `
  "$external/run-artifacts-api.json"
$artifactList = Get-Content -Raw "$external/run-artifacts-api.json" | ConvertFrom-Json
if ($artifactList.total_count -ne 1 -or @($artifactList.artifacts).Count -ne 1) {
  throw 'Exactly one candidate release-security artifact is required'
}
$artifactId = $artifactList.artifacts[0].id
Save-GhApiResponse "repos/$repository/actions/artifacts/$artifactId/zip" `
  "$external/release-security-$candidate.zip"

uv run python -m inclusive_shift_har.artifacts.github_evidence remote `
  --repository-response "$external/repository-api.json" `
  --main-ref-response "$external/main-ref-api.json" `
  --candidate-commit $candidate --queried-at-utc $now `
  --allowed-output-root $external --output remote.json
Assert-NativeSuccess 'remote evidence capture'

uv run python -m inclusive_shift_har.artifacts.github_evidence ci `
  --workflow-run-response "$external/workflow-run-api.json" `
  --artifact-response "$external/run-artifacts-api.json" `
  --downloaded-archive "$external/release-security-$candidate.zip" `
  --candidate-commit $candidate --queried-at-utc $now `
  --extraction-root $external --allowed-output-root $external `
  --output ci.json
Assert-NativeSuccess 'external CI evidence capture'
```

Run the CI capture a second time into the repository root only if a fresh local
`.audit/release-attestations/$candidate/` does not already exist; that mirrored
layout is needed to assemble the exact ignored candidate report. Put the second
capture record outside that destination, for example
`.audit/release-capture/$candidate/ci.json`; otherwise output-directory
preflight would collide with the create-only mirror. Never merge or overwrite
two capture attempts.

## 6. Assemble the exact ignored candidate attestation

Copy the original precommit staged-index record byte-for-byte into the fresh
candidate attestation directory. Prepare a reviewed strict-JSON
`attestation_spec.json` with exactly:

- `schema_version: 1.0.0`;
- `spec_kind: final_release_gate_attestation_spec`;
- canonical UTC timestamp, candidate, first parent, and `worktree_clean: true`;
- all exact gates: staged-index, repository, secret, licence, tests, lint,
  format, types, manifests, splits, configuration, artifacts, and
  `ci_quality_proxy`;
- for each gate, `status: pass`, repository-relative path, and exact file
  SHA-256.

The repository/secret/licence records and generic quality record must be the
downloaded CI bytes. The staged-index record must be the precommit bytes. Then:

```powershell
$localCapture = ".audit/release-capture/$candidate"
$localEvidence = ".audit/release-attestations/$candidate"
if (Test-Path -LiteralPath $localEvidence) {
  throw "Local candidate evidence already exists: $localEvidence"
}
New-Item -ItemType Directory -Path $localCapture | Out-Null
uv run python -m inclusive_shift_har.artifacts.github_evidence ci `
  --workflow-run-response "$external/workflow-run-api.json" `
  --artifact-response "$external/run-artifacts-api.json" `
  --downloaded-archive "$external/release-security-$candidate.zip" `
  --candidate-commit $candidate --queried-at-utc $now `
  --extraction-root . --allowed-output-root .audit `
  --output "release-capture/$candidate/ci.json"
Assert-NativeSuccess 'local CI evidence capture'

$evidence = ".audit/release-attestations/$candidate"
$precommitSource = ".audit/release-precommit/$review/staged_index_scan.json"
$precommitDestination = "$evidence/staged_index_scan.json"
if (Test-Path -LiteralPath $precommitDestination) {
  throw "Create-only staged-index destination already exists: $precommitDestination"
}
Copy-Item -LiteralPath $precommitSource -Destination $precommitDestination

# Create and independently review $evidence/attestation_spec.json here, using
# exact hashes of the preserved staged-index and mirrored CI component files.
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
uv run python -m inclusive_shift_har.artifacts.release_gate assemble `
  --repository-root . `
  --policy configs/release/release_gate_policy_v1.json `
  --mode exact_candidate_attestation --created-at-utc $now `
  --spec "$evidence/attestation_spec.json" `
  --output "$evidence/final_release_gate_report.json"
uv run python -m inclusive_shift_har.artifacts.release_gate validate `
  --report "$evidence/final_release_gate_report.json" --repository-root .
Assert-NativeSuccess 'exact candidate attestation validation'

$externalEvidence = "$external/.audit/release-attestations/$candidate"
if (-not (Test-Path -LiteralPath $externalEvidence -PathType Container)) {
  throw "External CI mirror is missing: $externalEvidence"
}
$externalFiles = @(Get-ChildItem -LiteralPath $externalEvidence -File)
if ($externalFiles.Count -ne 45) {
  throw "External CI mirror must contain exactly 45 validated files before augmentation"
}
foreach ($destination in @(
  "$externalEvidence/staged_index_scan.json",
  "$externalEvidence/attestation_spec.json",
  "$externalEvidence/final_release_gate_report.json"
)) {
  if (Test-Path -LiteralPath $destination) {
    throw "Refusing to overwrite external candidate evidence: $destination"
  }
}
Copy-Item -LiteralPath $precommitSource `
  -Destination "$externalEvidence/staged_index_scan.json"
Copy-Item -LiteralPath "$evidence/attestation_spec.json" `
  -Destination "$externalEvidence/attestation_spec.json"
Copy-Item -LiteralPath "$evidence/final_release_gate_report.json" `
  -Destination "$externalEvidence/final_release_gate_report.json"
```

This exact report still does not call its within-job proxy "completed CI." The
separate external `ci_run_evidence` record proves completed remote CI.

## 7. Build and validate the external `ready` inventory

The reviewed external `release-spec.json` remains schema version 1.0 while the
generated inventory is version 1.1. It has exactly these top-level keys:
`schema_version`, `spec_kind`, `requested_status`, `created_at_utc`,
`protocol_tag`, `remote`, `confirmatory_state`, `artifacts`, `gates`,
`postconfirmatory_statuses`, `exact_candidate_attestation`, and `blockers`.

Use `requested_status: ready` and `protocol_tag: protocol-v1.2.0`. All 20
required roles are bound to their canonical candidate paths by the runtime;
substituting another tracked file is rejected. Include all ten
post-confirmatory tracks explicitly, including `blocked`, `incomplete`,
`not_run`, or `not_applicable` tracks, and keep
`primary_claim_eligible: false`. A `ready` inventory requires no open blocker.

Every gate must cite `spec_root` evidence. Generic quality gates cite the same
CI `quality_gates.json`; staged-index cites the preserved precommit record;
tracked-file, secret, and licence gates cite CI component bytes; `ci` cites the
completed external `ci.json`. Quality logs referenced inside the CI record must
also be present under the mirrored `.audit/release-attestations/$candidate/`
layout. Remote state cites `remote.json`. The exact-candidate attestation must
cite `.audit/release-attestations/$candidate/final_release_gate_report.json`.
Keep the four saved raw GitHub API response files named in `remote.json` and
`ci.json`, plus the downloaded Actions ZIP named in `ci.json`, at the external
spec root. Inventory validation reparses and cross-binds each API response,
rehashes the ZIP, reconstructs its exact 45-member set, and byte-binds every
member to the candidate mirror.

The fixture in `tests/test_release_inventory.py` is the executable field-level
spec example. Generation and validation are create-only and never load raw
signals. Neither operation loads sensor arrays or raw target recordings:

```powershell
$spec = "$external/release-spec.json"
uv run python -m inclusive_shift_har.artifacts.release_inventory generate `
  --spec $spec --repository-root . --candidate-commit $candidate `
  --require-clean-worktree --allowed-output-root .audit `
  --destination "release-assets/$candidate/final_release_evidence_inventory.json"
Assert-NativeSuccess 'ready inventory generation'

uv run python -m inclusive_shift_har.artifacts.release_inventory validate `
  --inventory ".audit/release-assets/$candidate/final_release_evidence_inventory.json" `
  --spec $spec --repository-root . --candidate-commit $candidate
Assert-NativeSuccess 'ready inventory offline validation'
```

Changing and merely re-self-hashing an inventory cannot pass: validation
reconstructs it from candidate Git blobs, exact external bytes, semantic record
shapes, tag/ref state, completed CI, remote state, canonical roles, and the
one-opening/no-rerun state.

## 8. Assemble assets and publish a private prerelease

Build a create-only evidence bundle outside the repository containing:

- exact inventory and exact external release spec;
- every cited external record and sanitized quality log, preserving paths;
- the original downloaded Actions ZIP referenced and reconstructed by `ci.json`;
- exact ignored candidate report and its attestation spec;
- staged-index and staged-secret review evidence;
- sanitized local CUDA gate/tool/machine records.

Exclude raw data, checkpoints, the source ZIP, raw Gitleaks findings, secrets,
absolute workstation paths, and unreviewed logs. Extract the completed bundle
into a new directory and rerun offline inventory/report validation before
uploading it.

The benchmark tag already has its policy-required fixed message; do not recreate
it to add asset hashes. Record inventory file/record hashes and the evidence
bundle hash in the draft release notes instead. The package version is
`0.1.2a0`, so this is a prerelease and must not be marked latest:

```powershell
gh release create benchmark-v0.1.2 `
  --repo $repository --verify-tag --draft --prerelease --latest=false `
  --title "InclusiveShift-HAR benchmark v0.1.2 prerelease" `
  --notes-file "$external/release-notes.md" `
  ".audit/release-assets/$candidate/final_release_evidence_inventory.json" `
  "$external/inclusive-shift-har-$candidate-evidence.zip"
Assert-NativeSuccess 'private draft prerelease creation'
```

Download both draft assets into a fresh directory, compare names, byte sizes,
GitHub digests, and local SHA-256 values, extract the bundle, and reconstruct the
inventory offline. Only then publish with `draft=false`, `prerelease=true`, and
`latest=false`.

Final checks must prove: repository still private; remote `main` equals the
candidate; all six annotated tag objects and targets match policy; release is
published as a private prerelease; assets are byte-identical; local worktree is
clean; no DOI was minted. Keep the repository private until a separate public,
licence, claim, and lineage audit authorizes disclosure.
