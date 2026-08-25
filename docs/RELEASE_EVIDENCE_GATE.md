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
$toolRoot = ".audit/tools/$review/gitleaks-8.30.1-windows-x64"
if (Test-Path -LiteralPath $toolRoot) { throw "$toolRoot already exists; inspect and stop" }
New-Item -ItemType Directory -Path $toolRoot -ErrorAction Stop | Out-Null
gh release download v8.30.1 --repo gitleaks/gitleaks `
  --pattern gitleaks_8.30.1_windows_x64.zip --dir $toolRoot
Assert-NativeSuccess 'Gitleaks archive download'
$archive = Join-Path $toolRoot 'gitleaks_8.30.1_windows_x64.zip'
if ((Get-Item -LiteralPath $archive).Length -ne 8438883 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant() -cne
      'd29144deff3a68aa93ced33dddf84b7fdc26070add4aa0f4513094c8332afc4e') {
  throw 'Gitleaks Windows archive pin differs'
}
$expanded = Join-Path $toolRoot 'expanded'
New-Item -ItemType Directory -Path $expanded -ErrorAction Stop | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $expanded -ErrorAction Stop
$gitleaks = Join-Path $expanded 'gitleaks.exe'
if ((Get-Item -LiteralPath $gitleaks).Length -ne 22575104 -or
    (Get-FileHash -Algorithm SHA256 -LiteralPath $gitleaks).Hash.ToLowerInvariant() -cne
      '17157e2ee8b76fc8b1d8bee607a250e34b8a8023c8bc81822d4b5ee4d78fcb7c') {
  throw 'Gitleaks Windows executable pin differs'
}
if ((& $gitleaks version).Trim() -cne '8.30.1') { throw 'Gitleaks version differs' }
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
  if ($firstIndex.$field -cne $secondIndex.$field) {
    throw "Reviewed index changed before commit: $field"
  }
}
git diff --cached --check
Assert-NativeSuccess 'final cached whitespace check'
if (git diff --name-only) { throw "Tracked worktree changed after index review" }

git commit -m "Prepare benchmark release candidate"
Assert-NativeSuccess 'release-candidate commit'
$candidate = (git rev-parse HEAD).Trim()
if ((git rev-parse "$candidate^").Trim() -cne $base) { throw "Candidate parent changed" }
if (git status --porcelain=v1 --untracked-files=all) { throw "Candidate is not clean" }
if ((git branch --show-current).Trim() -cne 'main') { throw "Candidate is not on main" }
uv run python -m inclusive_shift_har.artifacts.release_gate validate `
  --report results/release/final_release_gate_report.json --repository-root .
Assert-NativeSuccess 'candidate-bound tracked pending-report validation'
```

Rerun all Section 1 gates against this clean candidate through the candidate-
bound capture command below. It executes the full tests, static checks,
manifests, splits, configuration, artifacts, Git integrity, and an actual CUDA
allocation without invoking the confirmatory evaluator. Raw logs remain ignored
local evidence; the self-hashed record cites sanitized copies for the release
bundle. Also bind the already-completed staged Gitleaks report to the reviewed
index and new candidate:

```powershell
$local = ".audit/local-candidate/$candidate"
uv run python -m inclusive_shift_har.artifacts.release_local_evidence `
  capture-local-gates --repository-root . --candidate-commit $candidate `
  --output-root $local
Assert-NativeSuccess 'candidate-bound local CUDA gates'

$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
uv run python -m inclusive_shift_har.artifacts.release_local_evidence `
  attest-staged-gitleaks --repository-root . --candidate-commit $candidate `
  --policy configs/release/release_gate_policy_v1.json `
  --staged-index ".audit/release-precommit/$review/staged_index_scan.json" `
  --raw-report $stagedSecretReport --gitleaks-executable $gitleaks `
  --gitleaks-exit-code $stagedGitleaksExit --created-at-utc $now `
  --output "$local/staged_secret_scan.json" --allowed-output-root $local
Assert-NativeSuccess 'staged Gitleaks attestation'
```

Do not create another content commit afterward.

## 3. Create and validate the benchmark tag before release CI

The release policy byte-pins all eight historical annotated tags. This includes
the immutable `benchmark-v0.1.0` through `benchmark-v0.1.4` candidates. Runs
`32799146947` and `32801378375` failed before tests on invalid frozen-option
combinations; run `32802922698` passed Ubuntu but exposed CRLF checkout
conversion in six Windows byte-hash tests; run `32811935288` passed both
operating-system matrices before its Linux licence audit identified the
XGBoost-transitive NCCL runtime; run `32829208254` exposed a Linux-only
synthetic-fixture mismatch after that licence gate was hardened. Its Windows
matrix passed tests, lint, formatting, and mypy, while release-security was
skipped because Ubuntu failed. The policy also retains the preserved
`protocol-v1.0.0` name/message mismatch. It permits exactly one
successor candidate tag with fixed metadata. Create it only after the pre-tag
local gates pass:

```powershell
git show-ref --verify --quiet refs/tags/benchmark-v0.1.5
$tagProbe = $LASTEXITCODE
if ($tagProbe -eq 0) {
  throw "benchmark-v0.1.5 already exists; inspect it and stop"
}
if ($tagProbe -ne 1) { throw "Unable to determine benchmark tag state" }
if ((git config --get user.name).Trim() -cne 'Abdulla Huseyinli' -or
    (git config --get user.email).Trim() -cne 'abdullahuseyinli@gmail.com') {
  throw 'Git tagger identity differs from the release policy'
}
git tag -a benchmark-v0.1.5 $candidate `
  -m "InclusiveShift-HAR benchmark v0.1.5"
Assert-NativeSuccess 'benchmark tag creation'
if ((git cat-file -t benchmark-v0.1.5).Trim() -cne 'tag') { throw "Tag is not annotated" }
if ((git rev-parse 'benchmark-v0.1.5^{commit}').Trim() -cne $candidate) {
  throw "Benchmark tag targets another commit"
}
```

The policy validates the raw UTF-8 tag object: direct `type commit`, exact tag
name, exact one-line message, exact tagger name/email, canonical headers, and
candidate target. Gitleaks does not inspect annotated-tag payloads, so this
independent byte-level policy is mandatory. For a later release, move this tag
to the pinned historical set in a new policy and declare a new candidate tag;
never retarget `benchmark-v0.1.0`, `benchmark-v0.1.1`, `benchmark-v0.1.2`,
`benchmark-v0.1.3`, or `benchmark-v0.1.4`.

Now create candidate-bound local repository, secret, and licence evidence in a
fresh path. The repository scanner requires all policy-declared tags to exist,
rejects non-commit refs, lightweight/unreviewed/nested tags, tag-only blobs,
replace refs, grafts, shallow history, raw/checkpoint/archive paths, symlinks,
submodules, unsupported binaries, and blobs over 16 MiB.

```powershell
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$local = ".audit/local-candidate/$candidate"
if (-not (Test-Path -LiteralPath $local -PathType Container) -or
    -not (Test-Path -LiteralPath "$local/local_cuda_gate.json" -PathType Leaf) -or
    -not (Test-Path -LiteralPath "$local/staged_secret_scan.json" -PathType Leaf)) {
  throw 'Candidate-bound local gate or staged-secret evidence is missing'
}
foreach ($newEvidence in @('repository_scan.json','gitleaks.json',
    'secret_scan.json','python_licenses.json','license_audit.json')) {
  if (Test-Path -LiteralPath "$local/$newEvidence") {
    throw "Create-only local evidence already exists: $newEvidence"
  }
}

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

On Linux x86-64, the audit expects XGBoost 3.2.0 and its exact
`nvidia-nccl-cu12==2.31.2` edge. The exception is valid only when the candidate
`uv.lock` hash, dependency marker, wheel URL/name/size/hash, PyPI metadata,
embedded licence file, and archived NVIDIA page bytes reconstruct the tracked
review in `docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json`. The audit record
contains the complete normalized package/licence list so CI, inventory, and
bundle validators can independently reproduce the exception and violations.
Omitting either the XGBoost parent or NCCL package, changing platform or
artifact provenance, or emitting an empty/fabricated exception application
fails closed. The wheel remains a locally installed third-party runtime and is
not included in repository history, release assets, or container images. The
review records that the official `nccl_2312` archive paths render a 2.29.2 page
label; those page bytes are supplementary and are not described as
distribution-specific 2.31.2 licence text.

The official Gitleaks 8.30.1 Linux x64 archive SHA-256 is
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`.
The Windows archive and executable pins used above are also recorded in the
versioned release policy.

## 4. Validate the existing private remote and push atomically

The private repository was created during the preserved v0.1.0 release attempt.
Fail closed unless the authenticated owner, configured origin, repository
visibility, and default branch are exactly the expected existing remote. Never
recreate, transfer, rename, or overwrite it.

```powershell
gh auth status
Assert-NativeSuccess 'GitHub authentication'
$owner = (gh api user --jq .login).Trim()
Assert-NativeSuccess 'authenticated GitHub owner lookup'
$repository = "$owner/inclusive-shift-har"
$remoteState = gh api "repos/$repository" | ConvertFrom-Json
Assert-NativeSuccess 'existing repository lookup'
if (-not $remoteState.private -or $remoteState.full_name -cne $repository -or
    $remoteState.default_branch -cne 'main') {
  throw "Existing remote is not the intended private main repository"
}
$expectedOrigin = "https://github.com/$repository.git"
$actualOrigin = (git remote get-url origin).Trim()
Assert-NativeSuccess 'origin lookup'
if ($actualOrigin -cne $expectedOrigin) {
  throw "Origin differs from authenticated private repository: $actualOrigin"
}
$remoteBase = (git ls-remote origin refs/heads/main).Split("`t")[0].Trim()
Assert-NativeSuccess 'remote main concurrency check'
if ($remoteBase -cne $base) {
  throw "Remote main moved after review: expected $base, observed $remoteBase"
}
$candidateTagProbe = @(git ls-remote origin refs/tags/benchmark-v0.1.5 `
  'refs/tags/benchmark-v0.1.5^{}')
Assert-NativeSuccess 'remote candidate-tag absence check'
if ($candidateTagProbe.Count -ne 0) {
  throw 'Remote benchmark-v0.1.5 already exists; inspect it and stop'
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
  refs/tags/benchmark-v0.1.2 `
  refs/tags/benchmark-v0.1.3 `
  refs/tags/benchmark-v0.1.4 `
  refs/tags/benchmark-v0.1.5
Assert-NativeSuccess 'atomic main and tag push'
$remoteRefLines = @(git ls-remote origin `
  refs/heads/main `
  refs/tags/legacy-audit-v0.1.0 'refs/tags/legacy-audit-v0.1.0^{}' `
  refs/tags/protocol-v1.0.0 'refs/tags/protocol-v1.0.0^{}' `
  refs/tags/protocol-v1.2.0 'refs/tags/protocol-v1.2.0^{}' `
  refs/tags/benchmark-v0.1.0 'refs/tags/benchmark-v0.1.0^{}' `
  refs/tags/benchmark-v0.1.1 'refs/tags/benchmark-v0.1.1^{}' `
  refs/tags/benchmark-v0.1.2 'refs/tags/benchmark-v0.1.2^{}' `
  refs/tags/benchmark-v0.1.3 'refs/tags/benchmark-v0.1.3^{}' `
  refs/tags/benchmark-v0.1.4 'refs/tags/benchmark-v0.1.4^{}' `
  refs/tags/benchmark-v0.1.5 'refs/tags/benchmark-v0.1.5^{}' |
  Tee-Object -FilePath ".audit/local-candidate/$candidate/remote-refs.txt")
Assert-NativeSuccess 'remote ref verification'
$remoteRefs = [System.Collections.Generic.Dictionary[string,string]]::new(
  [System.StringComparer]::Ordinal
)
foreach ($line in $remoteRefLines) {
  if ($line -notmatch '^([0-9a-f]{40})\s+(.+)$' -or $remoteRefs.ContainsKey($Matches[2])) {
    throw "Malformed or duplicate remote-ref row: $line"
  }
  $remoteRefs[$Matches[2]] = $Matches[1]
}
$expectedRemoteRefs = [System.Collections.Generic.Dictionary[string,string]]::new(
  [System.StringComparer]::Ordinal
)
$expectedRemoteRefs.Add('refs/heads/main', $candidate)
foreach ($tag in @(
  'legacy-audit-v0.1.0', 'protocol-v1.0.0', 'protocol-v1.2.0',
  'benchmark-v0.1.0', 'benchmark-v0.1.1', 'benchmark-v0.1.2', 'benchmark-v0.1.3',
  'benchmark-v0.1.4', 'benchmark-v0.1.5'
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
      $remoteRefs[$name] -cne $expectedRemoteRefs[$name]) {
    throw "Remote ref differs: $name"
  }
}
```

Verify remote `main` and all nine tag objects/peeled targets with `git
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
  $_.status -ceq 'completed' -and $_.conclusion -ceq 'success' -and
  $_.headSha -ceq $candidate -and $_.event -ceq 'push' -and $_.headBranch -ceq 'main'
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

## 8. Assemble exact assets and publish a private prerelease

The final outer ZIP is fail-closed. Its standard member spec enumerates exactly
the 45 validated Actions members, saved GitHub API responses, original Actions
ZIP, candidate attestation, inventory/spec, staged-index and staged-secret
attestations, local licence audit, candidate-bound CUDA record, and only the
sanitized logs cited by that record. The builder rejects raw/checkpoint/model
paths, raw Gitleaks reports, raw dependency inventories, nested archives other
than the pinned Actions ZIP, symlinks, path/case collisions, absolute user
paths, unsafe compression, changed bytes, and any unmanifested outer member.

Copy the create-only inventory and reviewed local evidence into the external
spec root. Preserve raw local logs in `.audit`; copy only `.log` files that are
not `.raw.log`:

```powershell
$inventory = ".audit/release-assets/$candidate/final_release_evidence_inventory.json"
$externalInventory = "$external/final_release_evidence_inventory.json"
$externalLocal = "$external/local"
if ((Test-Path -LiteralPath $externalInventory) -or
    (Test-Path -LiteralPath $externalLocal)) {
  throw 'External bundle preparation paths already exist'
}
Copy-Item -LiteralPath $inventory -Destination $externalInventory
New-Item -ItemType Directory -Path $externalLocal -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath "$local/staged_secret_scan.json" `
  -Destination "$externalLocal/staged_secret_scan.json"
Copy-Item -LiteralPath "$local/local_cuda_gate.json" `
  -Destination "$externalLocal/local_cuda_gate.json"
Copy-Item -LiteralPath "$local/license_audit.json" `
  -Destination "$externalLocal/license_audit.json"
$sanitizedLogs = @(Get-ChildItem -LiteralPath $local -File -Filter '*.log' |
  Where-Object { $_.Name -notlike '*.raw.log' })
if ($sanitizedLogs.Count -lt 1) { throw 'No sanitized local logs found' }
foreach ($log in $sanitizedLogs) {
  Copy-Item -LiteralPath $log.FullName -Destination "$externalLocal/$($log.Name)"
}

$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$bundleSpec = "$external/release-bundle-spec.json"
$bundleManifest = "$external/release-bundle-manifest.json"
$bundle = "$external/inclusive-shift-har-$candidate-evidence.zip"
uv run python -m inclusive_shift_har.artifacts.release_bundle standard-spec `
  --source-root $external --repository-root . --candidate-commit $candidate `
  --created-at-utc $now `
  --output $bundleSpec --allowed-output-root $external
Assert-NativeSuccess 'outer-bundle exact spec assembly'
uv run python -m inclusive_shift_har.artifacts.release_bundle manifest `
  --spec $bundleSpec --source-root $external --created-at-utc $now `
  --output $bundleManifest --allowed-output-root $external
Assert-NativeSuccess 'outer-bundle manifest assembly'
uv run python -m inclusive_shift_har.artifacts.release_bundle build `
  --spec $bundleSpec --manifest $bundleManifest --source-root $external `
  --output $bundle --allowed-output-root $external
Assert-NativeSuccess 'outer-bundle build'
uv run python -m inclusive_shift_har.artifacts.release_bundle notes `
  --inventory $externalInventory --manifest $bundleManifest --archive $bundle `
  --candidate-commit $candidate --output "$external/release-notes.md" `
  --allowed-output-root $external
Assert-NativeSuccess 'candidate-bound release-notes assembly'

$preuploadExtract = "$external-preupload-extract"
$preuploadValidation = "$external/bundle-validation-preupload.json"
uv run python -m inclusive_shift_har.artifacts.release_bundle validate `
  --archive $bundle --spec $bundleSpec --manifest $bundleManifest `
  --candidate-commit $candidate --repository-root . `
  --extraction-root $preuploadExtract --created-at-utc $now `
  --output $preuploadValidation --allowed-output-root $external
Assert-NativeSuccess 'outer-bundle offline reconstruction'

$bundleGitleaks = "$external/bundle-gitleaks.json"
& $gitleaks dir $preuploadExtract --no-banner --config .gitleaks.toml `
  --exit-code 1 --report-format json --report-path $bundleGitleaks
if ($LASTEXITCODE -ne 0) { throw 'Outer-bundle Gitleaks scan failed' }
```

The raw outer-bundle Gitleaks report and pre-upload validation remain external
review evidence and are not retroactively inserted into the already-built ZIP.

The benchmark tag already has its policy-required fixed message; do not recreate
it to add asset hashes. Record inventory file/record hashes and the evidence
bundle hash in the draft release notes instead. The package version is
`0.1.5a0`, so this is a prerelease and must not be marked latest:

```powershell
$releaseTitle = "InclusiveShift-HAR benchmark v0.1.5 prerelease"
$releaseNotesPath = "$external/release-notes.md"
$releaseNotesBody = [System.IO.File]::ReadAllText(
  (Resolve-Path -LiteralPath $releaseNotesPath).Path,
  [System.Text.UTF8Encoding]::new($false)
)
gh release create benchmark-v0.1.5 `
  --repo $repository --verify-tag --draft --prerelease --latest=false `
  --title $releaseTitle --notes-file $releaseNotesPath `
  $externalInventory $bundle
Assert-NativeSuccess 'private draft prerelease creation'
```

Capture the draft REST object, require exactly the two expected assets, compare
GitHub names, sizes, and `sha256:` digests with local bytes, then download both
assets into a fresh directory. Revalidate the downloaded bundle with a new
extraction root before publication:

```powershell
$draftApi = "$external/draft-release-api.json"
Save-GhApiResponse "repos/$repository/releases?per_page=100" $draftApi
$draftMatches = @((Get-Content -Raw -LiteralPath $draftApi | ConvertFrom-Json) |
  Where-Object { $_.tag_name -ceq 'benchmark-v0.1.5' })
if ($draftMatches.Count -ne 1) { throw 'Expected exactly one draft release' }
$draft = $draftMatches[0]
if (-not $draft.draft -or -not $draft.prerelease -or $draft.assets.Count -ne 2) {
  throw 'Draft release state or exact asset count differs'
}
if ($draft.name -cne $releaseTitle -or $draft.body -cne $releaseNotesBody) {
  throw 'Draft release title or notes differ from generated metadata'
}
$expectedAssets = [System.Collections.Generic.Dictionary[string,string]]::new(
  [System.StringComparer]::Ordinal
)
$expectedAssets.Add((Split-Path -Leaf $externalInventory), $externalInventory)
$expectedAssets.Add((Split-Path -Leaf $bundle), $bundle)
foreach ($asset in $draft.assets) {
  if (-not $expectedAssets.ContainsKey($asset.name)) {
    throw "Unexpected draft asset: $($asset.name)"
  }
  $source = $expectedAssets[$asset.name]
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
  if ($asset.size -ne (Get-Item -LiteralPath $source).Length -or
      $asset.digest -cne "sha256:$hash") {
    throw "Draft asset metadata differs: $($asset.name)"
  }
}

$draftDownload = "$external-draft-download"
if (Test-Path -LiteralPath $draftDownload) { throw 'Draft download path exists' }
New-Item -ItemType Directory -Path $draftDownload -ErrorAction Stop | Out-Null
gh release download benchmark-v0.1.5 --repo $repository --dir $draftDownload
Assert-NativeSuccess 'draft asset download'
foreach ($name in $expectedAssets.Keys) {
  $downloaded = Join-Path $draftDownload $name
  if (-not (Test-Path -LiteralPath $downloaded -PathType Leaf) -or
      (Get-FileHash -Algorithm SHA256 -LiteralPath $downloaded).Hash -cne
      (Get-FileHash -Algorithm SHA256 -LiteralPath $expectedAssets[$name]).Hash) {
    throw "Downloaded draft asset differs: $name"
  }
}
$downloadExtract = "$external-draft-extract"
$downloadValidation = "$external/bundle-validation-draft-download.json"
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
uv run python -m inclusive_shift_har.artifacts.release_bundle validate `
  --archive (Join-Path $draftDownload (Split-Path -Leaf $bundle)) `
  --spec $bundleSpec --manifest $bundleManifest --candidate-commit $candidate `
  --repository-root . --extraction-root $downloadExtract --created-at-utc $now `
  --output $downloadValidation --allowed-output-root $external
Assert-NativeSuccess 'downloaded draft bundle validation'

gh release edit benchmark-v0.1.5 --repo $repository `
  --draft=false --prerelease --latest=false
Assert-NativeSuccess 'private prerelease publication'

$publishedApi = "$external/published-release-api.json"
Save-GhApiResponse "repos/$repository/releases/tags/benchmark-v0.1.5" $publishedApi
$published = Get-Content -Raw -LiteralPath $publishedApi | ConvertFrom-Json
if ($published.draft -or -not $published.prerelease -or
    $published.tag_name -cne 'benchmark-v0.1.5' -or
    $published.assets.Count -ne 2) {
  throw 'Published prerelease state differs from the reviewed draft'
}
if ($published.id -ne $draft.id) { throw 'Published release ID differs from reviewed draft' }
if ($published.name -cne $releaseTitle -or $published.body -cne $releaseNotesBody -or
    $published.name -cne $draft.name -or $published.body -cne $draft.body) {
  throw 'Published release title or notes differ from the reviewed draft'
}
$draftAssetsByName = [System.Collections.Generic.Dictionary[string,object]]::new(
  [System.StringComparer]::Ordinal
)
foreach ($asset in $draft.assets) { $draftAssetsByName[$asset.name] = $asset }
foreach ($asset in $published.assets) {
  if (-not $draftAssetsByName.ContainsKey($asset.name)) {
    throw "Published asset name differs: $($asset.name)"
  }
  $draftAsset = $draftAssetsByName[$asset.name]
  if ($asset.id -ne $draftAsset.id -or $asset.size -ne $draftAsset.size -or
      $asset.digest -cne $draftAsset.digest) {
    throw "Published asset identity differs from reviewed draft: $($asset.name)"
  }
  $source = $expectedAssets[$asset.name]
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
  if ($asset.size -ne (Get-Item -LiteralPath $source).Length -or
      $asset.digest -cne "sha256:$hash") {
    throw "Published asset bytes differ: $($asset.name)"
  }
}

$publishedDownload = "$external-published-download"
if (Test-Path -LiteralPath $publishedDownload) { throw 'Published download path exists' }
New-Item -ItemType Directory -Path $publishedDownload -ErrorAction Stop | Out-Null
gh release download benchmark-v0.1.5 --repo $repository --dir $publishedDownload
Assert-NativeSuccess 'published asset download'
foreach ($name in $expectedAssets.Keys) {
  $downloaded = Join-Path $publishedDownload $name
  if (-not (Test-Path -LiteralPath $downloaded -PathType Leaf) -or
      (Get-FileHash -Algorithm SHA256 -LiteralPath $downloaded).Hash -cne
      (Get-FileHash -Algorithm SHA256 -LiteralPath $expectedAssets[$name]).Hash) {
    throw "Published downloaded asset differs: $name"
  }
}

$finalRepositoryApi = "$external/final-repository-api.json"
$finalMainApi = "$external/final-main-ref-api.json"
Save-GhApiResponse "repos/$repository" $finalRepositoryApi
Save-GhApiResponse "repos/$repository/git/refs/heads/main" $finalMainApi
$finalRepository = Get-Content -Raw -LiteralPath $finalRepositoryApi | ConvertFrom-Json
$finalMain = Get-Content -Raw -LiteralPath $finalMainApi | ConvertFrom-Json
if (-not $finalRepository.private) { throw 'Repository is no longer private' }
if ($finalMain.object.type -cne 'commit' -or $finalMain.object.sha -cne $candidate) {
  throw 'Remote main no longer equals the release candidate'
}
$requiredTags = @(
  'benchmark-v0.1.0','benchmark-v0.1.1','benchmark-v0.1.2','benchmark-v0.1.3',
  'benchmark-v0.1.4','benchmark-v0.1.5',
  'legacy-audit-v0.1.0','protocol-v1.0.0','protocol-v1.2.0'
)
foreach ($tag in $requiredTags) {
  if ((git cat-file -t "refs/tags/$tag").Trim() -cne 'tag') {
    throw "Local tag is not annotated: $tag"
  }
  $localObject = (git rev-parse "refs/tags/$tag").Trim()
  $localTarget = (git rev-parse "refs/tags/$tag^{}").Trim()
  if ($localObject -cne $expectedRemoteRefs["refs/tags/$tag"] -or
      $localTarget -cne $expectedRemoteRefs["refs/tags/$tag^{}"]) {
    throw "Local annotated tag moved after the policy-validated pre-push snapshot: $tag"
  }
  if ($tag -ceq 'benchmark-v0.1.5' -and $localTarget -cne $candidate) {
    throw 'Candidate benchmark tag no longer targets the candidate'
  }
  $remoteRows = @(git ls-remote --tags origin "refs/tags/$tag" "refs/tags/$tag^{}")
  Assert-NativeSuccess "remote tag query $tag"
  $remoteTagRefs = [System.Collections.Generic.Dictionary[string,string]]::new(
    [System.StringComparer]::Ordinal
  )
  foreach ($row in $remoteRows) {
    if ($row -notmatch '^([0-9a-f]{40})\s+(.+)$' -or
        $remoteTagRefs.ContainsKey($Matches[2])) {
      throw "Malformed or duplicate final remote tag row: $row"
    }
    $remoteTagRefs[$Matches[2]] = $Matches[1]
  }
  if ($remoteTagRefs.Count -ne 2 -or
      $remoteTagRefs["refs/tags/$tag"] -cne $expectedRemoteRefs["refs/tags/$tag"] -or
      $remoteTagRefs["refs/tags/$tag^{}"] -cne $expectedRemoteRefs["refs/tags/$tag^{}"]) {
    throw "Remote annotated tag identity differs: $tag"
  }
}
$finalPolicyScan = ".audit/local-candidate/$candidate/final_repository_scan.json"
if (Test-Path -LiteralPath $finalPolicyScan) {
  throw 'Final policy scan destination already exists'
}
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
uv run python -m inclusive_shift_har.artifacts.release_gate scan-repository `
  --repository-root . --candidate-commit $candidate `
  --policy configs/release/release_gate_policy_v1.json `
  --created-at-utc $now --output $finalPolicyScan
Assert-NativeSuccess 'final nine-tag policy scan'
if (git status --porcelain=v1 --untracked-files=all) {
  throw 'Final local worktree is not clean'
}
```

Final checks must prove: repository still private; remote `main` equals the
candidate; all nine annotated tag objects and targets match policy; release is
published as a private prerelease; assets are byte-identical; local worktree is
clean; no DOI was minted. Keep the repository private until a separate public,
licence, claim, and lineage audit authorizes disclosure.
