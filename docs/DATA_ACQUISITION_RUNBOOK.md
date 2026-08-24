# Official data acquisition and local layout

This runbook is for a fresh or explicitly audited workspace. It downloads only
the artifacts named in the reviewed dataset manifests, verifies their pinned
sizes and SHA-256 digests before publication, and never overwrites an existing
raw file or receipt. Raw data remain outside Git and retain their source terms.

## Environment and preflight

Use CPython 3.11 and the locked CUDA research environment from the repository
root:

```powershell
uv sync --locked --extra training-cuda --group research
uv run inclusive-shift-har validate-manifests --json
```

Create only missing directory containers. Stop if either name is already a
non-directory; do not remove or replace existing contents.

```powershell
$containers = @("data/raw", "results/acquisition")
foreach ($container in $containers) {
  if (Test-Path -LiteralPath $container -PathType Leaf) {
    throw "Expected a directory, found a file: $container"
  }
  if (-not (Test-Path -LiteralPath $container -PathType Container)) {
    New-Item -ItemType Directory -Path $container -ErrorAction Stop | Out-Null
  }
}
```

The acquisition command validates the complete manifest and every destination
before network access. It uses official credential-free HTTPS URLs, requires a
pinned hash, publishes raw files read-only with create-if-absent semantics, and
writes a self-hashed create-only receipt. On a transfer failure it returns
nonzero, retains any verified files and partial transfer, and writes a failure
receipt. Preserve those files; investigate rather than retrying into the same
paths.

## InclusiveHAR version 4

```powershell
uv run python -m inclusive_shift_har.data.acquire_manifest `
  --manifest manifests/datasets/inclusivehar_v4.json `
  --repository-root . `
  --data-root data/raw `
  --receipt-root results/acquisition `
  --receipt-output inclusivehar_v4.manifest_acquisition.json
```

Expected immutable layout:

| Local path below `data/raw` | Bytes | Expected SHA-256 |
|---|---:|---|
| `inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv` | 148,915,541 | `0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34` |
| `inclusivehar/v4/Description of individuals with disabilities.docx` | 16,997 | `b8c033dd630412103b8dd463c8a171aa4727fc7222ba6b7fdc1e2487315e0dfb` |

The source is Mendeley Data version 4, DOI
`10.17632/r78dn3f6nc.4`. The manifest records CC BY 4.0 and the repository policy
`do_not_redistribute_raw`. The DOCX contains sensitive free text: do not copy its
contents into ordinary artifacts.

After acquisition, use the existing read authorization and privacy-safe audit;
never invent a replacement gate:

```powershell
uv run inclusive-shift-har audit-data `
  --manifest manifests/datasets/inclusivehar_v4.json `
  --read-only `
  --data-root data/raw `
  --gate-record results/gates/raw_data_read_access_inclusivehar_v4.json `
  --profile inclusivehar-v4 `
  --json
```

## UCI-HAR version 1

```powershell
uv run python -m inclusive_shift_har.data.acquire_manifest `
  --manifest manifests/datasets/uci_har_v1.json `
  --repository-root . `
  --data-root data/raw `
  --receipt-root results/acquisition `
  --receipt-output uci_har_v1.manifest_acquisition.json
```

Expected downloaded layout:

| Local path below `data/raw` | Bytes | Expected SHA-256 |
|---|---:|---|
| `uci_har/v1/human+activity+recognition+using+smartphones.zip` | 61,005,872 | `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` |
| `uci_har/v1/UCI HAR Dataset.names` | 6,304 | `eaee53f45825f349681a1a1b92a997a0082648f9b5bf93ddb3290caeede4b3a8` |

UCI publishes no archive SHA-256; these are locally observed pins from the
official HTTPS endpoints, not provider-published checksums. The current catalog
says CC BY 4.0, while the embedded Version 1.0 notice prohibits commercial use.
That conflict is unresolved, so raw redistribution remains prohibited.

The official wrapper contains a processed archive used by the adapter. Materialize
only that exact member after verifying the wrapper. This PowerShell operation is
create-only; a mismatch leaves the unique partial file visible for diagnosis.

```powershell
$outer = (Resolve-Path -LiteralPath "data/raw/uci_har/v1/human+activity+recognition+using+smartphones.zip").Path
$outerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outer).Hash.ToLowerInvariant()
if ($outerHash -ne "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031") {
  throw "UCI wrapper hash mismatch"
}
$inner = Join-Path (Split-Path -Parent $outer) "UCI HAR Dataset.zip"
if (Test-Path -LiteralPath $inner) { throw "Refusing to overwrite $inner" }
$partial = "$inner.partial.$([Guid]::NewGuid().ToString('N'))"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($outer)
try {
  $entry = $archive.GetEntry("UCI HAR Dataset.zip")
  if ($null -eq $entry -or $entry.Length -ne 60999314) { throw "Unexpected inner archive member" }
  $input = $entry.Open()
  $output = [IO.File]::Open($partial, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
  try { $input.CopyTo($output); $output.Flush($true) } finally { $output.Dispose(); $input.Dispose() }
} finally { $archive.Dispose() }
$innerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash.ToLowerInvariant()
if ($innerHash -ne "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0") {
  throw "Inner archive hash mismatch; partial preserved at $partial"
}
New-Item -ItemType HardLink -Path $inner -Target $partial -ErrorAction Stop | Out-Null
Remove-Item -LiteralPath $partial -ErrorAction Stop
(Get-Item -LiteralPath $inner).IsReadOnly = $true
```

The resulting `data/raw/uci_har/v1/UCI HAR Dataset.zip` must be 60,999,314
bytes with SHA-256
`2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0`.
The checked-in acquisition receipt records the audited wrapper inventory and a
preserved historical path error; do not delete that evidence.

## Completion gate

Acquisition is complete only when manifest validation passes, every expected
file has the exact size/hash above, the create-only receipt is complete and
self-hashed, the appropriate read-only audit passes, and `git status --short`
shows no raw data staged or tracked. A successful download alone is not a data
or protocol acceptance result.
