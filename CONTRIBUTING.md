# Contributing to InclusiveShift-HAR

InclusiveShift-HAR is evidence-gated research software. Contributions are
welcome when they preserve participant separation, provenance, and the declared
scientific scope.

## Development setup

Use Python 3.11 and the locked environment:

```powershell
uv sync --locked --extra training-cpu --group research
uv pip install --python .venv --require-hashes -r requirements/external-har-research.lock
```

Follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for validation commands.

## Evidence rules

- Do not reopen InclusiveHAR P11-P20 or use target outcomes for tuning.
- Partition participants before segmentation or windowing.
- Fit preprocessing on training participants and calibrate on validation data.
- Never use identity, disability/device metadata, location, timestamps, labels,
  or order proxies as model features.
- Preserve failures, exclusions, harms, manual interventions, and conditional
  passes in result records.
- Do not overwrite create-only evidence. Use a new versioned destination.
- Label every result as locked, development, post-confirmatory, diagnostic,
  failed, quarantined, or superseded.
- Treat accuracy, pooled macro-F1, and participant macro-F1 as distinct metrics.

## Data and licences

Do not commit raw datasets, source archives, unrestricted participant data,
checkpoints, secrets, caches, or third-party code without verified redistribution
rights. Apache-2.0 covers repository-authored material only. Add provider identity,
version, licence, citation, hashes, and role before introducing a new dataset.

## Code changes

- Keep functions typed and deterministic where the protocol requires it.
- Add tests for leakage boundaries, alignment, unit conversion, evidence gates,
  or nontrivial numerical behavior.
- Avoid tests that simply duplicate implementation details.
- Update the relevant protocol, model/data card, evidence index, and supersession
  map when a change affects scientific interpretation.
- Do not silently change a locked endpoint, split, ontology, seed, calibration
  rule, checkpoint rule, or evidence status.

Run before review:

```powershell
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync mypy src tests
uv run --no-sync python -m inclusive_shift_har.artifacts.parallel_test_gate `
  --output .audit/contributor-test-shards-001 --workers 2
uv run --no-sync inclusive-shift-har validate-manifests --json
uv run --no-sync inclusive-shift-har validate-artifacts `
  --artifact-root results --require-artifacts --json
git diff --check
```

## Review checklist

A contribution should state:

- the concrete problem and resulting behavior;
- data, channels, cohort, partitions, seeds, and supervision used;
- primary and lower-tail participant metrics where performance is involved;
- uncertainty, harms, failures, and interface limitations;
- exact validation commands and results;
- third-party licence/provenance implications;
- whether the change affects a historical record or only current interpretation.

Publication, remote pushes, tags, releases, and DOI actions are handled separately
under [docs/PUBLICATION_CHECKLIST.md](docs/PUBLICATION_CHECKLIST.md).
