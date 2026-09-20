# GitHub and Zenodo publication checklist

This checklist distinguishes a validated branch update from a formal
versioned release. A branch push publishes the reviewed source revision to the
existing repository; it does not create a tag, GitHub release, Zenodo deposit,
DOI, or change repository visibility.

## 1. Freeze the release scope

- Include repository-authored source, tests, configurations, protocols, cards,
  aggregate evidence, and preserved failure records.
- Keep raw datasets, participant-level unrestricted material, large checkpoints,
  local `.audit` evidence, caches, and temporary third-party references outside
  the Git/Zenodo archive.
- Retain the historical locked target outcome unchanged.
- Present CTGR/HERA as reused-source development and AICOS/HARTH as diagnostics.
- Use [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md) and
  [EVIDENCE_SUPERSESSION.md](EVIDENCE_SUPERSESSION.md) as the claim authorities.

## 2. Review the candidate contents

- Inspect every modified and untracked path.
- Include only first-party AICOS adapters, experiment controllers, confirmation
  code, tests, configurations, and current documentation.
- Keep `.tmp_harth_*` files local. Their redistribution rights are unresolved;
  a local audit receipt preserves their hashes.
- Confirm that no raw sensor files, archives, secrets, credentials, checkpoints,
  model pickles, or unrestricted logs enter the Git index.
- Confirm all Markdown links and all JSON/YAML syntax.

## 3. Validate software and evidence

Run the clean-checkout commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md):

- locked environment and offline lock check;
- complete synthetic test collection;
- Ruff lint and formatting;
- strict mypy;
- manifest validation;
- split audit;
- configuration parsing;
- retained artifact validation;
- `git diff --check` and `git fsck --full`;
- wheel and source-distribution build plus content inspection.

The latest routing experiment's 27 focused tests are supporting evidence, not a
substitute for these candidate-wide gates.

## 4. Verify documentation and metadata

- README, project status, results index, model cards, and supersession map agree.
- Accuracy, pooled macro-F1, and participant macro-F1 are labelled explicitly.
- Cohort, participant count, channels, seeds, supervision, and evidence status
  accompany every headline metric.
- `pyproject.toml`, `src/inclusive_shift_har/__init__.py`, `CITATION.cff`,
  `.zenodo.json`, release notes, and the eventual tag use one version.
- Dataset and external-method citations are complete and licences are not
  conflated with Apache-2.0.
- Zenodo metadata does not claim a DOI before one exists.

## 5. Form the local Git candidate

After review and successful validation:

1. Stage only the reviewed paths.
2. Scan the exact staged index for raw data, large files, credentials, and
   third-party materials.
3. Record the parent commit and staged-index hash.
4. Commit logical first-party changes without rewriting historical tags.
5. Rerun candidate-bound validation against the resulting commit.
6. Merge through the normal review path; do not force-push or move an existing tag.

The detailed [release evidence gate](RELEASE_EVIDENCE_GATE.md) documents the
existing candidate-bound security and remote-verification machinery. Its older
CUDA and tag lineage remains historical evidence; regenerate attestations for the
actual candidate rather than copying an earlier pass.

## 6. GitHub branch update and release

For a research-branch or `main` update:

- push the reviewed branch without rewriting remote history;
- require green CI on the exact head SHA;
- run complete-history and candidate-tree secret scans;
- retain the commit identifier and remote validation evidence.

On `main`, also require the complete-history security job, including its
candidate-bound licence and CI bundle checks. Follow
[MAIN_VERIFICATION.md](MAIN_VERIFICATION.md) for the current immutable tag policy.

For a separately commissioned versioned release:

- review dependency licences;
- create one new annotated tag without moving prior tags;
- build release assets from that tag in a clean environment;
- verify downloaded assets byte-for-byte;
- publish release notes that link the evidence and supersession indexes.

## 7. Zenodo deposition

Create the Zenodo software deposit from the same tagged commit and reviewed
assets. Before publishing:

- replace prerelease version `0.1.7a0` only if a final version is approved;
- update `CITATION.cff` and `.zenodo.json` together;
- include Apache-2.0 repository material only;
- exclude raw data, temporary HARTH files, local `.audit`, and unreviewed
  predictions/checkpoints;
- cite InclusiveHAR, AICOS, HARTH, and every redistributed external artifact
  under its own terms;
- add the issued DOI to citation metadata only after deposition;
- archive the final Zenodo record JSON and checksums as release evidence.

## Publication-ready definition

The candidate is ready for review when it has a coherent evidence index,
no ambiguous current claims, no unreviewed third-party files in scope, a clean
candidate diff, passing full gates, buildable distributions, and matching
metadata. A research branch is distinct from an archived release. A combined
GitHub/Zenodo release is complete when its tag, assets, CI evidence, and Zenodo
record all identify the same source revision.
