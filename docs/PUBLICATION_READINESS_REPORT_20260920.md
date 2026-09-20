# Publication readiness report

**Review date:** 2026-09-20

**Candidate status:** ready for repository-owner review; not committed, tagged,
pushed, released, or deposited.

**Candidate parent:** `279c53d2ace6f28546c2246c866fdcaa324301a6`

**Branch:** `research/har-substantiation-pilot-pipeline-20260908`

## Review conclusion

The repository now has a coherent public entry point, a current evidence index,
an append-only supersession map, model and reproducibility documentation,
publication metadata, contribution/security guidance, and an explicit release
checklist. Historical scientific records remain preserved. The current package
can be reviewed as an exploratory research-software and evidence candidate.

This state does not authorize or claim a GitHub release, immutable tag, Zenodo
deposit, DOI, independent confirmation, clinical validity, or state-of-the-art
performance. Those actions require approval of an exact commit and completion of
the candidate-bound remote gates.

## Main adjustments

- Replaced the stale repository landing page with a concise status, comparable
  result groups, contribution boundaries, quick start, and repository map.
- Added current evidence, supersession, CTGR/HERA model, reproducibility, and
  publication-control documents.
- Updated `CITATION.cff` and `.zenodo.json` to use one title, version, author,
  licence scope, and non-claiming description.
- Classified the paper outline and addendum as historical drafts so they cannot
  be mistaken for the final evidence narrative.
- Added a machine-readable publication evidence record and tests binding every
  headline metric to its canonical tracked source.
- Included the corrected AICOS adapter, fixed routing validation, and prepared
  native-nine confirmation code with their contract tests.
- Excluded preserved `.tmp_harth_*` reference inputs from Git and distribution
  scope without deleting them.
- Added narrowly scoped Gitleaks allow-list rules for four public internal
  identifiers; the rules require both an exact path and exact full match.
- Restored one final blank-line byte in the FoG compact protocol from its
  authenticated run snapshot. The restored SHA-256
  `b5256e8268899e7ca9d0855a3c7d76fd511cae3e609dbe96e9580350adc57ec3`
  matches the frozen configuration and runner.

## Validation evidence

| Gate | Result |
|---|---|
| Complete synthetic suite | 1,507 tests reported; 0 failures, 0 errors, 4 documented CUDA-only skips |
| Ruff lint | Pass |
| Ruff formatting | 346 files formatted; pass |
| Strict mypy | 346 source files; no issues |
| Dependency lock | `uv lock --check --offline` passed |
| Dataset manifests | 3 manifests valid |
| Released-block split audit | `pass_conditional_released_block`; target not accessed; three historical boundary/rate warnings retained |
| Artifact validation | 12 files across 2 manifests valid; one existing quarantine status remains visible |
| Configuration and metadata parse | 120 configuration files plus `CITATION.cff` and `.zenodo.json` parsed |
| Local Markdown links | 130 links across 137 Markdown files; none missing |
| Git worktree whitespace | Pass, with three documented byte-preservation attributes |
| Git object integrity | Pass; unreachable blobs reported by `git fsck` are not corrupt objects |
| Distribution build | Wheel and source archive built successfully as version `0.1.7a0` |
| Wheel smoke check | Installed into an isolated audit target and imported as `0.1.7a0` |
| Archive scope | No raw-data, `.audit`, `.tmp_harth_*`, credential, cache, or checkpoint path included |
| Pinned Gitleaks 8.30.1 | Source archive scanned after exact scoped false-positive review; 0 findings |

The clean full-suite record is
`.audit/publication_candidate_validation_20260920-004/test-shards/summary.json`.
Earlier failed attempts remain preserved under candidate directories `-001` and
`-003`; they exposed the missing protocol byte and the attributes-order contract
and were not overwritten. Final secret-scan and build inspection records are in
the latest numbered candidate-validation directory.

## Scientific claim boundary

- The one-time InclusiveHAR target opening remains unchanged. Compact DANN and
  CORAL were effectively tied, and the preregistered MoRe-HAR hypothesis was not
  supported.
- CTGR is the retained source-development advancement. Strict HERA-v1 is the
  highest matched source point estimate, but its incremental interval crosses
  zero and its promotion gates failed.
- AICOS and HARTH remain external diagnostics with their recorded unit,
  coordinate-interface, placement, and endpoint limitations.
- The latest U9 routing rule is closed after source regression and no practical
  AICOS development gain.
- No result from reused participants is described as independent confirmation.

## Owner review and publication boundary

The owner can now review the candidate through the
[evidence index](EVIDENCE_INDEX.md), [project status](PROJECT_STATUS.md),
[documentation map](README.md), and the Git diff. Before publication, the owner
must choose whether `0.1.7a0` remains a prerelease or becomes a new final version
and approve the exact candidate contents.

After that approval, form the real Git index, rerun the exact staged-index and
complete-history security gates, create one immutable commit, require green CI on
that commit, and only then create a new tag, GitHub release, and matching Zenodo
deposit. Add a DOI only after Zenodo issues it.
