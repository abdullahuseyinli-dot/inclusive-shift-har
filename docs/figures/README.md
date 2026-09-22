# Research figures

These figures summarize tracked aggregate evidence. They do not introduce new
experiments or combine results from different datasets into a leaderboard.

| Figure | Scope |
|---|---|
| [Source development comparison](source_development_comparison.svg) | Left: seed-11 six-channel methods. Right: five-seed RMRP/CTGR/HERA means, including the additional native-gravity input |
| [Method development map](method_development_map.svg) | Project contribution sequence and the separate target, personalization and external studies |

SVG files provide scalable figures for GitHub and exported documents. PNG copies
are included for viewers without SVG support. Both use the same plotting code.
The [generated tables](source_development_tables.md) contain accuracy and
participant macro-F1; the [receipt](reproduction_receipt.json) binds all five
generated files to their two source records by SHA-256.

From the repository root, regenerate into a new directory:

```powershell
uv run --no-sync python -m inclusive_shift_har.artifacts.research_figures `
  --output .audit/reproduced-research-figures-001
```

The generator validates record hashes, confusion-matrix accuracy, participant
means, seed identities and agreement between the aggregate records before
rendering. No raw data, prediction archives or model fitting are involved.
The [development audit](../research/SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md)
describes the separate prediction-level verification and its restored inputs.

The method map expresses the research sequence, not estimated causal effects.
The comparison figure has a zero-based percentage axis, explicit channel and
seed grouping, and no significance markers. Participant uncertainty and harms
are reported in the [research report](../RESEARCH_REPORT.md).
