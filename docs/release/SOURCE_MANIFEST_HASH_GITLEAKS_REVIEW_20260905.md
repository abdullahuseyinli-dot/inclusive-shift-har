# Reviewed source-manifest false positive

On 2026-09-05 Gitleaks 8.30.1 reported `generic-api-key` in the create-only
real-FoG metamorphic record. Its matched value was exactly SHA-256
`71c41f68123709208a6743fefc65e1f93ea3b6c8ef2176b99435f069f9b492a5`, independently
recomputed from `docs/research/CONTROLLED_DATA_ACCESS_CHECKLIST.md`. It is a
public tracked documentation digest, not an access credential. The original
failed scan and unmodified result are preserved under `.audit/session_grid_v3/`.

The correction adds a rule-local AND allowlist constrained simultaneously to
generated `results/research/cross_dataset_har_v3/*.json` paths, the exact matched
checklist field, and this one verified digest. No arbitrary hexadecimal string,
credential field, other file path, other rule, or future changed digest is
exempted. Existing historical fingerprint exceptions remain unchanged. Tests
check that altered digests and credential fields remain outside the exception.
The release policy's configuration hash is updated to the actual TOML bytes.
This is not permission to hide findings or broaden a secret exclusion later.
