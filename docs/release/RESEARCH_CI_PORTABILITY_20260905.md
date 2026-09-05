# Research-branch CI portability corrections

The first research-branch [CI run](https://github.com/abdullahuseyinli-dot/inclusive-shift-har/actions/runs/33942883072)
at `5a585e9764a82d9e40bc805b684e13274551fdb0` failed. Its logs, collection plan,
command receipts and JUnit artifacts are preserved under
`.audit/session_grid_v3/ci_33942883072`. This does not erase the complete local
799-test pass, but the local result did not establish remote portability.

Windows collected all 799 tests but could launch neither of its two shards.
Their serialized command lines were 40,061 and 44,218 characters; both produced
`WinError 206`. The local four-shard gate had shorter commands. The correction
uses create-only, hash-bound pytest argument files. The full collected-node plan
is retained, each file has one node per line, mutation is checked after execution,
and reported counts must still match the entire collection. No tests are omitted.
Argument-file support is part of the locked pytest version; the
[official pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html#specifying-which-tests-to-run)
documents it from version 8.2. A regression collects a parameterized node containing
spaces through an argument file whose path also contains spaces.

Ubuntu executed the 799-node collection: 792 passed, one failed and six
CUDA-dependent tests skipped. The failing whitespace-normalization test supplied
an incomplete synthetic license inventory on Linux. The existing policy correctly
requires the reviewed XGBoost/NCCL parent/dependency pair for that platform. The
fixture now explicitly tests Windows and Linux and includes both required Linux
packages. The release license policy and its missing-parent rejection are not
weakened. Focused portability regressions passed locally before the next full gate;
only a subsequent completed remote run can close the CI acceptance gap.

These changes affect test orchestration and fixtures, not the frozen `5a585e9`
experiment checkout, model parameters, data processing, seeds or predictions.
