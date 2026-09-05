# Observed HAR-PMD transport equivalence

This is a reporting qualification, not a change to the frozen model, sampling,
window, split, seed, or statistical protocol. It permits one narrowly evidenced
comparison across remote ZIP access and a verified local copy of the same public
provider archive. It does not permit changed data or preprocessing to be described
as an improvement.

The retained clean-source audit at
`.audit/phase2_external_v3/pmd_all_participant_transport_parity_ecafde4/parity.json`
has record SHA-256
`eabc24a957a423b79adc959663352b1e59e414c8ab1051d14b77d7cf87253a46`.
It was executed from clean commit
`ecafde4c2712d01c6f59435ef5537787f91b6946`, without model fitting, annotation
mutation, or performance scoring. All 120 participants, 1,200 source members,
and 248,496 windows were compared. Signal, gravity, label, participant, session,
trial, and window arrays matched bitwise between the preserved old loader and
the verified-copy loader. To bound memory use, the probe restricted the ZIP
directory listing to one participant at a time, exhaustively; member bytes and
the ZIP reading interface were unchanged.

The archive is provider record 7939223, version 2, under CC BY 4.0:
[HAR-PMD provider record](https://zenodo.org/records/7939223).
Its SHA-256 is
`3dd04d8239540ebc1eb50c569f0c421ce54875b6b14a69dddc6f3459b6f0d665`,
its MD5 is `cdc7b79aaa0450d68d94f430f2a2de63`, and its size is
7,288,182,191 bytes. These are verified provider bytes, not synthetic or imputed
measurements. The raw archive remains outside Git.

## Acceptance for a reconstructed table

The default table builder still requires identical preprocessing source files
for HAR-PMD. An explicit `--transport-parity PATH` enables the witnessed exception
only after checking all of the following:

- Witness self-hash, pinned probe hash, complete population, clean Git state,
  source-manifest-to-commit binding, old committed loader hash, and provider-copy
  size/digest/license contract.
- Every actual run's exact source-member hashes, full participant/window counts,
  class support, loader hash, and prediction identity arrays against the witness.
- The remaining dataset metadata, ontology, seeds, inference endpoint, protocol,
  scoring grid, and metric contract across the runs.

Only `raw_local_mirror` and `source_storage_audit` are normalized for this
comparison. Their original values, both source hashes, and the witness reference
remain in the resulting table. Tests reject changed labels, ordering, dtypes,
source members, cohorts, loader hashes, and sampling metadata. A supplied witness
does not bypass independent validation of each result package.

This establishes equivalence on the observed full HAR-PMD population only. It
does not prove that arbitrary loader revisions are interchangeable, make native
gravity equivalent to derived gravity, or erase a failed or manually stopped
enclosing campaign. Six-channel and native-nine-channel comparisons remain
separate representation lanes, with per-method evidence status and runtime
qualifications retained.
