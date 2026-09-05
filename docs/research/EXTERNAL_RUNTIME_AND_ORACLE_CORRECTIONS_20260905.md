# External execution and oracle-schema corrections

These corrections were declared before replacement outcomes. They do not
change FoG session-grid-v3, participant partitions, seeds, model sizes,
optimization budgets, metrics, or evidence roles. Earlier campaigns and all
failed diagnostics remain immutable.

## External neural backend

The corrected FoG campaign, its fixed duplicate, and complete-case IMU campaign
wrote reconstructable result packages and then terminated with Windows native
exit `0xC0000409`. Import, allocation, and TinyHAR-style probes exited normally.
A standalone synthetic CUDA LSTM forward reproduced the failure without project
data or model code. Model construction alone, CPU execution, and CUDA execution
without cuDNN exited normally. Repeating under a separately installed non-Store
CPython did not remove the cuDNN failure. This isolates a triggering backend
path; it does not establish the underlying native-library defect.

The repository already documented this failure and a successful CUDA/no-cuDNN
policy in `docs/HARDWARE.md` and retained older failure artifacts. The external
runner had omitted that setting. The new external protocol explicitly sets
`disable_cudnn=true` for DeepConvLSTM folds, keeps model/data tensors on the
selected device, and restores the prior backend state after success or failure.
TinyHAR-style retains its inherited cuDNN policy. The setting travels with
configuration, checkpoint, training record and result. No exit code is masked.

The effective declaration is
`configs/protocols/external_neural_cuda_nocudnn_v2.json`. Its v1 predecessor is
retained: a pre-outcome amendment corrected the mistaken phrase “FP32 precision”
to the actual inherited FP16 autocast/GradScaler policy with FP32 master
parameters. No precision change is introduced. Backend changes can alter dropout
and floating-point execution; replacements are never selected by whichever
backend scores highest, and bitwise equivalence is not assumed.

The old numerical packages remain explicitly qualified by their failed process
termination. A fresh validation exit zero does not rewrite that failure. Nested
primary-suite reports must carry notices from their enclosing campaign, not
only their immediate parent directory.

Suite-level progress summaries now read the seed-averaged participant metric,
not the secondary probability-ensemble report. Original result packages already
contain both estimands; older suite progress summaries are not substituted for
the primary machine-readable statistic.

## Sole-HARmony empty unknown markers

The full 12-participant/two-session oracle campaign stopped at
`C002/C002_1664294172/DataStruct.mat`. A source-receipted diagnostic found six
finite zero-duration camera rows with label `-1` among 181 annotations, and no
overlap between positive-duration intervals. Such rows cover no sensor samples.

`configs/protocols/sole_harmony_zero_duration_unknown_v1.json` permits only those
finite, exactly zero-duration unknown markers to be omitted, with exact
timestamps, stable sorted source indices and zero excluded duration recorded.
Inverted or non-finite intervals, nonintegral codes, positive-duration overlap,
and malformed supported intervals still fail visibly. Positive-duration bouts
are not relabeled, shifted, joined, or selected by model outcomes.

Retry the same ordered 12 participants and two sessions, with all three frozen
seeds. This remains a camera-bout oracle diagnostic; it is not a deployable
streaming temporal model. A successful schema correction cannot turn
annotation-defined reset boundaries into inference-observable boundaries.

## Preservation and clean-clone reporting

The first long-path clone failed checkout and is retained. A new clone succeeded
with Git long paths enabled. That option also needs to be stored in the clone's
local Git configuration for subsequent status checks; no global setting was
changed. Tables generated before a clean launch-state check are diagnostic
reconstructions, not the final clean-clone evidence. Final reconstruction uses
new output directories and checks clean Git state explicitly.
