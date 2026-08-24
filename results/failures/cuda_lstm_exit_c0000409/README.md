# CUDA BiLSTM native-process failure

Status: **failed and quarantined; excluded from tuning, aggregation, model
selection, calibration, and confirmatory evaluation**.

Three deterministic source-only `legacy_bilstm_h192` development executions
completed their Python-level training, checkpoint, prediction, and summary
writes, but `python.exe` then terminated natively with Windows exception
`0xc0000409` in `ucrtbase.dll` (`BEX64`). Two identical learning-rate/weight-
decay attempts reproduced byte-identical selected checkpoint SHA-256
`fdc191ca994b234fb04f8f5e4c8628b8d1d210a087069fd462869ad365832878`.
Because the process did not exit successfully, none of the emitted metrics is
accepted as a completed trial.

A subsequent CUDA fold attempt for the original jointly trained
`legacy_joint_bilstm256_cnn128` reproduced the same native exit code and is
quarantined here as a fourth failed attempt.

An exact `deepconvlstm` CUDA development attempt under commit
`fb99510c3606498b90d8126293a7ef1f4871e07d` also completed its Python-level
artifact writes and then exited with the same `0xc0000409` native failure. Its
summary is preserved here as the fifth failed attempt. The successful
DeepConvLSTM development and grouped-CV records use CUDA with cuDNN disabled
explicitly in the hashed configuration.

Windows Application Error/WER event IDs 1000/1001 recorded report IDs
`03ec83ca-1707-45ff-85b8-929ead104f7e`,
`a20174c0-d048-49b3-b911-762b68402152`, and
`1f04629a-2947-4bbc-bd51-6b70e0a8ec8b`. The failure occurred with PyTorch
2.12.0+cu132 on the RTX PRO 3000 Blackwell Laptop GPU under commits
`7f873f300c6533f7e9b00408ef197ea6ed51e5c2` and
`fb99510c3606498b90d8126293a7ef1f4871e07d` while another independent local
GPU process was visible. The benchmark does not attribute causality to that
contention.

The generated summaries were moved here without deletion. Their ignored
run/checkpoint/prediction directories remain under
`results/runs/source-dev-v2/`. CPU fallback trials use new immutable run IDs.
No target subject, signal, label, prediction, or performance was accessed.

A bounded two-epoch diagnostic exited successfully when cuDNN was disabled
while tensors remained on CUDA. That diagnostic is preserved separately under
`results/diagnostics/cuda_recurrent_backend/`; the backend choice is now an
explicit hashed training configuration field rather than an unrecorded runtime
toggle.
