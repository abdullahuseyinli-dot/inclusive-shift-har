# Cancelled CPU recurrent-fold loop

Status: **cancelled by user direction; excluded from grouped aggregation,
selection, calibration, and confirmatory evaluation**.

After the cuDNN-backed CUDA failures, one `legacy_bilstm_h192` source fold
completed on CPU. During the next fold the user directed the project to avoid
further CPU training and use CUDA. The running loop was interrupted immediately.
The completed fold summary is preserved here with file SHA-256
`14a454a2bc00fc2aecc6a193907727c023bc04d5d0ec0d1cbdb6a2b855e85ca2`;
the second fold's empty ignored run directory is retained. An incomplete fold
set is not scientific performance evidence.

Work continued with an explicit non-cuDNN CUDA recurrent backend. No target
subject, signal, label, prediction, or performance was accessed.
