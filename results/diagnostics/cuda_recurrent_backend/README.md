# CUDA recurrent-backend diagnostic

This is bounded engineering evidence, not a benchmark trial or model result.

After cuDNN-backed bidirectional LSTM jobs repeatedly terminated Python with
Windows exception `0xc0000409`, a two-epoch source-only diagnostic kept model
tensors and optimization on CUDA but disabled cuDNN. It exited normally and
wrote summary SHA-256
`d057bd3e39521c6a4f4d280d194fb8a31985183d3a350d1aadfc2b39df00b463`.

The diagnostic used an uncommitted runtime toggle and is excluded from tuning
and evaluation. Its only accepted conclusion is that a recorded non-cuDNN CUDA
path is technically viable. The follow-up implementation adds
`disable_cudnn` to the hashed `TrainingConfig` and checkpoint environment.
No target data or performance was accessed.
