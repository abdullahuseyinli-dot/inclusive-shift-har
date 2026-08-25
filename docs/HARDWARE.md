# Hardware and CUDA environment

The confirmatory machine was observed read-only on 2026-08-24. The canonical privacy-safe record is `results/environment/confirmatory_machine_20260824.json` with record SHA-256 `8b73d8dfa5db995a218dd3a1a7174d833496cb5b11434f5ebfdfb02c0c05ea7e`.

| Component | Observed value |
|---|---|
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| GPU memory | 12,227 MiB |
| Compute capability | 12.0 |
| NVIDIA driver | 596.72 |
| Driver CUDA compatibility | 13.2 |
| PyTorch | 2.12.0+cu132 |
| PyTorch CUDA runtime | 13.2 |
| cuDNN | 92000 |
| CPU | Intel Core Ultra 9 285H, 16 logical processors |
| System memory | 68,137,205,760 bytes |
| OS | Windows 11 Pro 10.0.26200, build 26200 |
| Python | CPython 3.11.9 |

`torch.cuda.is_available()` was true, and an actual CUDA tensor allocation and reduction returned the expected value. This is runtime evidence, not merely a driver capability report.

All completed reported neural executions in this snapshot—including the locked primary suite, corrected UCI reproduction, disabled within-group evaluation, few-person curve, sensor stress, qualified CCIL/BPD adaptations, and raw/total-acceleration sensitivity—used CUDA with mixed precision where configured. No reported final neural CPU fallback was used. Recurrent cuDNN kernels repeatedly crashed on this Windows stack with exit `0xc0000409`; those failures are preserved under `results/failures/cuda_lstm_exit_c0000409/`. Successful recurrent runs kept model/data tensors on CUDA while setting `disable_cudnn=true`.

Random forest, SVM, and logistic regression use native scikit-learn CPU implementations by design. XGBoost training used its CUDA device path. These classical exceptions are not represented as neural CPU fallbacks.

The post-confirmatory efficiency operation validated 80 checkpoints and 320 CUDA profiles with 641 sampled contention snapshots. Its timing status is `valid_with_declared_allowlisted_ambient_system_processes`; `dwm.exe` and `explorer.exe` were the exact predeclared WDDM ambient processes, and no unapproved compute process was sampled. Batch-1 FP32 mean device-resident forward latency was 1.3561 ms for compact ERM, 1.4985 ms for compact DANN, 2.0207 ms for MoRe-HAR full, 11.7601 ms for DeepConvLSTM, and 19.9391 ms for the legacy joint CNN/BiLSTM. FP16 autocast was slower for all 16 models at both batch sizes, while reducing peak allocated VRAM in 30 of 32 model-batch comparisons; this is a memory tradeoff rather than an acceleration result. The 60 recurrent profiles used CUDA with cuDNN disabled. These values exclude host-to-device transfer and are not full-graph, end-to-end, mobile-device, or portable latency claims.
