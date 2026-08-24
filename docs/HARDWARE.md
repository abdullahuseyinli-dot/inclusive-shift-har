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

All final neural fits and locked neural inference ran sequentially on CUDA with mixed precision where configured. No final neural CPU fallback was used. Recurrent cuDNN kernels repeatedly crashed on this Windows stack with exit `0xc0000409`; those failures are preserved under `results/failures/cuda_lstm_exit_c0000409/`. Successful recurrent runs kept model/data tensors on CUDA while setting `disable_cudnn=true`.

Random forest, SVM, and logistic regression use native scikit-learn CPU implementations by design. XGBoost training used its CUDA device path. These classical exceptions are not represented as neural CPU fallbacks.
