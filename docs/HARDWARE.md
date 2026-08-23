# Hardware and CUDA environment

This is a privacy-safe, read-only observation of the current development machine on 2026-08-23. The machine-readable record is `results/environment/hardware_20260823.json`. Hostname, username, filesystem paths, and hardware serial identifiers are intentionally excluded.

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| GPU memory | 12,227 MiB (approximately 11.94 GiB) |
| NVIDIA driver | 596.72 |
| Driver-reported CUDA compatibility | 13.2 |
| CPU | Intel(R) Core(TM) Ultra 9 285H; 16 reported cores and 16 logical processors |
| System memory | 68,137,205,760 bytes (approximately 63.458 GiB) |
| Operating system | Microsoft Windows 11 Pro, version/build 10.0.26200/26200, 64-bit |
| Project Python | CPython 3.11.9, AMD64 |
| uv | 0.11.29 |

## CUDA interpretation

`nvidia-smi` detected the GPU and reports CUDA 13.2 compatibility for the installed driver. This is a driver capability statement, not evidence that a matching CUDA toolkit or a usable framework runtime is installed. `nvcc` was not found on `PATH`.

The project virtual environment currently has no PyTorch installation. Consequently, no `torch.cuda.is_available()` check, CUDA tensor allocation, mixed-precision probe, VRAM stress test, or training run has been performed. CUDA runtime usability remains **not tested** until project dependencies are installed through the approved environment workflow.
