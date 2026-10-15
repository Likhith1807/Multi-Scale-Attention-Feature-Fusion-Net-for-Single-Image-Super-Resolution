# Multi-Scale Attention Feature Fusion Network for Single-Image Super-Resolution (MSAF-Net)

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.13.0+cu130](https://img.shields.io/badge/pytorch-2.13.0%2Bcu130-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tested on RTX 5060](https://img.shields.io/badge/GPU-RTX%205060%20(8GB)-green.svg)](https://www.nvidia.com/)

A lightweight, high-performance deep neural network architecture for $4\times$ Single-Image Super-Resolution (SISR) built with PyTorch. Features an original multi-scale dilated attention baseline, paired DIV2K data pipeline, strict Y-channel PSNR/SSIM evaluation, and controlled research paths for error-supervised scale selection.

---

## Architecture Overview

The core architecture (`CompactMSAF` in `model.py`) contains **586,077 parameters** in its default $4\times$ learned-fusion configuration:

```
Low-Resolution (LR) Input [3 x 48 x 48]
       │
       ▼
┌──────────────────────────────────────────────┐
│  Shallow Feature Extraction (Stem Conv 3x3)  │  --> 48 channels
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Trunk: 6 × MultiScaleBlocks                 │
│  ├─ Dilated Convolutions (d = 1, 2, 3)       │
│  ├─ Spatial Soft Attention Gate              │
│  ├─ Squeeze-and-Excitation Channel Attention │
│  └─ Residual Scaling (0.2)                   │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Body Conv 3x3 + Long Skip (Stem addition)   │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Upsampler (2 × PixelShuffle x2 + LeakyReLU) │  --> 48 x 192 x 192
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  RGB Head (3x3 Conv) + Global Bicubic Skip   │
└──────────────────────────────────────────────┘
       │
       ▼
Super-Resolved (SR) Output [3 x 192 x 192]
```

### Key Highlights
- **Multi-Scale Spatial Attention:** Three parallel $3\times3$ convolutions with dilation factors $d \in \{1, 2, 3\}$ giving $3\times3$, $5\times5$, and $7\times7$ receptive fields with zero parameter bloat.
- **Dynamic Soft Gating:** Feature-dependent $1\times1$ convolution producing position-specific softmax weights over spatial dilation branches.
- **Squeeze-and-Excitation Channel Attention:** Adaptive channel-wise recalibration following spatial feature fusion.
- **Bicubic Residual Learning:** High-frequency detail learning via explicit addition of PyTorch bicubic upsampling.
- **Controlled Research Extensions:** Implemented auxiliary-only and detached local reconstruction error targets for studying branch routing and degradation shifts.

---

## Experimental & Hardware Specifications

- **Verified Environment:** Windows 11, Python 3.14.4, PyTorch 2.13.0+cu130, CUDA 13.0.
- **Target GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (~8 GB VRAM, Driver 596.08).
- **Core Dependencies:** PyTorch, NumPy ($\ge 1.24, < 3$), Pillow ($\ge 10, < 13$), SciPy ($\ge 1.10, < 2$).
- **Benchmark Dataset:** DIV2K (800 training pairs, 100 validation pairs at $4\times$ bicubic downscaling).

---

## Project Structure & Launchers

The project includes an end-to-end suite of Windows batch launchers (`.cmd`) and Python CLI scripts:

| Launcher / Script | Description |
|---|---|
| `01_SETUP_RTX5060.cmd` | Automated setup: creates `.venv5060`, installs PyTorch CUDA 13.0, and dependencies |
| `02_VERIFY_RTX5060.cmd` | Verifies hardware, runs unit tests, and measures CUDA baseline throughput |
| `03_PREPARE_DATA_OFFLINE.cmd` | Extracts offline DIV2K archives and validates all 900 image pairs |
| `04_ONE_PATCH_CHECK.cmd` | Rapid sanity check verifying model capacity to overfit a single patch |
| `05_TRAIN_PILOT.cmd` | 1,000-update baseline pilot training |
| `06_EVALUATE_PILOT.cmd` | Evaluates checkpoint against 10 full DIV2K validation images |
| `07_VERIFY_RESEARCH_UPDATE.cmd` | Regression checks and research training benchmark |
| `08A_TRAIN_MATCHED_BASELINE_3K.cmd` | 3,000-update matched baseline experiment |
| `08B_TRAIN_AUXILIARY_CONTROL_3K.cmd` | 3,000-update auxiliary-only control experiment |
| `08C_TRAIN_ERROR_SUPERVISED_3K.cmd` | 3,000-update error-supervised research experiment |
| `09_COMPARE_RESEARCH_3K.cmd` | Cross-validates configurations, dataset hashes, and compares metrics |
| `10_DIAGNOSE_CANDIDATE_DEPTH.cmd` | Evaluates forced routing through 1, 3, and 6 blocks without retraining |
| `11_VERIFY_DEPTH6_UPDATE.cmd` | Depth-6 regression tests and RTX 5060 memory cost gate |
| `12A_TRAIN_DEPTH6_AUXILIARY_3K.cmd` | Matched depth-6 auxiliary screen (3,000 updates) |
| `12B_TRAIN_DEPTH6_ERROR_3K.cmd` | Matched depth-6 error-supervised screen (3,000 updates) |
| `13_COMPARE_DEPTH6_3K.cmd` | Verifies and compares completed depth-6 screens |
| `14_EXPORT_PRESENTATION_RESULTS.cmd` | Generates qualitative zoom comparisons and training curves |
| `model.py` | PyTorch implementation of `CompactMSAF` network |
| `train.py` / `evaluate.py` | Complete training engine, checkpointing, and Y-channel PSNR/SSIM evaluation |
| `metrics.py` | High-precision YCbCr conversion, PSNR, and SSIM metrics with border cropping |
| `pairs.py` | DIV2K image pairing, crop extraction, random augmentations, and hash checks |
| `research.py` | Detached error loss targets, loss schedules, and gating diagnostic logs |
| `self_test.py` | Full unit test suite (metrics, crops, gradients, forward/backward, checkpoint resume) |

---

## Quickstart Guide

### 1. Installation
Clone the repository and run the setup script:

```bash
git clone https://github.com/Likhith1807/Multi-Scale-Attention-Feature-Fusion-Net-for-Single-Image-Super-Resolution.git
cd Multi-Scale-Attention-Feature-Fusion-Net-for-Single-Image-Super-Resolution
```

On Windows, double-click or run:
```bat
01_SETUP_RTX5060.cmd
```

Or manually create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pillow scipy
# Install PyTorch with CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 2. Verification
Execute the test suite to ensure numeric precision and GPU compatibility:
```bash
python self_test.py
python check_environment.py
```

### 3. Data Preparation
Place the official DIV2K zip archives in `data/archives/`:
- `DIV2K_train_HR.zip`
- `DIV2K_train_LR_bicubic_X4.zip`
- `DIV2K_valid_HR.zip`
- `DIV2K_valid_LR_bicubic_X4.zip`

Extract and verify image pairs:
```bat
03_PREPARE_DATA_OFFLINE.cmd
```
*(Or run `python verify_data.py --data data`)*

### 4. Training
To train the baseline model:
```bash
python train.py --run-dir runs/baseline_run --steps 3000 --batch-size 16 --crop-size 48 --scale 4 --device cuda
```

### 5. Evaluation
Evaluate a trained model checkpoint on the full DIV2K validation set:
```bash
python evaluate.py --checkpoint runs/baseline_run/checkpoint_step_003000.pt --data data --scale 4 --output runs/eval_results.json
```

---

## Research Documentation
- Detailed mathematical formulation and ablation design: [`docs/ARCHITECTURE_AND_METHOD.md`](docs/ARCHITECTURE_AND_METHOD.md)
- Step-by-step experimentation guidelines: [`START_HERE.md`](START_HERE.md)
- Experimental roadmap and validation record: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md), [`VALIDATION.md`](VALIDATION.md)
- Development milestone logs: [`WEEK1.md`](WEEK1.md), [`WEEK2.md`](WEEK2.md), [`WEEK3.md`](WEEK3.md), [`WEEK4.md`](WEEK4.md)

---

## License

This project is licensed under the [MIT License](LICENSE).
