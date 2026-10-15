# 29 September 2026 depth-6 training update
- Recorded the completed depth and temperature diagnostics.
- Froze research depth 6 and target temperature 0.05 before new training.
- Extended research paths to supervise every gate in the selected suffix while
  preserving the ordinary inference graph and parameter count.
- Added regression equivalence checks, a GPU cost gate, matched depth-6
  launchers and a comparison launcher.
- Added `WEEK4.md` with run order, resume constraints and stopping criteria.

# 25 September 2026 update
- New Windows setup/verification launchers for RTX 5060, Python 3.14 and CUDA 13.0 PyTorch.
- Offline extraction launcher and blank dataset folders.
- Downloader displays total size and percentage when supplied by the server.
- Added all-pair dataset verification with IDs, decode/dimension checks and hashes.
- Environment report now includes NumPy, Pillow and SciPy versions.
- Added exact pilot/overfit/evaluation launchers and updated research specification.
- Baseline model, training mathematics, metrics and resume behavior are unchanged.
- No datasets, pretrained weights or proposed error-supervised implementation included.

# 27 September 2026 research update
- Implemented shared-decoder forced branch reconstructions at the final block.
- Added auxiliary-only and error-supervised training variants.
- Added detached local branch-error targets, warm-up/ramp scheduling and diagnostics.
- Preserved the original parameter names and normal inference path.
- Extended resume tests to the error-supervised path.
- Added research compute benchmarking, matched 3,000-update launchers and a
  comparison tool that verifies configurations and dataset hashes.
- Synthetic degradation generation and publication-level evaluation remain pending.

# 27 September 2026 candidate-depth diagnostic update
- Recorded the matched Version 1 screen: error supervision did not beat baseline.
- Added forced paths through the last 1, 3 or 6 blocks without adding parameters.
- Added an FP32 checkpoint diagnostic for RGB and gradient error diversity,
  target entropy, oracle potential, preference margins and spatial stability.
- Extended model tests so depth one exactly reproduces the original candidates.
- Added launcher 10; no new training schedule is authorized by this update.
