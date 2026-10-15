# Validation — 29 September 2026

Completed for this research-update package in the preparation environment:
- Parsed all Python files and exercised the command-line entry points.
- The suite discovered nine tests: six data/metric/ZIP tests passed and three
  PyTorch model/research/resume tests were skipped because PyTorch is
  unavailable in this Linux preparation environment.
- Offline preparation rejected missing input without network access.
- Offline preparation extracted a small local ZIP without network access.
- New all-pair checker passed on 900 generated synthetic pairs with official
  ID patterns, and rejected a dimension-mismatched pair.
- ZIP contents/integrity checked; no datasets, environments or checkpoints included.

User-supplied evidence from the RTX 5060 laptop before the research update:
- Environment: Python 3.14.4, torch 2.13.0+cu130, CUDA available, 7.96 GiB GPU.
- All eight baseline-package self-tests passed.
- CUDA synthetic baseline benchmark: 586,077 parameters and 0.144642 seconds
  per optimizer update over 100 timed updates (microbatch 4, accumulation 4,
  LR patch 48, AMP). This excludes decoding, validation and checkpointing.
- DIV2K x4 verification passed for 800 training and 100 validation pairs.
- The one-patch sanity check reduced L1 from 0.047938 to 0.025985.
- The 1,000-update pilot reached development-crop Y-PSNR 28.6797 and Y-SSIM
  0.74728. Full-image evaluation on its ten-image development subset produced
  Y-PSNR 29.476624 and Y-SSIM 0.797229, versus the script's PyTorch bicubic
  reference at 29.079631 and 0.782134.

Pending for this research update on the RTX 5060 laptop:
- Run `07_VERIFY_RESEARCH_UPDATE.cmd`. It should report nine tests, including
  the new research-gradient test and baseline/error-supervised resume tests.
- Measure the training-only research path; its three candidate reconstructions
  make it slower and more memory-intensive than normal inference.
- Complete the matched baseline, auxiliary-control and error-supervised pilots.
- Synthetic degradation, multi-seed confirmation, benchmark-set evaluation,
  complexity reporting and statistical analysis remain future work.

User-supplied matched Version 1 evidence:
- Baseline: 28.986740 dB / 0.756286; auxiliary: 28.979654 / 0.756137;
  error-supervised: 28.982423 / 0.756208 on ten center-cropped development images.
- Dataset hashes and material run configuration matched across all three runs.
- In the final 500 updates, error-supervised target entropy was 1.098586 versus
  the three-branch maximum 1.098612, gate usage was approximately one third per
  branch, and agreement was 33.56 percent. Version 1 did not establish benefit.

Completed depth-selection evidence supplied on 29 September 2026:
- All nine tests passed twice on the RTX 5060.
- Depth 6 produced 5.51 times the pairwise output difference and 6.25 times
  the RGB oracle improvement of depth 1 on the frozen ten-image crop protocol.
- Temperature 0.05 produced normalized target entropy 0.9204 for RGB and
  0.9348 for gradients; 0.10 and 0.25 remained softer.

Pending for the depth-6 training update:
- RTX 5060 forward/backward regression and 100-update synthetic cost gate.
- Matched auxiliary and error-supervised 3,000-update screen.
- Evidence that a learned gate captures oracle potential and improves PSNR or
  SSIM; the diagnostic alone does not establish this.

Synthetic test images are temporary verification inputs and are not packaged as
training data. No result from the new error-supervised method is claimed yet.
