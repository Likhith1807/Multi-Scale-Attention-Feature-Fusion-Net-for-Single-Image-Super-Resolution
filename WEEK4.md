# Week 4: depth-6 matched mechanism screen

## Frozen decision
The no-training sweep selected forced-path depth 6. Compared with depth 1,
depth 6 increased pairwise candidate-output L1 by 5.51 times and RGB oracle
improvement by 6.25 times. Temperature 0.05 was selected before new training:
its normalized RGB target entropy was 0.9204, versus 0.9670 at 0.10 and 0.9924
at 0.25. This choice is now frozen for the screen.

This is mechanism-selection evidence, not proof of improved super-resolution.
Normal inference and the 586,077 deployed parameters remain unchanged.

## Run gate
First run only:

```bat
11_VERIFY_DEPTH6_UPDATE.cmd
```

It executes all tests and measures the depth-6 training-only graph on the RTX
5060. Share its complete output before running either 12 launcher. An OOM,
non-finite value, failing regression, or unreasonable timing stops the study.

## Matched training order
After the cost gate is reviewed, run one at a time:

```bat
12A_TRAIN_DEPTH6_AUXILIARY_3K.cmd
12B_TRAIN_DEPTH6_ERROR_3K.cmd
13_COMPARE_DEPTH6_3K.cmd
```

The existing matched baseline is reused because depth and temperature affect
only training-only research paths. Auxiliary and error-supervised runs share
depth 6, temperature 0.05, data, seed, sampling, optimizer, schedule and budget.

Resume commands append `--resume` to the exact corresponding `train.py`
command in the launcher. Do not change batch, steps, workers or loss settings.

## Decision rule
Proceed to multi-seed and external-benchmark work only if error supervision
beats both the depth-6 auxiliary control and the existing matched baseline by
a practically meaningful margin, with finite and non-collapsed diagnostics.
A single 3,000-update development screen cannot establish superiority.
