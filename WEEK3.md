# Week 3: candidate-depth diagnosis before Version 2

## Why training is paused
The matched 3,000-update screen was valid, but final-block error supervision
did not beat baseline. During the final 500 updates, the three candidate losses
differed by only about 0.000006 on average. Target entropy was almost exactly
`ln(3)`, so the supervision taught the gate to predict equal weights rather
than a meaningful spatial preference.

This identifies candidate-error collapse. It does not prove that internal
features are identical; only their decoded errors under the Version 1
candidate construction were indistinguishable.

## Diagnostic question
Does forcing the same dilation branch through more consecutive blocks create
candidate outputs whose reconstruction errors differ meaningfully?

The existing matched baseline checkpoint is evaluated at depths:

| Depth | Candidate definition |
|---:|---|
| 1 | Force a branch only in the final block; reproduces Version 1 |
| 3 | Force the same branch through the final three blocks |
| 6 | Force the same branch through all six blocks |

All candidates reuse existing parameters and the same decoder. Normal inference
is unchanged. The diagnostic runs in FP32 on the same ten 128x128 LR center
crops and does not update weights.

## Metrics
- Pairwise candidate-output L1: whether outputs differ at all.
- RGB and gradient branch-error spread: whether those differences matter
  against HR targets.
- Target entropy and preference margins: whether a routing target is informative.
- Oracle improvement: the upper-bound local error reduction available from
  choosing the best individual candidate with HR knowledge.
- Winner usage and neighbor agreement: whether preferences use multiple
  branches and have spatial coherence.

Large output differences alone are not success. They must produce lower oracle
error and stable, nonuniform preferences. These are development diagnostics,
not benchmark results or a novelty claim.

## Run
Retain all existing `data`, `runs`, `.venv5060` and checkpoints. Copy the update
over the project, then run:

```bat
10_DIAGNOSE_CANDIDATE_DEPTH.cmd
```

Upload `runs/candidate_depth_diagnostic_5060.json`. Do not start new training.
If deeper paths remain uninformative, the next controlled pivot is explicit
frequency specialization rather than a lower temperature or larger gate loss.
