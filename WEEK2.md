# Week 2: controlled scale-selection experiment

## Confirmed Week 1 evidence
The user reported all nine hundred DIV2K x4 pairs verified, all eight earlier
self-tests passing, and the 1,000-update baseline completing. On ten full DIV2K
development images, the pilot reported 29.4766 dB / 0.79723 versus PyTorch
bicubic 29.0796 dB / 0.78213. This is development-pilot evidence only.

## Experiment definitions
All variants use the same 586,077-parameter deployed model.

| Name | Main SR loss | Forced-branch SR losses | Error-derived gate loss |
|---|---:|---:|---:|
| baseline | yes | no | no |
| auxiliary | yes | yes | no |
| error_supervised | yes | yes | yes |

The auxiliary experiment is the central control. Comparing only the proposal
with baseline would confound gate supervision with extra deep supervision.
The candidate decoders share every parameter; they do not add deployed weights.

## Run order
1. Copy this update over the project while retaining data/runs/.venv5060.
2. Run `07_VERIFY_RESEARCH_UPDATE.cmd` and share its complete output.
3. If memory/timing is acceptable, run `08A_TRAIN_MATCHED_BASELINE_3K.cmd`.
4. Run `08B_TRAIN_AUXILIARY_CONTROL_3K.cmd`.
5. Run `08C_TRAIN_ERROR_SUPERVISED_3K.cmd`.
6. Run `09_COMPARE_RESEARCH_3K.cmd` and share its output/JSON.

The 08 launchers use two Windows data-loader workers. First-update time may be
high while workers spawn. Per-index random generation makes sample selection
independent of worker scheduling. Do not compare their results with the old
1,000-update run as if the budgets were matched.

## Resume commands
Use the exact applicable command plus `--resume`:

```bat
python train.py --experiment baseline --run runs/research_baseline_3k_s2026 --steps 3000 --validate-every 500 --workers 2 --gate-warmup 500 --gate-ramp 500 --resume
python train.py --experiment auxiliary --run runs/research_auxiliary_3k_s2026 --steps 3000 --validate-every 500 --workers 2 --gate-warmup 500 --gate-ramp 500 --resume
python train.py --experiment error_supervised --run runs/research_error_3k_s2026 --steps 3000 --validate-every 500 --workers 2 --gate-warmup 500 --gate-ramp 500 --resume
```

Resume from the most recent saved update; at most the unsaved interval repeats.
Do not change total steps or experimental settings when resuming.

## Decision criterion
Continue to larger studies only if error-supervised development performance
reproducibly exceeds the auxiliary-only control, diagnostics remain healthy,
and added training cost is acceptable. Inspect whether branch usage collapses,
whether gate/target agreement rises, and whether branch regret falls. No fixed
PSNR threshold is declared after seeing results; effect size, consistency, and
per-image behavior matter.

If the proposal is not better, preserve the negative result and diagnose it on
development data. Do not repeatedly tune against final benchmark sets.

## Completed outcome
The matched 3,000-update screen did not pass the decision criterion. Baseline
reached 28.98674 dB / 0.756286, auxiliary reached 28.97965 / 0.756137, and
error-supervised reached 28.98242 / 0.756208 on the frozen center-crop protocol.
The error-supervised result was 0.00432 dB below baseline. Its final target
entropy was essentially `ln(3)` and candidate errors were almost identical.
Version 1 is therefore a negative ablation; proceed to `WEEK3.md`, not a larger
training study.
