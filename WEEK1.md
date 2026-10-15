# Week 1 milestones — updated for the new laptop
There is no need to wait a calendar day between successful milestones.
Follow START_HERE.md for exact commands and current paths.

| Milestone | Action | Evidence |
|---|---|---|
| Environment | Run 01_SETUP and 02_VERIFY | CUDA ready, all eight tests pass, new benchmark |
| Data | Paste ZIPs; run 03_PREPARE_DATA_OFFLINE | 800 training and 100 validation pairs checked |
| Learning sanity | Run 04_ONE_PATCH_CHECK | Training loss decreases |
| Baseline pilot | Run 05_TRAIN_PILOT | Loss/history, checkpoint and image outputs |
| Evaluation | Run 06_EVALUATE_PILOT | Full-image metrics and matching visual crops |
| Literature | Fill literature_notes.md | Exact prior-art distinctions |
| Review | Compare outputs and convergence | Decide next controlled experiment |

The original eight tests passed on the previous RTX 4060 laptop; verification
on the RTX 5060/Python 3.14 environment is still required.
Training uses official DIV2K bicubic x4 pairs. The research extension is not
part of the baseline pilot. Validation is development data, not an untouched test.
