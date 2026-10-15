# RTX 5060 laptop: start here
Updated 27 September 2026. Project: Multi-Scale Attention Feature Fusion for SISR.

Extract the ZIP into `C:\`. Its top-level folder is `SISR_Computer-Vision`.
Open `C:\SISR_Computer-Vision` in VS Code. `train.py` should be directly inside it.
If you already have a project there, extract into a separate temporary folder first.
Copy the updated scripts and documentation into your existing project; retain your
`data`, `runs`, checkpoints, and previous package-lock files. Do not copy an old
virtual environment from another laptop.

This package contains the verified baseline plus controlled auxiliary and
error-supervised training paths. It excludes datasets, virtual environments,
checkpoints and pretrained weights. Synthetic degradation generation is not implemented.

## 1. Install on the new laptop
Double-click `01_SETUP_RTX5060.cmd`, or run its filename in Command Prompt.
It creates `.venv5060` using your Python 3.14, installs PyTorch 2.13.0 with CUDA
13.0, installs NumPy/Pillow/SciPy, and writes `requirements-lock-5060.txt`.
The setup needs internet access and downloads several GB. It stops on an error.
Use this only for the new laptop with driver 596.08; the old RTX 4060 driver
used a different CUDA build. No separate CUDA Toolkit installation is needed
for these prebuilt packages.

Official Windows Python 3.14 wheel checked on 25 September 2026:
https://download.pytorch.org/whl/cu130/torch/
Installation matrix: https://pytorch.org/get-started/previous-versions/

In VS Code, select interpreter `.venv5060\Scripts\python.exe`.
For manual commands in Command Prompt:
```bat
cd /d C:\SISR_Computer-Vision
.venv5060\Scripts\activate.bat
```
## 2. Verify and benchmark
Connect laptop power and run `02_VERIFY_RTX5060.cmd`.
Required for this updated package: CUDA available, nine tests pass without
skips, and the baseline benchmark completes.
Reports: `runs/environment_5060.json` and `runs/benchmark_5060.json`.
Send these reports and the test output before planning long runs.
The user's long RTX 5060 baseline benchmark measured about 0.145 seconds/update
for synthetic device-resident data. Real DIV2K training measured roughly
1.3-1.6 seconds/update with zero workers because PNG decoding was included.

## 3. Paste your datasets
Put the four completed ZIPs into:
`C:\SISR_Computer-Vision\data\archives`

- DIV2K_train_HR.zip
- DIV2K_train_LR_bicubic_X4.zip
- DIV2K_valid_HR.zip
- DIV2K_valid_LR_bicubic_X4.zip

Leave them zipped. Run `03_PREPARE_DATA_OFFLINE.cmd`.
It does not access the internet. It extracts local archives and checks every
pair. Existing extracted folders are also supported; see `data/README.md`.
Full verification decodes 1,800 images and hashes their bytes, so allow time.
Success: `Dataset verification passed: 800 training pairs and 100 validation pairs.`
If a ZIP is incomplete/corrupt, replace that ZIP with a complete download.
The checker validates IDs, readable files, sizes, and split separation; it
cannot prove correct pixel registration or authenticate source files.

## 4. Check that the model learns
Run `04_ONE_PATCH_CHECK.cmd`. It trains 200 updates on one fixed training patch.
Inspect loss reduction. Validation on a different image may not improve.
Send `runs/overfit_one_patch_5060/history.jsonl` for review.

## 5. Run the pilot
After the sanity check, run `05_TRAIN_PILOT.cmd`.
This is 1,000 updates, using 100 training images and 10 development images.
It is not full-dataset or converged publication training.
Checkpoints and example images go to `runs/baseline_x4_pilot_5060`.
The trainer rejects a nonempty run folder. To resume, activate the environment:
```bat
python train.py --run runs/baseline_x4_pilot_5060 --resume
```
Do not change total steps or training settings while resuming. Use a new run
name for a new experiment. Keep prior checkpoints and data unchanged.

## 6. Evaluate
Run `06_EVALUATE_PILOT.cmd`. This evaluates 10 full development images.
Full images can need more GPU memory than training crops. If memory runs out,
use an activated terminal and a NEW output folder:
```bat
python evaluate.py --checkpoint runs/baseline_x4_pilot_5060/best.pt --lr-dir data/DIV2K_valid_LR_bicubic/X4 --hr-dir data/DIV2K_valid_HR --limit 1 --device cpu --output runs/pilot_cpu_one_image_5060
```
This slower check is only one image; do not report it as a full benchmark.
Naive tiling changes this model because channel attention uses global pooling.

## 7. Verify the research update
After copying this update into the existing project while retaining `data`,
`runs`, and `.venv5060`, run `07_VERIFY_RESEARCH_UPDATE.cmd`. It runs the
expanded self-tests and benchmarks the heavier error-supervised training path.
Send its complete output before starting the matched runs. This verification
does not retrain or alter the completed Week 1 checkpoints.

The research method keeps normal inference unchanged. During training only,
the final block is also forced through each of its three branches, using the
same attention, projection and reconstruction decoder. `research.py` derives
local HR-error preference targets. `train.py` supports:

- `--experiment baseline`: original main reconstruction loss.
- `--experiment auxiliary`: main plus three branch reconstruction losses.
- `--experiment error_supervised`: matched auxiliary control plus gate loss.

## 8. Matched 3,000-update development experiment
Only after research verification, run 08A, 08B and 08C one at a time. Never run
them simultaneously on the same GPU. They use the same seed, images, crops,
architecture, optimizer, steps, development set and two data-loader workers.
Error supervision is zero through update 500, ramps to 0.05 by update 1,000,
then remains fixed. Auxiliary weight is 0.1 and temperature is 1.0.

When all three finish, run `09_COMPARE_RESEARCH_3K.cmd`. It refuses mismatched
configuration or dataset hashes. A single short development seed is a screening
experiment, not publication evidence. Do not choose a final claim from it alone.

To resume an interrupted research run, repeat its full train command with
`--resume`; the exact command is documented in `WEEK2.md`.

## 9. Version 1 result and Version 2 diagnostic
The completed matched screen reported 28.98674 dB for baseline, 28.97965 dB
for auxiliary-only and 28.98242 dB for error-supervised training. The proposed
variant therefore did not beat baseline. Its target entropy stayed almost at
the three-branch maximum `ln(3)`, and gate-target agreement stayed near chance.
This is candidate-error collapse: the candidate reconstructions were too
similar to define a useful routing target.

Those runs were not extended. Launcher 10 completed on the RTX 5060 and showed
that forcing each branch through all six blocks produced the strongest
candidate diversity and oracle potential. It did not train, overwrite a
checkpoint or modify data. `WEEK3.md` records its protocol and interpretation.

## 10. Frozen depth-6 update
The completed diagnostic showed that depth 6 increased candidate-output
diversity 5.51 times and RGB oracle improvement 6.25 times relative to depth
1. The temperature sweep selected 0.05: normalized RGB target entropy was
0.9204, compared with 0.9670 at 0.10 and 0.9924 at 0.25.

Install this update and run only:

```bat
11_VERIFY_DEPTH6_UPDATE.cmd
```

It runs the nine regression tests and benchmarks the training-only depth-6
graph. Send the complete output before starting launcher 12. Normal inference,
model parameters, existing data, results and checkpoints are unchanged. See
`WEEK4.md` for the frozen protocol and stopping rule.

## If setup or training fails
Stop and share the exact terminal output. For benchmark OOM, activate the venv:
`python benchmark.py --micro-batch 2 --accumulate 8 --output runs/benchmark_5060_b2.json`
If this setting is needed for training, use a fresh run:
`python train.py --run runs/baseline_x4_pilot_5060_b2 --micro-batch 2 --accumulate 8`
The default evaluation launcher targets the default pilot name; adjust the
checkpoint path when using a different run.
