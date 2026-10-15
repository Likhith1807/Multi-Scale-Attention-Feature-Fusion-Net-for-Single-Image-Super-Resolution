#!/usr/bin/env bash
# Train 4x SISR Baseline model
set -euo pipefail

RUN_DIR=${1:-"runs/baseline_x4_linux"}
STEPS=${2:-3000}
BATCH_SIZE=${3:-16}
DEVICE=${4:-"cuda"}

echo "Training MSAF-Net 4x Baseline: run_dir=$RUN_DIR, steps=$STEPS, batch_size=$BATCH_SIZE, device=$DEVICE"

python train.py \
  --run-dir "$RUN_DIR" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --crop-size 48 \
  --scale 4 \
  --device "$DEVICE"
