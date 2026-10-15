#!/usr/bin/env bash
# Evaluate trained checkpoint on DIV2K validation set
set -euo pipefail

CHECKPOINT=${1:-"runs/baseline_x4_pilot_5060/best.pt"}
DATA_DIR=${2:-"data"}
OUTPUT=${3:-"runs/eval_results.json"}

echo "Evaluating checkpoint: $CHECKPOINT on $DATA_DIR"

python evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_DIR" \
  --scale 4 \
  --output "$OUTPUT"
