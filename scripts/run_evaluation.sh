#!/usr/bin/env bash
# Run the full evaluation pipeline: baselines + DM tests + plots.
#
# Prerequisites:
#   1. Trained linear probe: outputs/probe/linear_probe.pth
#   2. Test embeddings: outputs/probe/test_embeddings.npy
#   3. GARCH model (fitted in evaluation script)
#
# Usage: bash scripts/run_evaluation.sh

set -euo pipefail

CONFIG="configs/probe_config.yaml"
OUTPUT_DIR="outputs/evaluation"

mkdir -p "$OUTPUT_DIR"

echo "Running full evaluation pipeline..."
python3 -m src.evaluation.run_full_evaluation \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"

echo "Evaluation complete. Results in: $OUTPUT_DIR"
