#!/usr/bin/env bash
# =====================================================================
# Run end-to-end training pipeline (with optional model selection)
# Usage:
#   bash scripts/run_train.sh all
#   bash scripts/run_train.sh textcnn
# =====================================================================

set -e  # stop on error
MODEL=${1:-all}

echo "🚀 Starting training for model: $MODEL"

# 1️⃣ Activate environment (optional)
source ~/.bashrc 2>/dev/null || true
conda activate webmd_env 2>/dev/null || source venv/bin/activate 2>/dev/null || true

# 2️⃣ Run training
python -m pipeline.train_nn --model "$MODEL" --input data/raw/webmd.csv --processed-dir data/processed

echo "✅ Training complete. Results in reports/ and experiments/"
