#!/usr/bin/env bash
# =====================================================================
# Run preprocessing pipeline to generate processed datasets
# =====================================================================

set -e
INPUT=${1:-data/raw/webmd.csv}
OUTPUT_DIR=${2:-data/processed}

echo "🧹 Running data preprocessing pipeline..."
python -m src.data.make_dataset --input "$INPUT" --output-dir "$OUTPUT_DIR"

echo "✅ Processed files saved under $OUTPUT_DIR/"
