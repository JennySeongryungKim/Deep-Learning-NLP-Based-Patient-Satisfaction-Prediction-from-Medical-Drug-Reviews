#!/usr/bin/env bash
# =====================================================================
# Download WebMD Drug Reviews dataset from Kaggle
# Prerequisites: kaggle CLI (pip install kaggle) and API key (~/.kaggle/kaggle.json)
# =====================================================================

set -e
DATA_DIR="data/raw"
mkdir -p $DATA_DIR

echo "⬇️ Downloading WebMD dataset from Kaggle..."
kaggle datasets download -d rohanharode07/webmd-drug-reviews-dataset -p $DATA_DIR

echo "📦 Extracting..."
unzip -o $DATA_DIR/webmd-drug-reviews-dataset.zip -d $DATA_DIR

echo "✅ Data ready at $DATA_DIR/webmd.csv"
