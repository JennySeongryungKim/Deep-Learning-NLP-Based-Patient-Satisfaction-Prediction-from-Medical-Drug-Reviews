# -*- coding: utf-8 -*-
from pathlib import Path

# Project root = this file → ../../
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"
LOGS = ARTIFACTS / "logs"
EDA_OUT = ARTIFACTS / "EDA_output"

DOCS = ROOT / "docs"
NOTEBOOKS = ROOT / "notebooks"
SRC = ROOT / "src"

# Ensure common dirs exist
for p in [DATA, RAW, EXTERNAL, INTERIM, PROCESSED, ARTIFACTS, FIGURES, MODELS, LOGS, EDA_OUT]:
    p.mkdir(parents=True, exist_ok=True)
