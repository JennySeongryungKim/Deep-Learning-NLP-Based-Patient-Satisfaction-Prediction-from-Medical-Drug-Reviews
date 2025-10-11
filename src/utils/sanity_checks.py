# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Sequence, Dict
import numpy as np
import pandas as pd

def check_required_columns(df: pd.DataFrame, required: Sequence[str]) -> Dict[str, bool]:
    missing = [c for c in required if c not in df.columns]
    return {"ok": len(missing) == 0, "missing": missing}

def check_missing_values(df: pd.DataFrame, top_k: int = 20) -> pd.Series:
    na = df.isna().mean().sort_values(ascending=False)
    return na.head(top_k)

def check_class_balance(labels: Sequence[int] | pd.Series) -> Dict[int, int]:
    ser = pd.Series(labels)
    counts = ser.value_counts().sort_index().to_dict()
    totals = sum(counts.values())
    ratios = {k: v / totals for k, v in counts.items()}
    return {"counts": counts, "ratios": ratios, "imbalance_ratio": (max(counts.values()) / max(1, min(counts.values())))}
