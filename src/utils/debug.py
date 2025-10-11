# -*- coding: utf-8 -*-
from typing import Iterable, Sequence
import pandas as pd

def debug_print(msg: str):
    print(f"[DEBUG] {msg}")

def assert_true(cond: bool, msg: str = "Assertion failed"):
    if not cond:
        raise AssertionError(msg)

def assert_has_columns(df: pd.DataFrame, cols: Sequence[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing required columns: {missing}")

def sample_output(df: pd.DataFrame, n: int = 5, cols: Sequence[str] | None = None) -> pd.DataFrame:
    if cols:
        cols = [c for c in cols if c in df.columns]
        return df.loc[:, cols].head(n)
    return df.head(n)

def peek_iter(it: Iterable, k: int = 3):
    out = []
    for i, x in enumerate(it):
        if i >= k: break
        out.append(x)
    debug_print(f"peek_iter first {len(out)} items: {out}")
    return out
