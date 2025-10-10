# -*- coding: utf-8 -*-
"""
pipeline/grid_search.py

Grid search for:
  - SVM + TF-IDF (scikit-learn GridSearchCV)
  - (optional) TextCNN via Optuna (if installed and --textcnn passed)

Usage:
  # 1️⃣ Download the original dataset from Kaggle
  # https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset

  # 2️⃣ Run preprocessing (creates data/processed/webmed_*.parquet)
  python -m src.data.make_dataset --input data/raw/webmd.csv --output-dir data/processed

  #  3️⃣ Then run grid search
  python -m pipeline.grid_search --train data/processed/webmed_train.parquet --val data/processed/webmed_val.parquet
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer, f1_score
import joblib

# Optional deep learning (TextCNN via Optuna)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

import torch
from torch.utils.data import DataLoader
from functools import partial

# Local models
from models import (
    Vocabulary, ReviewDataset, ImprovedTextCNN,
    FocalLoss, train_epoch, evaluate, collate_fn_cnn_lstm
)

def ensure_dirs():
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1) SVM + TF-IDF GridSearch
# ---------------------------------------------------------------------
def run_svm_grid(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path) -> dict:
    X_train = train_df["text_clean"].astype(str).values
    y_train = train_df["sent_label"].values
    X_val = val_df["text_clean"].astype(str).values
    y_val = val_df["sent_label"].values

    pipe = SkPipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LinearSVC(class_weight="balanced", max_iter=5000, dual=False)),
    ])

    param_grid = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df": [2, 5],
        "tfidf__max_df": [0.9, 0.95],
        "clf__C": [0.5, 1.0, 2.0]
    }

    scorer = make_scorer(f1_score, average="macro")
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring=scorer, n_jobs=-1, cv=3, verbose=1)
    gs.fit(X_train, y_train)

    # Evaluate on validation
    y_pred = gs.predict(X_val)
    val_macro_f1 = f1_score(y_val, y_pred, average="macro")

    # Save best model
    model_path = out_dir / "svm_tfidf_best.joblib"
    joblib.dump(gs.best_estimator_, model_path)
    print(f"[SVM] Saved best model → {model_path}")

    # Save summary
    result = {
        "best_params": gs.best_params_,
        "cv_best_score_macro_f1": float(gs.best_score_),
        "val_macro_f1": float(val_macro_f1)
    }
    (out_dir / "svm_tfidf_grid_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[SVM] Grid result → {out_dir/'svm_tfidf_grid_result.json'}")
    return result

# ---------------------------------------------------------------------
# 2) TextCNN Optuna (optional)
# ---------------------------------------------------------------------
def objective_textcnn(trial, train_df: pd.DataFrame, val_df: pd.DataFrame, device):
    # Search space
    embed_dim   = trial.suggest_categorical("embed_dim", [128, 200, 300])
    num_filters = trial.suggest_categorical("num_filters", [64, 96, 128])
    dropout     = trial.suggest_float("dropout", 0.3, 0.7)
    lr          = trial.suggest_float("lr", 5e-4, 2e-3, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [32, 48, 64])

    vocab = Vocabulary(max_size=50000, min_freq=2)
    vocab.build_vocab(train_df["text_clean"].values)

    train_ds = ReviewDataset(train_df["text_clean"].values, train_df["sent_label"].values,
                             tokenizer=None, augment=True, augmentation_prob=0.2)
    val_ds   = ReviewDataset(val_df["text_clean"].values, val_df["sent_label"].values,
                             tokenizer=None, augment=False)

    collate  = partial(collate_fn_cnn_lstm, vocab=vocab, max_length=256)
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0,
                    pin_memory=torch.cuda.is_available())
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0,
                    pin_memory=torch.cuda.is_available())

    model = ImprovedTextCNN(vocab_size=len(vocab.word2idx), embed_dim=embed_dim, num_classes=3,
                            kernel_sizes=[2,3,4,5], num_filters=num_filters,
                            dropout=dropout, embed_dropout=0.3, l2_reg=0.005).to(device)

    # Class weights for FocalLoss
    cls_counts = train_df["sent_label"].value_counts().to_dict()
    total = sum(cls_counts.values())
    w = torch.tensor([total/(3*cls_counts.get(i, 1)) for i in range(3)], dtype=torch.float32, device=device)

    criterion = FocalLoss(alpha=w, gamma=2.0)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_f1 = 0.0
    epochs = 8  # short for tuning
    for ep in range(1, epochs+1):
        train_epoch(model, tr, optim, criterion, device)
        _, _, val_f1, _, _, _ = evaluate(model, va, criterion, device)
        best_f1 = max(best_f1, val_f1)
        trial.report(val_f1, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_f1

def run_textcnn_optuna(train_df, val_df, out_dir: Path, trials: int = 20, seed: int = 42):
    if not OPTUNA_AVAILABLE:
        print("[TextCNN] Optuna not installed. Skipping.")
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="textcnn_tuning")
    study.optimize(lambda t: objective_textcnn(t, train_df, val_df, device), n_trials=trials, gc_after_trial=True)

    best = study.best_trial
    summary = {
        "best_value_macro_f1": float(best.value),
        "best_params": best.params
    }
    (out_dir / "textcnn_optuna_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[TextCNN] Optuna best → {summary}")

    # Save study for later review
    try:
        with open(out_dir / "textcnn_optuna_trials.json", "w", encoding="utf-8") as f:
            json.dump([{**t.params, "value": t.value} for t in study.trials], f, indent=2)
    except Exception:
        pass
    return summary

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grid search for WebMD models")
    parser.add_argument("--train", type=str, required=True, help="Path to train parquet")
    parser.add_argument("--val", type=str, required=True, help="Path to val parquet")
    parser.add_argument("--textcnn", action="store_true", help="Also run TextCNN optuna search (if optuna installed)")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials for TextCNN")
    args = parser.parse_args()

    ensure_dirs()
    out_dir = Path("artifacts/models")

    train_df = pd.read_parquet(args.train)
    val_df   = pd.read_parquet(args.val)

    print("\n=== SVM + TF-IDF GridSearch ===")
    svm_res = run_svm_grid(train_df, val_df, out_dir)

    textcnn_res = {}
    if args.textcnn:
        print("\n=== TextCNN Optuna (optional) ===")
        textcnn_res = run_textcnn_optuna(train_df, val_df, out_dir, trials=args.trials)

    # Summary
    summary = {
        "svm_tfidf": svm_res,
        "textcnn_optuna": textcnn_res
    }
    (Path("reports") / "grid_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[SUMMARY] reports/grid_search_summary.json written.")

if __name__ == "__main__":
    main()
