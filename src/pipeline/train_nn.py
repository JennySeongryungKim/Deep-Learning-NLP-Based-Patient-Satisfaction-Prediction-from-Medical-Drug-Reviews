# -*- coding: utf-8 -*-
"""
pipeline/train_nn.py

End-to-end training runner for WebMD Drug Reviews:
  1) Load processed parquet (or run preprocess to create them)
  2) (Optional) Run EDA
  3) Train/evaluate SVM / TextCNN / BiLSTM / BERT
  4) Save metrics, plots, and a model card

Usage:
  python -m pipeline.train_nn --input webmd.csv --model all(Kaggle)
  python -m pipeline.train_nn --model textcnn --epochs 15
"""

from __future__ import annotations
import os
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

# ------------------------------
# Local imports
# ------------------------------
from src.data.make_dataset import preprocess_pipeline
from models import (
    # architectures
    Vocabulary, ReviewDataset, ImprovedTextCNN, ImprovedBiLSTMAttention,
    ImprovedBERTClassifier,
    #losses / callbacks / loops / eval
    FocalLoss, LabelSmoothingCrossEntropy, EarlyStopping,
    get_optimizer_with_decay, mixup_data,
    train_epoch, collate_fn_cnn_lstm, evaluate, visualize_confusion_matrix
)

# Optional: EDA (soft-dependency)
try:
    from src.analysis.eda import WebMDEDA
    EDA_AVAILABLE = True
except Exception:
    EDA_AVAILABLE = False

# Torch (only needed when training DL models)
import torch
from torch.utils.data import DataLoader
from functools import partial
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

# Plotting utils
import matplotlib.pyplot as plt


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs():
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("experiments").mkdir(parents=True, exist_ok=True)
    Path("artifacts/preds").mkdir(parents=True, exist_ok=True)


def load_or_preprocess(input_csv: str, processed_dir: str):
    """Load processed parquet if exists, else run preprocess."""
    processed = Path(processed_dir)
    train_p = processed / "webmed_train.parquet"
    val_p   = processed / "webmed_val.parquet"
    test_p  = processed / "webmed_test.parquet"

    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        print("[INFO] Processed data not found. Running preprocessing pipeline...")
        preprocess_pipeline(filepath=input_csv, output_dir=processed)
    else:
        print("[INFO] Found processed datasets. Skipping preprocess.")

    train_df = pd.read_parquet(train_p)
    val_df   = pd.read_parquet(val_p)
    test_df  = pd.read_parquet(test_p)
    print(f"[INFO] Loaded Processed: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    return train_df, val_df, test_df


def run_eda_if_available(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    if not EDA_AVAILABLE:
        print("[INFO] EDA module not found. Skipping EDA.")
        return
    print("\n" + "="*80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    try:
        eda = WebMDEDA(combined, output_dir="reports/eda")
        eda.generate_report()
        print("[SUCCESS] EDA completed → reports/eda/")
    except Exception as e:
        print(f"[WARNING] EDA failed: {e}")


def generate_model_card(results: dict, output_path: str = "reports/model_card.md"):
    """Simple model card writer."""
    lines = [
        "# Model Card: WebMD Drug Review Sentiment Analysis",
        "",
        "## Model Performance",
        ""
    ]
    for name, metrics in results.items():
        lines.append(f"### {name.upper()}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"- {k}: {v:.4f}")
        lines.append("")
    lines += [
        "## Dataset",
        "- Source: WebMD Drug Reviews (Kaggle)",
        "- Task: 3-class sentiment (0/1/2) & 10-class satisfaction (1–10)",
        "",
        "## Preprocessing",
        "- Text cleaning, negation tagging, (optional) medical NER",
        "",
        "## Limitations & Ethics",
        "- Not for medical decision-making; potential subgroup biases.",
        ""
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Model card saved → {output_path}")


def save_results_table(results: dict, out_csv: str = "reports/model_comparison.csv"):
    if not results:
        return
    df = pd.DataFrame(results).T
    df = df.round(4)
    df.to_csv(out_csv)
    print(f"[INFO] Results table saved → {out_csv}")
    print(df.sort_values(df.columns[0], ascending=False).to_string())


# =============================================================================
# Model runners (thin wrappers around existing classes)
# =============================================================================

def run_svm_tfidf(train_df, val_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    X_train = train_df["text_clean"].astype(str).values
    y_train = train_df["sent_label"].values
    X_val   = val_df["text_clean"].astype(str).values
    y_val   = val_df["sent_label"].values

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                          min_df=2, max_df=0.95, stop_words="english",
                          sublinear_tf=True, norm="l2")
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    clf = LinearSVC(class_weight="balanced", max_iter=2000, C=1.0, dual=False)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xva)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
        "cohen_kappa": cohen_kappa_score(y_val, y_pred)
    }
    print("[SVM] ", metrics)
    pd.DataFrame({"text": X_val, "true": y_val, "pred": y_pred}).to_csv("experiments/svm_val_preds.csv", index=False)
    return metrics


def build_vocab(train_df):
    vocab = Vocabulary(max_size=50_000, min_freq=2)
    vocab.build_vocab(train_df["text_clean"].values)
    return vocab


def run_textcnn(train_df, val_df, device, epochs=15, batch_size=32, lr=1e-3):
    # Vocab & datasets
    vocab = build_vocab(train_df)
    train_ds = ReviewDataset(train_df["text_clean"].values, train_df["sent_label"].values,
                             tokenizer=None, augment=True, augmentation_prob=0.2)
    val_ds   = ReviewDataset(val_df["text_clean"].values, val_df["sent_label"].values,
                             tokenizer=None, augment=False)
    collate  = partial(collate_fn_cnn_lstm, vocab=vocab, max_length=256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    model = ImprovedTextCNN(vocab_size=len(vocab.word2idx), embed_dim=300, num_classes=3,
                            kernel_sizes=[2,3,4,5], num_filters=128, dropout=0.5,
                            embed_dropout=0.3, l2_reg=0.005).to(device)
    print(f"[TextCNN] Params: {sum(p.numel() for p in model.parameters()):,}")

    # Focal loss with inverse-frequency weights
    cls_counts = train_df["sent_label"].value_counts().to_dict()
    total = sum(cls_counts.values())
    w = torch.tensor([total/(3*cls_counts.get(i, 1)) for i in range(3)], dtype=torch.float32, device=device)
    criterion = FocalLoss(alpha=w, gamma=2.0)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=5, T_mult=2, eta_min=1e-6)
    best_f1, best_state = 0.0, None

    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, criterion, device)
        va_loss, va_acc, va_f1, va_kappa, _, _ = evaluate(model, val_loader, criterion, device)
        sched.step()
        print(f"[TextCNN][{ep}/{epochs}] tr_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} kappa={va_kappa:.4f}")
        if va_f1 > best_f1:
            best_f1, best_state = va_f1, model.state_dict()
            torch.save(best_state, "experiments/best_textcnn.pt")
            print(f"[TextCNN] ✅ New best F1={best_f1:.4f} → experiments/best_textcnn.pt")

    if best_state:
        model.load_state_dict(best_state)
    # final
    va_loss, va_acc, va_f1, va_kappa, y_pred, y_true = evaluate(model, val_loader, criterion, device)
    return {"accuracy": va_acc, "macro_f1": va_f1, "cohen_kappa": va_kappa}


def run_bilstm(train_df, val_df, device, epochs=20, batch_size=32, lr=1e-3):
    vocab = build_vocab(train_df)
    train_ds = ReviewDataset(train_df["text_clean"].values, train_df["sent_label"].values,
                             tokenizer=None, augment=True, augmentation_prob=0.25)
    val_ds   = ReviewDataset(val_df["text_clean"].values, val_df["sent_label"].values,
                             tokenizer=None, augment=False)
    collate  = partial(collate_fn_cnn_lstm, vocab=vocab, max_length=256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0,
                              pin_memory=torch.cuda.is_available())

    model = ImprovedBiLSTMAttention(vocab_size=len(vocab.word2idx), embed_dim=300,
                                    hidden_dim=128, num_classes=3, num_layers=2,
                                    dropout=0.7, recurrent_dropout=0.5,
                                    embed_dropout=0.4, l2_reg=0.02).to(device)
    criterion = LabelSmoothingCrossEntropy(classes=3, smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=2)
    es = EarlyStopping(patience=5, min_delta=0.001, mode='max')

    best_f1, best_state = 0.0, None
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, criterion, device)
        va_loss, va_acc, va_f1, va_kappa, _, _ = evaluate(model, val_loader, criterion, device)
        sched.step(va_f1)
        print(f"[BiLSTM][{ep}/{epochs}] tr_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} kappa={va_kappa:.4f}")
        if va_f1 > best_f1:
            best_f1, best_state = va_f1, model.state_dict()
            torch.save(best_state, "experiments/best_bilstm.pt")
            print(f"[BiLSTM] ✅ New best F1={best_f1:.4f} → experiments/best_bilstm.pt")
        if es(va_f1):
            print("[BiLSTM] Early stopping.")
            break

    if best_state:
        model.load_state_dict(best_state)
    va_loss, va_acc, va_f1, va_kappa, _, _ = evaluate(model, val_loader, criterion, device)
    return {"accuracy": va_acc, "macro_f1": va_f1, "cohen_kappa": va_kappa}


def run_bert(train_df, val_df, device, model_name="emilyalsentzer/Bio_ClinicalBERT",
             epochs=3, batch_size=16, lr=2e-5):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tr_ds = ReviewDataset(train_df["text_clean"].values, train_df["sent_label"].values,
                          tokenizer=tokenizer, max_length=256)
    va_ds = ReviewDataset(val_df["text_clean"].values, val_df["sent_label"].values,
                          tokenizer=tokenizer, max_length=256)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=torch.cuda.is_available())
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=torch.cuda.is_available())

    model = ImprovedBERTClassifier(model_name=model_name, num_classes=3,
                                   dropout=0.3, hidden_dropout=0.5,
                                   freeze_bert_layers=4, use_pooler=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)

    best_f1, best_state = 0.0, None
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, tr, optim, criterion, device, accumulation_steps=2)
        va_loss, va_acc, va_f1, va_kappa, _, _ = evaluate(model, va, criterion, device)
        sched.step()
        print(f"[BERT][{ep}/{epochs}] tr_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} kappa={va_kappa:.4f}")
        if va_f1 > best_f1:
            best_f1, best_state = va_f1, model.state_dict()
            torch.save(best_state, "experiments/best_bert.pt")
            print(f"[BERT] ✅ New best F1={best_f1:.4f} → experiments/best_bert.pt")

    if best_state:
        model.load_state_dict(best_state)
    va_loss, va_acc, va_f1, va_kappa, _, _ = evaluate(model, va, criterion, device)
    return {"accuracy": va_acc, "macro_f1": va_f1, "cohen_kappa": va_kappa}


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Train models on WebMD Drug Reviews")
    parser.add_argument("--input", type=str, default="data/raw/webmd.csv", help="Raw CSV path (for preprocess)")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Processed parquet directory")
    parser.add_argument("--model", type=str, default="all",
                        choices=["svm", "textcnn", "bilstm", "bert", "all"],
                        help="Which model to run")
    parser.add_argument("--epochs", type=int, default=0, help="Override epochs for DL models (0 = use defaults)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0 = auto)")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    args = parser.parse_args()

    # Load or preprocess
    train_df, val_df, test_df = load_or_preprocess(args.input, args.processed_dir)

    # Optional EDA
    if not args.skip_eda:
        run_eda_if_available(train_df, val_df, test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    results = {}

    # Model selection
    run_all = (args.model == "all")
    override_epochs = args.epochs if args.epochs > 0 else None
    override_bs = args.batch_size if args.batch_size > 0 else None

    # 1) SVM (baseline)
    if run_all or args.model == "svm":
        print("\n" + "="*80); print("MODEL: SVM + TF-IDF"); print("="*80)
        results["svm_tfidf"] = run_svm_tfidf(train_df, val_df)

    # 2) TextCNN
    if run_all or args.model == "textcnn":
        print("\n" + "="*80); print("MODEL: TextCNN"); print("="*80)
        e = override_epochs if override_epochs is not None else 15
        bs = override_bs if override_bs is not None else (64 if len(train_df) > 20_000 else 32)
        results["textcnn"] = run_textcnn(train_df, val_df, device, epochs=e, batch_size=bs, lr=1e-3)

    # 3) BiLSTM
    if run_all or args.model == "bilstm":
        print("\n" + "="*80); print("MODEL: BiLSTM + Attention"); print("="*80)
        e = override_epochs if override_epochs is not None else 20
        bs = override_bs if override_bs is not None else (64 if len(train_df) > 20_000 else 32)
        results["bilstm"] = run_bilstm(train_df, val_df, device, epochs=e, batch_size=bs, lr=1e-3)

    # 4) BERT
    if run_all or args.model == "bert":
        print("\n" + "="*80); print("MODEL: BERT (Bio_ClinicalBERT)"); print("="*80)
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available. Skipping BERT (CPU is extremely slow).")
        else:
            e = override_epochs if override_epochs is not None else 3
            bs = override_bs if override_bs is not None else 16
            results["bert"] = run_bert(train_df, val_df, device, epochs=e, batch_size=bs, lr=2e-5)

    # Save results & model card
    if results:
        save_results_table(results, out_csv="reports/model_comparison.csv")
        generate_model_card(results, output_path="reports/model_card.md")

        # Optional: simple bar plot
        try:
            fig_path = Path("reports/figures/model_comparison_bars.png")
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            labels = list(results.keys())
            macro_f1 = [results[m].get("macro_f1", 0.0) for m in labels]
            plt.figure(figsize=(10, 4))
            plt.bar(labels, macro_f1)
            plt.title("Macro-F1 by Model")
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[INFO] Simple bar plot saved → {fig_path}")
        except Exception as e:
            print(f"[WARNING] Plot failed: {e}")

    print("\n" + "="*80)
    print("🎉 TRAINING PIPELINE FINISHED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
