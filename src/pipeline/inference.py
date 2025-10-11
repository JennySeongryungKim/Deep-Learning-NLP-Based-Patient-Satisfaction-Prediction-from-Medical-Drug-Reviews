# -*- coding: utf-8 -*-
"""
pipeline/inference.py

Batch inference for trained models:
  - textcnn / bilstm / bert 

Examples:
  python -m pipeline.inference --model textcnn --ckpt experiments/best_textcnn.pt --input my_texts.csv --text-col Review
  python -m pipeline.inference --model bert --ckpt experiments/best_bert.pt --input data/processed/webmed_test.parquet
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial

from models import (
    Vocabulary, ReviewDataset,
    ImprovedTextCNN, ImprovedBiLSTMAttention, ImprovedBERTClassifier,
    collate_fn_cnn_lstm
)

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

def ensure_dirs():
    Path("artifacts/preds").mkdir(parents=True, exist_ok=True)

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    elif p.suffix.lower() in [".csv"]:
        return pd.read_csv(p)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

def build_or_load_vocab(vocab_path: str | None, processed_dir: str | None) -> Vocabulary:
    vocab = Vocabulary(max_size=50_000, min_freq=2)
    if vocab_path and Path(vocab_path).exists():
        # load saved vocab (json)
        import json
        obj = json.loads(Path(vocab_path).read_text())
        vocab.word2idx = obj["word2idx"]
        vocab.idx2word = {int(k): v for k, v in obj["idx2word"].items()}
        print(f"[Vocab] Loaded from {vocab_path} (size={len(vocab.word2idx)})")
        return vocab

    # Rebuild from processed train parquet
    if processed_dir:
        tr_p = Path(processed_dir) / "webmed_train.parquet"
        if tr_p.exists():
            tr = pd.read_parquet(tr_p)
            vocab.build_vocab(tr["text_clean"].astype(str).values)
            print(f"[Vocab] Rebuilt from {tr_p} (size={len(vocab.word2idx)})")
            return vocab

    raise ValueError("Vocab not provided. Use --vocab or set --processed-dir with train parquet.")

def save_vocab(vocab: Vocabulary, path: str):
    import json
    obj = {
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}
    }
    Path(path).write_text(json.dumps(obj), encoding="utf-8")

def run_textcnn_infer(df: pd.DataFrame, text_col: str, ckpt: str,
                      vocab: Vocabulary, device, batch_size=64):
    ds = ReviewDataset(df[text_col].astype(str).values, [0]*len(df), tokenizer=None, augment=False)
    collate = partial(collate_fn_cnn_lstm, vocab=vocab, max_length=256)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0,
                    pin_memory=torch.cuda.is_available())

    model = ImprovedTextCNN(vocab_size=len(vocab.word2idx), embed_dim=300, num_classes=3,
                            kernel_sizes=[2,3,4,5], num_filters=128, dropout=0.5,
                            embed_dropout=0.3, l2_reg=0.005).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["text"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_probs

def run_bilstm_infer(df: pd.DataFrame, text_col: str, ckpt: str,
                     vocab: Vocabulary, device, batch_size=64):
    ds = ReviewDataset(df[text_col].astype(str).values, [0]*len(df), tokenizer=None, augment=False)
    collate = partial(collate_fn_cnn_lstm, vocab=vocab, max_length=256)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0,
                    pin_memory=torch.cuda.is_available())

    model = ImprovedBiLSTMAttention(vocab_size=len(vocab.word2idx), embed_dim=300,
                                    hidden_dim=128, num_classes=3, num_layers=2,
                                    dropout=0.7, recurrent_dropout=0.5,
                                    embed_dropout=0.4, l2_reg=0.02).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["text"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_probs

def run_bert_infer(df: pd.DataFrame, text_col: str, ckpt: str, device,
                   model_name="emilyalsentzer/Bio_ClinicalBERT", batch_size=32):
    if not TRANSFORMERS_OK:
        raise ImportError("transformers not installed. pip install transformers")
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = ReviewDataset(df[text_col].astype(str).values, [0]*len(df), tokenizer=tok, max_length=256)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=torch.cuda.is_available())

    model = ImprovedBERTClassifier(model_name=model_name, num_classes=3,
                                   dropout=0.3, hidden_dropout=0.5,
                                   freeze_bert_layers=0, use_pooler=True).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch in dl:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description="Batch inference for WebMD models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["textcnn", "bilstm", "bert"])
    parser.add_argument("--input", type=str, required=True, help="CSV or Parquet with texts")
    parser.add_argument("--text-col", type=str, default="text_clean")
    parser.add_argument("--ckpt", type=str, help="Checkpoint for single model")
    parser.add_argument("--vocab", type=str, default=None, help="Path to saved vocab json")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="Directory to rebuild vocab if --vocab not provided")
    parser.add_argument("--out-prefix", type=str, default=None, help="Prefix for output files")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_table(args.input)

    # Text column check
    if args.text_col not in df.columns:
        # try to fallback to Review
        if "Review" in df.columns:
            args.text_col = "Review"
            print(f"[INFO] Using fallback text column 'Review'")
        else:
            raise ValueError(f"Text column '{args.text_col}' not found in input.")

    # Output naming
    tag = args.out_prefix or Path(args.input).stem
    out_csv = Path("artifacts/preds") / f"{args.model}_{tag}_preds.csv"
    out_npy = Path("artifacts/preds") / f"{args.model}_{tag}_probs.npy"

    if args.model in ("textcnn", "bilstm"):
        vocab = build_or_load_vocab(args.vocab, args.processed_dir)
        # Save vocab next to preds (helpful for reproducibility)
        try:
            save_vocab(vocab, str(Path("artifacts/preds") / f"{args.model}_{tag}_vocab.json"))
        except Exception:
            pass

    if args.model == "textcnn":
        if not args.ckpt:
            raise ValueError("--ckpt is required for textcnn")
        preds, probs = run_textcnn_infer(df, args.text_col, args.ckpt, vocab, device)

    elif args.model == "bilstm":
        if not args.ckpt:
            raise ValueError("--ckpt is required for bilstm")
        preds, probs = run_bilstm_infer(df, args.text_col, args.ckpt, vocab, device)

    elif args.model == "bert":
        if not args.ckpt:
            raise ValueError("--ckpt is required for bert")
        preds, probs = run_bert_infer(df, args.text_col, args.ckpt, device)


        # Run three models
        p_textcnn, pr_textcnn = run_textcnn_infer(df, args.text_col, args.ckpt_textcnn, vocab, device)
        p_bilstm,  pr_bilstm  = run_bilstm_infer(df, args.text_col, args.ckpt_bilstm, vocab, device)
        p_bert,    pr_bert    = run_bert_infer(df, args.text_col, args.ckpt_bert, device)

        # Weighted soft voting (0.5, 0.3, 0.2)
        preds, probs = soft_vote([pr_bert, pr_bilstm, pr_textcnn], weights=[0.5, 0.3, 0.2])

    # Save outputs
    out_df = pd.DataFrame({
        "text": df[args.text_col],
        "pred": preds
    })
    out_df.to_csv(out_csv, index=False)
    np.save(out_npy, probs)
    print(f"[DONE] Predictions → {out_csv}")
    print(f"[DONE] Probabilities → {out_npy}")

if __name__ == "__main__":
    main()
