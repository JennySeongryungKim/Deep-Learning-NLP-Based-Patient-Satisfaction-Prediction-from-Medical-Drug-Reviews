from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Union, Tuple, Optional, Iterable, Set
import warnings

import numpy as np
import pandas as pd

from .split import stratified_split
from .negation import apply_negation_tagging

# Optional libs
import textblob
from textblob import TextBlob

# Optional: spaCy NER (lightweight)
try:
    import spacy
    SPACY_OK = True
except Exception:
    spacy = None  # type: ignore
    SPACY_OK = False

warnings.filterwarnings("ignore")

# --------------------
# Config
# --------------------
DEFAULT_TEXT_CANDIDATES = ("Review", "Reviews", "review_text")
DEFAULT_TEXT_COLUMN = "Review"
DEFAULT_MIN_WORDS = 3
DEFAULT_MAX_WORDS = 2000
DEFAULT_RANDOM_STATE = 42

# --------------------
# I/O
# --------------------
def load_and_validate_data(filepath: Union[str, Path]) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    print(f"[INFO] Loaded {len(df):,} rows, columns={list(df.columns)}")
    return df

def remove_duplicates(df: pd.DataFrame, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[text_column], keep="first")
    print(f"[INFO] Duplicates removed: {before - len(df)}")
    return df

def _ensure_text_column(df: pd.DataFrame, prefer: str = DEFAULT_TEXT_COLUMN) -> str:
    cols = set(df.columns)
    if prefer in cols:
        return prefer
    for c in DEFAULT_TEXT_CANDIDATES:
        if c in cols:
            return c
    # last resort: pick first object column
    for c in df.columns:
        if df[c].dtype == "object":
            print(f"[WARN] Using '{c}' as text column (best guess).")
            return c
    raise ValueError("No suitable text column found.")

def filter_by_length(
    df: pd.DataFrame,
    text_column: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> pd.DataFrame:
    def wc(x: str) -> int:
        if pd.isna(x):
            return 0
        return len(str(x).split())
    s = df[text_column].astype(str)
    df = df.assign(_wc=s.apply(wc))
    before = len(df)
    df = df[(df["_wc"] >= min_words) & (df["_wc"] <= max_words)].drop(columns=["_wc"])
    print(f"[INFO] Length filter removed: {before - len(df)} (remain={len(df):,})")
    return df

# --------------------
# Cleaning
# --------------------
_URL = re.compile(r"http\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_EMOJI = re.compile(
    "[" +
    "\U0001F600-\U0001F64F" +
    "\U0001F300-\U0001F5FF" +
    "\U0001F680-\U0001F6FF" +
    "\U0001F1E0-\U0001F1FF" +
    "\u2702-\u27B0" +
    "\u24C2-\U0001F251" +
    "]+",
    flags=re.UNICODE,
)

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = _HTML.sub("", t)
    t = _URL.sub("", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def apply_text_cleaning(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df[text_column].apply(clean_text)
    if "Drug" in df.columns:
        whitelist = set(df["Drug"].dropna().astype(str).str.lower().unique())
    else:
        whitelist = set()
    # ⚠️ 느리면 주석 유지
    # df["text_clean"] = df["text_clean"].apply(lambda x: spell_correction(x, whitelist))
    return df

# --------------------
# NER (light)
# --------------------
def extract_medical_entities(df: pd.DataFrame, text_column: str = "text_clean") -> pd.DataFrame:
    if not SPACY_OK:
        df = df.copy(); df["entities"] = [{}] * len(df)
        return df
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        df = df.copy(); df["entities"] = [{}] * len(df)
        return df

    ents = []
    for _, tx in df[text_column].astype(str).items():
        doc = nlp(tx)
        bucket = {"DRUG": set(), "CONDITION": set(), "SYMPTOM": set(), "ADR": set()}
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"]:
                bucket["DRUG"].add(ent.text)
            elif ent.label_ in ["DISEASE"]:
                bucket["CONDITION"].add(ent.text)
        ents.append(json.dumps({k: sorted(list(v)) for k, v in bucket.items()}))
    out = df.copy()
    out["entities"] = ents
    return out

# --------------------
# Labeling
# --------------------
def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive target labels for both regression and classification tasks.
    
    This function creates multiple label representations from the original 1-5 satisfaction scale:
    - satisfaction_score_10: Scaled from 1-5 to 1-10 for regression
    - satisfaction_class_10: 10-class classification labels (1-10)
    - sent_label: 3-class sentiment labels (0=negative, 1=neutral, 2=positive)
    - satisfaction_reg: Normalized regression target (0-1 scale)
    
    Args:
        dataframe (pd.DataFrame): Input dataframe containing 'Satisfaction' column
        
    Returns:
        pd.DataFrame: Dataframe with derived labels added
        
    Raises:
        ValueError: If 'Satisfaction' column is missing
        
    Example:
        >>> df_labeled = derive_labels(df)
        >>> print(df_labeled['sent_label'].value_counts())
    """
    
    if "Satisfaction" not in df.columns:
        raise ValueError("'Satisfaction' column not found.")
    out = df.copy()
    out["satisfaction_raw"] = pd.to_numeric(out["Satisfaction"], errors="coerce")
    out = out[(out["satisfaction_raw"] >= 1) & (out["satisfaction_raw"] <= 5)].copy()
    out["satisfaction_score_10"] = out["satisfaction_raw"] * 2
    out["satisfaction_class_10"] = out["satisfaction_score_10"].round().astype("Int64")

    def to_sent(x: float) -> int:
        if pd.isna(x): return -1
        if x <= 2: return 0
        elif x <= 3: return 1
        else: return 2

    out["sent_label"] = out["satisfaction_raw"].apply(to_sent)
    out["satisfaction_reg"] = (out["satisfaction_score_10"] - 1) / 9
    return out

   # Create balanced 3-class sentiment labels
    def score_to_sentiment_balanced(score: float) -> int:
        """
        Convert 1-5 satisfaction score to balanced 3-class sentiment.
        
        This mapping is designed to create relatively balanced classes:
        - 1-2: Negative sentiment (~136K samples)
        - 3:   Neutral sentiment (~52K samples)  
        - 4-5: Positive sentiment (~175K samples)
        
        Args:
            score (float): Satisfaction score from 1-5 scale
            
        Returns:
            int: Sentiment label (0=negative, 1=neutral, 2=positive)
        """
        if pd.isna(score):
            return -1
        if score <= 2:      # 1-2 stars → negative
            return 0
        elif score <= 3:    # 3 stars → neutral
            return 1
        else:               # 4-5 stars → positive
            return 2

    # Apply sentiment labeling
    dataframe['sent_label'] = dataframe['satisfaction_raw'].apply(score_to_sentiment_balanced)

    # Create normalized regression target (0-1 scale)
    dataframe['satisfaction_reg'] = (dataframe['satisfaction_score_10'] - 1) / 9

    # Display label distributions
    print(f"\n[INFO] 10-class distribution:")
    print(dataframe['satisfaction_class_10'].value_counts().sort_index())

    print(f"\n[INFO] 3-class Sentiment distribution:")
    sentiment_counts = dataframe['sent_label'].value_counts().sort_index()
    total_samples = sentiment_counts.sum()
    print(sentiment_counts)
    
    print(f"\n[INFO] Sentiment percentages:")
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    for label, count in sentiment_counts.items():
        if label in sentiment_names:
            label_name = sentiment_names[label]
            percentage = count / total_samples * 100
            print(f"  {label} ({label_name}): {count:>7,} ({percentage:>5.1f}%)")

    print(f"\n[INFO] Sentiment mapping:")
    print("  0 = Negative (1-2 stars)")
    print("  1 = Neutral  (3 stars)")
    print("  2 = Positive (4-5 stars)")

    return dataframe

# --------------------
# Save
# --------------------
def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/processed",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output / "webmed_train.parquet", index=False)
    val_df.to_parquet(output / "webmed_val.parquet", index=False)
    test_df.to_parquet(output / "webmed_test.parquet", index=False)
    pd.concat([train_df, val_df, test_df], ignore_index=True).to_parquet(output / "webmed_clean.parquet", index=False)
    print(f"[INFO] Saved to {output.resolve()}")

# --------------------
# Orchestrator
# --------------------
def preprocess_pipeline(filepath: Union[str, Path], 
                       output_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete WebMD drug review preprocessing pipeline.
    
    This function orchestrates the entire data preprocessing workflow including:
    1. Data loading and validation
    2. Duplicate removal and length filtering
    3. Text cleaning and normalization
    4. Medical entity extraction
    5. Negation tagging
    6. Label derivation for multiple tasks
    7. Stratified train/validation/test splitting
    8. Data saving to parquet format
    
    Args:
        filepath (Union[str, Path]): Path to the input CSV file containing WebMD reviews
        output_dir (str): Directory to save processed data files. Defaults to 'data/processed'
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes
        
    Example:
        >>> train_df, val_df, test_df = preprocess_pipeline('webmd_reviews.csv')
        >>> print(f"Pipeline complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    """
    print("="*80)
    print("WEBMD DRUG REVIEW - DATA PREPROCESSING PIPELINE")
    print("="*80)

    # Step 1.1: Load and validate raw data
    dataframe = load_and_validate_data(filepath)

    # Step 1.2: Data quality improvements
    dataframe = remove_duplicates(dataframe, text_column=DEFAULT_TEXT_COLUMN)
    dataframe = filter_by_length(dataframe, text_column=DEFAULT_TEXT_COLUMN, 
                                min_words=DEFAULT_MIN_WORDS, max_words=DEFAULT_MAX_WORDS)

    # Step 1.3: Text preprocessing
    dataframe = apply_text_cleaning(dataframe, text_column=DEFAULT_TEXT_COLUMN)

    # Step 1.4: Medical entity extraction (optional)
    dataframe = extract_medical_entities(dataframe, text_column='text_clean')

    # Step 1.5: Negation handling
    dataframe = apply_negation_tagging(dataframe, text_column='text_clean')

    # Step 1.6: Label derivation for multiple tasks
    dataframe = derive_labels(dataframe)

    # Step 1.7: Stratified data splitting
    train_dataframe, val_dataframe, test_dataframe = stratified_split(dataframe)
    
    # Step 1.8: Save processed data
    save_processed_data(train_dataframe, val_dataframe, test_dataframe, output_dir=output_dir)

    print("="*80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*80)

    return train_dataframe, val_dataframe, test_dataframe
                         
# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="WebMD Drug Review Data Preprocessing")
    p.add_argument("--input", type=str, required=True, help="Path to raw CSV (e.g., data/raw/webmd.csv)")
    p.add_argument("--output-dir", type=str, default="data/processed")
    p.add_argument("--temporal", action="store_true", help="Use temporal split if Date column exists")
    args = p.parse_args()

    train_df, val_df, test_df = preprocess_pipeline(
        filepath=args.input,
        output_dir=args.output_dir,
        use_temporal_split=args.temporal,
    )
    cols = [c for c in ["Drug", "Condition", "text_clean", "satisfaction_class_10", "sent_label"] if c in train_df.columns]
    print("\n[INFO] Sample rows:"); print(train_df[cols].head())
