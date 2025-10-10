from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.2
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 10

def stratified_split(
    dataframe: pd.DataFrame,
    target_column: str = "satisfaction_class_10",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/validation/test split with handling for rare classes.
    
    This function ensures that each split maintains the same class distribution
    as the original dataset, while handling classes with very few samples.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe to split
        target_column (str): Column name for stratification target
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        min_samples_per_class (int): Minimum samples needed for stratification
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes
        
    Example:
        >>> train_df, val_df, test_df = stratified_split(df)
        >>> print(f"Split sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")
    """
    df = dataframe[dataframe[target_column].notna()].copy()
    class_counts = df[target_column].value_counts()
    very_rare = class_counts[class_counts < min_samples_per_class].index.tolist()

    if very_rare:
        df_rare = df[df[target_column].isin(very_rare)].copy()
        df_common = df[~df[target_column].isin(very_rare)].copy()
        if len(df_common):
            train_common, temp_common = train_test_split(
                df_common,
                test_size=(1 - train_ratio),
                stratify=df_common[target_column],
                random_state=random_state,
            )
            temp_counts = temp_common[target_column].value_counts()
            can_stratify = all(temp_counts >= 2)
            val_ratio_adj = val_ratio / (val_ratio + test_ratio)
            if can_stratify:
                val_common, test_common = train_test_split(
                    temp_common,
                    test_size=(1 - val_ratio_adj),
                    stratify=temp_common[target_column],
                    random_state=random_state,
                )
            else:
                val_common, test_common = train_test_split(
                    temp_common,
                    test_size=(1 - val_ratio_adj),
                    random_state=random_state,
                )
        else:
            train_common = pd.DataFrame()
            val_common = pd.DataFrame()
            test_common = pd.DataFrame()

        train_df = pd.concat([train_common, df_rare], ignore_index=True).sample(frac=1, random_state=random_state)
        val_df, test_df = val_common, test_common
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df[target_column],
            random_state=random_state,
        )
        val_ratio_adj = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adj),
            stratify=temp_df[target_column],
            random_state=random_state,
        )

    return train_df, val_df, test_df


def temporal_split(
    dataframe: pd.DataFrame,
    date_col: str = "Date",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    as_datetime_format: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split by chronological order (no leakage)."""
    df = dataframe.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=as_datetime_format, errors="coerce")
    df = df.sort_values(date_col)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    return train_df, val_df, test_df
