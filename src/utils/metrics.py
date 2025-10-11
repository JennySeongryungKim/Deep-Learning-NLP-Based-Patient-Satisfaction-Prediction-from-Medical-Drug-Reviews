# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from pathlib import Path

def compute_basic_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

def classification_report_dict(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    """
    Sklearn classification_report as dict (per-class precision/recall/F1 + macro/weighted).
    """
    return classification_report(y_true, y_pred, output_dict=True, digits=4)

def plot_confusion_matrix_matplotlib(y_true: Sequence[int], y_pred: Sequence[int], ax=None, labels=None):
    """
    Returns (fig, ax) with a Matplotlib confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Tick labels
    if labels is None:
        labels = list(range(cm.shape[0]))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Annotate
    thresh = cm.max() / 2 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9
            )
    fig.tight_layout()
    return fig, ax

def save_confusion_matrix_matplotlib(y_true: Sequence[int], y_pred: Sequence[int], path: str | Path, labels=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plot_confusion_matrix_matplotlib(y_true, y_pred, labels=labels)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)
