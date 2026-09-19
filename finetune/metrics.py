"""Classification metrics for Hugging Face Trainer."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def build_compute_metrics():
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
        }

    return compute_metrics
