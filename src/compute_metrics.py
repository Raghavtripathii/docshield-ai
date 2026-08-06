from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    mask = labels != -100

    labels = labels[mask]

    predictions = predictions[mask]

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }