"""Entity-level evaluation metrics for token classification."""

from typing import Any

import evaluate
import numpy as np


def build_seqeval_metric():
    """Load the seqeval metric implementation."""
    return evaluate.load("seqeval")


def decode_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert model logits and labels into seqeval label sequences."""

    predicted_ids = np.argmax(
        predictions,
        axis=-1,
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        predicted_ids,
        labels,
    ):
        sequence_predictions = []
        sequence_labels = []

        for predicted_id, label_id in zip(
            prediction,
            label,
        ):
            if label_id == -100:
                continue

            sequence_predictions.append(
                label_list[int(predicted_id)]
            )

            sequence_labels.append(
                label_list[int(label_id)]
            )

        true_predictions.append(
            sequence_predictions
        )

        true_labels.append(
            sequence_labels
        )

    return true_predictions, true_labels


def compute_seqeval_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: list[str],
    metric: Any,
) -> dict[str, float]:
    """Compute entity-level precision, recall, F1 and accuracy."""

    true_predictions, true_labels = decode_predictions(
        predictions,
        labels,
        label_list,
    )

    results = metric.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": float(
            results["overall_precision"]
        ),
        "recall": float(
            results["overall_recall"]
        ),
        "f1": float(
            results["overall_f1"]
        ),
        "accuracy": float(
            results["overall_accuracy"]
        ),
    }