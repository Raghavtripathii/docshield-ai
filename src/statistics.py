"""Exploratory statistics for FUNSD."""

from collections import Counter

import pandas as pd


def document_statistics(dataset) -> pd.DataFrame:
    """Create per-document FUNSD statistics."""

    rows = []

    for split_name in ("train", "test"):
        for example in dataset[split_name]:
            rows.append(
                {
                    "split": split_name,
                    "document_id": example["id"],
                    "word_count": len(example["words"]),
                    "image_width": example["image"].width,
                    "image_height": example["image"].height,
                }
            )

    return pd.DataFrame(rows)


def label_distribution(dataset, split: str) -> Counter:
    """Count word-level entity labels in a dataset split."""

    if split not in dataset:
        raise ValueError(f"Unknown dataset split: {split}")

    counts = Counter()

    for example in dataset[split]:
        counts.update(example["ner_tags"])

    return counts