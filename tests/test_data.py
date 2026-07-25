from src.data import (
    dataset_summary,
    inspect_document,
    load_funsd,
    validate_columns,
)


def test_funsd_has_required_splits():
    dataset = load_funsd()

    assert "train" in dataset
    assert "test" in dataset


def test_funsd_has_required_columns():
    dataset = load_funsd()

    validate_columns(dataset)


def test_funsd_splits_are_not_empty():
    dataset = load_funsd()

    summary = dataset_summary(dataset)

    assert summary["train"] > 0
    assert summary["test"] > 0


def test_words_boxes_labels_are_aligned():
    dataset = load_funsd()

    example = inspect_document(dataset, "train", 0)

    assert len(example["words"]) == len(example["bboxes"])
    assert len(example["words"]) == len(example["ner_tags"])


def test_document_contains_image():
    dataset = load_funsd()

    example = inspect_document(dataset, "train", 0)

    assert example["image"] is not None