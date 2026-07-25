"""FUNSD dataset loading and validation utilities."""

from datasets import DatasetDict, load_dataset

from src.config import CONFIG


REQUIRED_COLUMNS = {
    "id",
    "words",
    "bboxes",
    "ner_tags",
    "image",
}


def load_funsd() -> DatasetDict:
    """Load FUNSD and verify required dataset splits."""

    dataset = load_dataset(CONFIG.dataset_name)

    required_splits = {"train", "test"}
    missing_splits = required_splits.difference(dataset.keys())

    if missing_splits:
        raise ValueError(
            f"Missing required FUNSD splits: {sorted(missing_splits)}"
        )

    return dataset


def validate_columns(dataset: DatasetDict) -> None:
    """Ensure required FUNSD fields exist in every split."""

    for split_name in ("train", "test"):
        columns = set(dataset[split_name].column_names)
        missing = REQUIRED_COLUMNS.difference(columns)

        if missing:
            raise ValueError(
                f"{split_name} is missing columns: {sorted(missing)}"
            )


def dataset_summary(dataset: DatasetDict) -> dict[str, int]:
    """Return document counts for all available splits."""

    return {
        split_name: len(split)
        for split_name, split in dataset.items()
    }


def inspect_document(
    dataset: DatasetDict,
    split: str,
    index: int,
) -> dict:
    """Return one document after validating annotation alignment."""

    example = dataset[split][index]

    word_count = len(example["words"])
    box_count = len(example["bboxes"])
    label_count = len(example["ner_tags"])

    if not (word_count == box_count == label_count):
        raise ValueError(
            "Words, bounding boxes and labels are not aligned."
        )

    return example


if __name__ == "__main__":
    funsd = load_funsd()
    validate_columns(funsd)

    print("FUNSD loaded successfully.")

    for split, size in dataset_summary(funsd).items():
        print(f"{split}: {size}")

    document = inspect_document(funsd, "train", 0)

    print("First document ID:", document["id"])
    print("Words:", len(document["words"]))