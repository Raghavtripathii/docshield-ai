from typing import Any

from src.config import CONFIG


def encode_document(
    processor: Any,
    image: Any,
    words: list[str],
    boxes: list[list[int]],
    labels: list[int],
):
    """Encode one annotated document for LayoutLMv3."""

    if not (
        len(words)
        == len(boxes)
        == len(labels)
    ):
        raise ValueError(
            "words, boxes, and labels must have identical lengths"
        )

    encoding = processor(
        images=image.convert("RGB"),
        text=words,
        boxes=boxes,
        word_labels=labels,
        truncation=True,
        padding="max_length",
        max_length=CONFIG.max_length,
        return_tensors="pt",
    )

    return encoding


def validate_encoding(encoding: Any) -> None:
    """Validate the critical LayoutLMv3 model inputs."""

    required = {
        "input_ids",
        "attention_mask",
        "bbox",
        "pixel_values",
        "labels",
    }

    missing = required.difference(encoding.keys())

    if missing:
        raise ValueError(
            f"Encoding is missing required fields: {sorted(missing)}"
        )

    sequence_length = encoding["input_ids"].shape[1]

    if sequence_length != CONFIG.max_length:
        raise ValueError(
            f"Expected sequence length {CONFIG.max_length}, "
            f"received {sequence_length}"
        )

    if encoding["bbox"].shape[-1] != 4:
        raise ValueError(
            "Each encoded bounding box must contain four coordinates"
        )

    if encoding["labels"].shape != encoding["input_ids"].shape:
        raise ValueError(
            "Encoded labels must align with encoded input tokens"
        )