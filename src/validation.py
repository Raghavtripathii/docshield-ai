"""Validation utilities for document annotations."""

from collections.abc import Sequence


def is_valid_box(box: Sequence[int]) -> bool:
    """Return whether a normalized document bounding box is valid."""

    if len(box) != 4:
        return False

    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in box
    ):
        return False

    x0, y0, x1, y1 = box

    if not all(0 <= value <= 1000 for value in box):
        return False

    if x1 < x0 or y1 < y0:
        return False

    return True


def validate_document_annotations(
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> None:
    """Validate annotation alignment and spatial coordinates."""

    if not (len(words) == len(boxes) == len(labels)):
        raise ValueError(
            "words, boxes and labels must contain the same number of items"
        )

    invalid_indices = [
        index
        for index, box in enumerate(boxes)
        if not is_valid_box(box)
    ]

    if invalid_indices:
        raise ValueError(
            f"Invalid bounding boxes at indices: {invalid_indices[:10]}"
        )