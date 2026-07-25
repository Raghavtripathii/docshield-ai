import pytest

from src.validation import (
    is_valid_box,
    validate_document_annotations,
)


def test_valid_box():
    assert is_valid_box([0, 0, 1000, 1000])


def test_box_rejects_wrong_length():
    assert not is_valid_box([0, 0, 10])


def test_box_rejects_out_of_range_coordinate():
    assert not is_valid_box([0, 0, 1001, 100])


def test_box_rejects_reversed_x_coordinates():
    assert not is_valid_box([500, 100, 100, 200])


def test_box_rejects_boolean_coordinate():
    assert not is_valid_box([False, 0, 100, 100])


def test_annotation_alignment():
    validate_document_annotations(
        ["hello", "world"],
        [[0, 0, 100, 100], [100, 100, 200, 200]],
        [0, 1],
    )


def test_annotation_alignment_failure():
    with pytest.raises(ValueError):
        validate_document_annotations(
            ["hello"],
            [[0, 0, 100, 100]],
            [0, 1],
        )