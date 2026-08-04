from PIL import Image
import numpy as np
import pytest

from src.corruptions import DocumentCorruptor


def test_rotation_returns_image():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (256, 256), "white")

    result = corruptor.rotation(
        image,
        severity=3,
    )

    assert isinstance(result, Image.Image)


def test_rotation_preserves_size():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    result = corruptor.rotation(
        image,
        severity=5,
    )

    assert result.size == image.size


def test_rotation_invalid_severity():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (64, 64), "white")

    with pytest.raises(ValueError):
        corruptor.rotation(image, 0)

    with pytest.raises(ValueError):
        corruptor.rotation(image, 6)


def test_rotation_changes_pixels():
    corruptor = DocumentCorruptor()

    image = Image.fromarray(
        np.random.randint(
            0,
            256,
            (128, 128, 3),
            dtype=np.uint8,
        )
    )

    result = corruptor.rotation(
        image,
        severity=5,
    )

    assert not np.array_equal(
        np.array(image),
        np.array(result),
    )