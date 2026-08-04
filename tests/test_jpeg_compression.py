from PIL import Image
import numpy as np
import pytest

from src.corruptions import DocumentCorruptor


def test_jpeg_returns_image():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (256, 256), "white")

    result = corruptor.jpeg_compression(
        image,
        severity=3,
    )

    assert isinstance(result, Image.Image)


def test_jpeg_preserves_size():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    result = corruptor.jpeg_compression(
        image,
        severity=4,
    )

    assert result.size == image.size


def test_jpeg_invalid_severity():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (64, 64), "white")

    with pytest.raises(ValueError):
        corruptor.jpeg_compression(image, 0)

    with pytest.raises(ValueError):
        corruptor.jpeg_compression(image, 6)


def test_jpeg_changes_pixels():
    corruptor = DocumentCorruptor()

    image = Image.fromarray(
        np.random.randint(
            0,
            256,
            (128, 128, 3),
            dtype=np.uint8,
        )
    )

    result = corruptor.jpeg_compression(
        image,
        severity=5,
    )

    assert not np.array_equal(
        np.array(image),
        np.array(result),
    )