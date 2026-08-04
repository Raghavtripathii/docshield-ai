from PIL import Image
import numpy as np
import pytest

from src.corruptions import DocumentCorruptor


def test_brightness_returns_image():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (256, 256), "white")

    result = corruptor.brightness(
        image,
        severity=3,
    )

    assert isinstance(result, Image.Image)


def test_brightness_preserves_size():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    result = corruptor.brightness(
        image,
        severity=4,
    )

    assert result.size == image.size


def test_brightness_invalid_severity():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (64, 64), "white")

    with pytest.raises(ValueError):
        corruptor.brightness(image, 0)

    with pytest.raises(ValueError):
        corruptor.brightness(image, 6)


def test_brightness_changes_pixels():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 128), "white")

    result = corruptor.brightness(
        image,
        severity=5,
    )

    assert not np.array_equal(
        np.array(image),
        np.array(result),
    )