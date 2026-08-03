from PIL import Image
import numpy as np
import pytest

from src.corruptions import DocumentCorruptor


def test_gaussian_blur_returns_image():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (256, 256), "white")

    blurred = corruptor.gaussian_blur(
        image,
        severity=3,
    )

    assert isinstance(blurred, Image.Image)


def test_gaussian_blur_preserves_size():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    blurred = corruptor.gaussian_blur(
        image,
        severity=4,
    )

    assert blurred.size == image.size


def test_gaussian_blur_invalid_severity():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (64, 64), "white")

    with pytest.raises(ValueError):
        corruptor.gaussian_blur(
            image,
            severity=0,
        )

    with pytest.raises(ValueError):
        corruptor.gaussian_blur(
            image,
            severity=6,
        )


def test_gaussian_blur_changes_pixels():
    corruptor = DocumentCorruptor()

    image = Image.fromarray(
        np.random.randint(
            0,
            256,
            (128, 128, 3),
            dtype=np.uint8,
        )
    )

    blurred = corruptor.gaussian_blur(
        image,
        severity=5,
    )

    assert not np.array_equal(
        np.array(image),
        np.array(blurred),
    )