from PIL import Image
import numpy as np

from src.corruptions import DocumentCorruptor
def test_gaussian_noise_returns_image():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (256, 256), "white")

    noisy = corruptor.gaussian_noise(
        image,
        severity=3,
        seed=42,
    )

    assert isinstance(noisy, Image.Image)

def test_gaussian_noise_preserves_size():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    noisy = corruptor.gaussian_noise(
        image,
        severity=2,
        seed=42,
    )

    assert noisy.size == image.size

def test_gaussian_noise_changes_pixels():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 128), "white")

    noisy = corruptor.gaussian_noise(
        image,
        severity=5,
        seed=42,
    )

    original = np.array(image)

    modified = np.array(noisy)

    assert not np.array_equal(
        original,
        modified,
    )