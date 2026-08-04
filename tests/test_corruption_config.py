from configs.corruption_config import (
    GAUSSIAN_NOISE_SIGMA,
    GAUSSIAN_BLUR_KERNEL,
    BRIGHTNESS_FACTOR,
    JPEG_QUALITY,
    ROTATION_ANGLE,
)


def test_noise_levels():
    assert len(GAUSSIAN_NOISE_SIGMA) == 5


def test_blur_levels():
    assert len(GAUSSIAN_BLUR_KERNEL) == 5


def test_brightness_levels():
    assert len(BRIGHTNESS_FACTOR) == 5


def test_jpeg_levels():
    assert len(JPEG_QUALITY) == 5


def test_rotation_levels():
    assert len(ROTATION_ANGLE) == 5