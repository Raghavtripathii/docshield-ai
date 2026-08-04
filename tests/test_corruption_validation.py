from PIL import Image
import pytest

from src.corruptions import (
    CorruptionResult,
    DocumentCorruptor,
)


@pytest.fixture
def image():
    return Image.new("RGB", (256, 256), "white")


@pytest.fixture
def corruptor():
    return DocumentCorruptor()


@pytest.mark.parametrize(
    "method",
    [
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "jpeg_compression",
        "rotation",
    ],
)
def test_all_corruptions_return_image(
    corruptor,
    image,
    method,
):
    corruption = getattr(
        corruptor,
        method,
    )

    if method == "gaussian_noise":
        result = corruption(
            image,
            severity=3,
            seed=42,
        )
    else:
        result = corruption(
            image,
            severity=3,
        )

    assert isinstance(
        result,
        Image.Image,
    )


@pytest.mark.parametrize(
    "method",
    [
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "jpeg_compression",
        "rotation",
    ],
)
def test_all_corruptions_preserve_size(
    corruptor,
    image,
    method,
):
    corruption = getattr(
        corruptor,
        method,
    )

    if method == "gaussian_noise":
        result = corruption(
            image,
            severity=4,
            seed=42,
        )
    else:
        result = corruption(
            image,
            severity=4,
        )

    assert result.size == image.size


@pytest.mark.parametrize(
    "method",
    [
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "jpeg_compression",
        "rotation",
    ],
)
def test_invalid_severity(
    corruptor,
    image,
    method,
):
    corruption = getattr(
        corruptor,
        method,
    )

    with pytest.raises(ValueError):
        corruption(
            image,
            severity=0,
        )

    with pytest.raises(ValueError):
        corruption(
            image,
            severity=6,
        )


def test_apply_returns_corruption_result(
    corruptor,
    image,
):
    result = corruptor.apply(
        image=image,
        corruption=corruptor.gaussian_blur,
        severity=3,
    )

    assert isinstance(
        result,
        CorruptionResult,
    )

    assert result.corruption == "gaussian_blur"

    assert result.severity == 3

    assert isinstance(
        result.image,
        Image.Image,
    )