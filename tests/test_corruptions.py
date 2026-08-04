from PIL import Image
import numpy as np
import pytest

from src.corruptions import DocumentCorruptor


def test_document_corruptor_creation():
    corruptor = DocumentCorruptor()
    assert corruptor is not None


def test_validate_severity_accepts_valid_values():
    corruptor = DocumentCorruptor()

    for severity in range(1, 6):
        corruptor._validate_severity(severity)


def test_validate_severity_rejects_invalid_values():
    corruptor = DocumentCorruptor()

    with pytest.raises(ValueError):
        corruptor._validate_severity(0)

    with pytest.raises(ValueError):
        corruptor._validate_severity(6)

    with pytest.raises(ValueError):
        corruptor._validate_severity(10)


def test_numpy_conversion():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    array = corruptor._to_numpy(image)

    assert isinstance(array, np.ndarray)
    assert array.shape == (64, 128, 3)


def test_pil_conversion():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (128, 64), "white")

    array = corruptor._to_numpy(image)

    recovered = corruptor._to_pil(array)

    assert isinstance(recovered, Image.Image)
    assert recovered.size == (128, 64)


def test_placeholder_methods():
    corruptor = DocumentCorruptor()

    image = Image.new("RGB", (32, 32), "white")

    with pytest.raises(NotImplementedError):
        corruptor.rotation(image, 3)