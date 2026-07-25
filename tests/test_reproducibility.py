import numpy as np
import torch

from src.reproducibility import get_device, set_seed


def test_numpy_seed_is_reproducible():
    set_seed(42)
    first = np.random.random(5)

    set_seed(42)
    second = np.random.random(5)

    assert np.array_equal(first, second)


def test_torch_seed_is_reproducible():
    set_seed(42)
    first = torch.rand(5)

    set_seed(42)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_device_is_valid():
    device = get_device()

    assert device.type in {"cpu", "cuda"}