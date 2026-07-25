"""Utilities for reproducible DocShield experiments."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed common random number generators."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the preferred PyTorch execution device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")