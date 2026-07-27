import torch

from src.config import CONFIG
from src.data import load_funsd
from src.dataset import FUNSDDataset
from src.processor import build_processor


def test_dataset_length_matches_funsd():
    dataset = load_funsd()
    processor = build_processor()

    wrapped = FUNSDDataset(
        dataset["train"],
        processor,
    )

    assert len(wrapped) == len(dataset["train"])


def test_dataset_returns_required_tensors():
    dataset = load_funsd()
    processor = build_processor()

    wrapped = FUNSDDataset(
        dataset["train"],
        processor,
    )

    item = wrapped[0]

    required = {
        "input_ids",
        "attention_mask",
        "bbox",
        "pixel_values",
        "labels",
    }

    assert required.issubset(item.keys())

    for key in required:
        assert isinstance(item[key], torch.Tensor)


def test_dataset_sequence_dimensions():
    dataset = load_funsd()
    processor = build_processor()

    item = FUNSDDataset(
        dataset["train"],
        processor,
    )[0]

    assert item["input_ids"].shape == (
        CONFIG.max_length,
    )

    assert item["bbox"].shape == (
        CONFIG.max_length,
        4,
    )

    assert item["labels"].shape == (
        CONFIG.max_length,
    )