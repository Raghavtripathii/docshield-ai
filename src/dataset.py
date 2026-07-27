"""PyTorch dataset wrapper for LayoutLMv3 FUNSD experiments."""

from typing import Any

import torch
from torch.utils.data import Dataset

from src.preprocessing import encode_document


class FUNSDDataset(Dataset):
    """Lazily encode FUNSD documents for LayoutLMv3."""

    def __init__(
        self,
        split: Any,
        processor: Any,
    ) -> None:
        self.split = split
        self.processor = processor

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.split[index]

        encoding = encode_document(
            processor=self.processor,
            image=example["image"],
            words=example["words"],
            boxes=example["bboxes"],
            labels=example["ner_tags"],
        )

        return {
            key: value.squeeze(0)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }