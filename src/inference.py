from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
)

from src.ocr import OCRExtractor


class InferenceEngine:

    def __init__(
        self,
        model_dir: str | Path = "outputs/robust_layoutlmv3",
    ) -> None:

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_dir}"
            )

        self.ocr = OCRExtractor()

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            apply_ocr=False,
        )

        self.model = (
            LayoutLMv3ForTokenClassification.from_pretrained(
                self.model_dir,
            )
        )

        self.model.eval()

    @torch.inference_mode()
    def predict_words(
        self,
        image: Image.Image,
        words: list[str],
        boxes: list[list[int]],
    ) -> dict:

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
        )

        outputs = self.model(**encoding)

        probabilities = outputs.logits.softmax(dim=-1)

        label_ids = (
            probabilities.argmax(dim=-1)
            .squeeze(0)
            .cpu()
            .tolist()
        )

        scores = (
            probabilities.max(dim=-1)
            .values.squeeze(0)
            .cpu()
            .tolist()
        )

        if isinstance(label_ids, int):
            label_ids = [label_ids]

        if isinstance(scores, float):
            scores = [scores]

        labels = [
            self.model.config.id2label[label_id]
            for label_id in label_ids
        ]

        return {
            "words": words,
            "boxes": boxes,
            "label_ids": label_ids,
            "labels": labels,
            "scores": scores,
            "image_size": image.size,
        }

    def predict_image(
        self,
        image: Image.Image,
    ) -> dict:

        words, boxes = self.ocr.extract(image)

        if not words:
            raise RuntimeError(
                "No text detected."
            )

        return self.predict_words(
            image=image,
            words=words,
            boxes=boxes,
        )

    def predict_path(
        self,
        image_path: str | Path,
    ) -> dict:

        image = Image.open(image_path).convert("RGB")

        return self.predict_image(image)

    def __repr__(self) -> str:

        return (
            f"InferenceEngine(model_dir='{self.model_dir}')"
        )