from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
)


class InferenceEngine:

    def __init__(
        self,
        model_dir: str | Path = "outputs/robust_layoutlmv3",
    ) -> None:

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                "Trained model not found.\n\n"
                "Expected directory:\n"
                f"{self.model_dir}\n\n"
                "Run robust training first or download the trained model."
            )

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
    def predict(
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

        prediction_ids = (
            probabilities.argmax(dim=-1)
            .squeeze(0)
            .cpu()
            .tolist()
        )

        confidence_scores = (
            probabilities.max(dim=-1)
            .values.squeeze(0)
            .cpu()
            .tolist()
        )

        if isinstance(prediction_ids, int):
            prediction_ids = [prediction_ids]

        if isinstance(confidence_scores, float):
            confidence_scores = [confidence_scores]

        labels = [
            self.model.config.id2label[index]
            for index in prediction_ids
        ]

        return {
            "words": words,
            "boxes": boxes,
            "label_ids": prediction_ids,
            "labels": labels,
            "scores": confidence_scores,
        }

    def __repr__(self) -> str:

        return (
            f"InferenceEngine("
            f"model_dir='{self.model_dir}')"
        )