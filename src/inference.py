from __future__ import annotations

from pathlib import Path

from PIL import Image
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
)


class InferenceEngine:
    """
    Lightweight inference wrapper for the trained LayoutLMv3 model.

    The engine loads the processor and model from a local directory
    produced after robust training.
    """

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

        predictions = outputs.logits.argmax(-1)

        return {
            "predictions": predictions.squeeze().tolist(),
            "logits": outputs.logits.detach().cpu(),
        }

    def __repr__(self) -> str:

        return (
            f"InferenceEngine("
            f"model_dir='{self.model_dir}')"
        )