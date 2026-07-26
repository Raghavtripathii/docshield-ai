from transformers import AutoProcessor

from src.config import CONFIG


def build_processor():
    """Create the LayoutLMv3 processor used by DocShield."""

    processor = AutoProcessor.from_pretrained(
        CONFIG.model_name,
        apply_ocr=False,
    )

    if processor.image_processor.apply_ocr:
        raise RuntimeError(
            "DocShield requires apply_ocr=False because FUNSD "
            "already provides aligned words and bounding boxes."
        )

    return processor