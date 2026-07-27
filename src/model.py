"""LayoutLMv3 token-classification model construction."""

from transformers import (
    LayoutLMv3ForTokenClassification,
)

from src.config import CONFIG


def build_model(
    label_list: list[str],
):
    """Load pretrained LayoutLMv3 with a token-classification head."""

    if not label_list:
        raise ValueError(
            "label_list must contain at least one label"
        )

    id2label = {
        index: label
        for index, label in enumerate(label_list)
    }

    label2id = {
        label: index
        for index, label in id2label.items()
    }

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        CONFIG.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    return model