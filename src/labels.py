"""FUNSD label mapping utilities."""


def get_label_list(dataset) -> list[str]:
    """Extract the ordered FUNSD token-classification label names."""

    feature = dataset["train"].features["ner_tags"]

    if not hasattr(feature, "feature") or not hasattr(
        feature.feature,
        "names",
    ):
        raise ValueError(
            "FUNSD ner_tags feature does not expose label names."
        )

    return list(feature.feature.names)


def build_label_maps(
    label_list: list[str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Create integer-to-label and label-to-integer mappings."""

    if not label_list:
        raise ValueError("label_list must not be empty")

    id2label = {
        index: label
        for index, label in enumerate(label_list)
    }

    label2id = {
        label: index
        for index, label in id2label.items()
    }

    return id2label, label2id