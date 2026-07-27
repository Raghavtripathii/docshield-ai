"""Utilities for analysing neural-network parameter counts."""


def count_parameters(model) -> dict[str, int | float]:
    """Return total and trainable parameter statistics."""

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = total - trainable

    percentage = (
        (trainable / total) * 100
        if total
        else 0.0
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percentage": percentage,
    }