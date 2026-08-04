from __future__ import annotations

from typing import Iterable


def robustness_drop(
    clean_f1: float,
    corrupted_f1: float,
) -> float:

    return clean_f1 - corrupted_f1


def relative_robustness(
    clean_f1: float,
    corrupted_f1: float,
) -> float:

    if clean_f1 == 0:
        raise ValueError(
            "clean_f1 must be greater than zero."
        )

    return corrupted_f1 / clean_f1


def average_robustness(
    scores: Iterable[float],
) -> float:

    scores = list(scores)

    if not scores:
        raise ValueError(
            "scores cannot be empty."
        )

    return sum(scores) / len(scores)


def worst_case_performance(
    scores: Iterable[float],
) -> float:

    scores = list(scores)

    if not scores:
        raise ValueError(
            "scores cannot be empty."
        )

    return min(scores)


def best_case_performance(
    scores: Iterable[float],
) -> float:

    scores = list(scores)

    if not scores:
        raise ValueError(
            "scores cannot be empty."
        )

    return max(scores)
