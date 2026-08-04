import pytest

from src.robustness_metrics import (
    average_robustness,
    best_case_performance,
    relative_robustness,
    robustness_drop,
    worst_case_performance,
)


def test_robustness_drop():

    drop = robustness_drop(
        clean_f1=0.90,
        corrupted_f1=0.75,
    )

    assert drop == pytest.approx(0.15)


def test_relative_robustness():

    score = relative_robustness(
        clean_f1=0.80,
        corrupted_f1=0.60,
    )

    assert score == pytest.approx(0.75)


def test_relative_robustness_zero_clean_score():

    with pytest.raises(ValueError):
        relative_robustness(
            clean_f1=0.0,
            corrupted_f1=0.50,
        )


def test_average_robustness():

    score = average_robustness(
        [
            0.80,
            0.75,
            0.70,
        ]
    )

    assert score == pytest.approx(0.75)


def test_average_empty():

    with pytest.raises(ValueError):
        average_robustness([])


def test_worst_case():

    score = worst_case_performance(
        [
            0.90,
            0.85,
            0.60,
            0.77,
        ]
    )

    assert score == 0.60


def test_worst_case_empty():

    with pytest.raises(ValueError):
        worst_case_performance([])


def test_best_case():

    score = best_case_performance(
        [
            0.90,
            0.85,
            0.60,
            0.77,
        ]
    )

    assert score == 0.90


def test_best_case_empty():

    with pytest.raises(ValueError):
        best_case_performance([])