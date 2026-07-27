import numpy as np

from src.metrics import decode_predictions


LABELS = [
    "O",
    "B-QUESTION",
    "I-QUESTION",
]


def test_decode_predictions_ignores_masked_tokens():
    predictions = np.array([
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    ])

    labels = np.array([
        [0, 1, -100]
    ])

    predicted, references = decode_predictions(
        predictions,
        labels,
        LABELS,
    )

    assert predicted == [
        ["O", "B-QUESTION"]
    ]

    assert references == [
        ["O", "B-QUESTION"]
    ]


def test_decode_predictions_preserves_sequence_length():
    predictions = np.array([
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    ])

    labels = np.array([
        [0, 1]
    ])

    predicted, references = decode_predictions(
        predictions,
        labels,
        LABELS,
    )

    assert len(predicted[0]) == 2
    assert len(references[0]) == 2