import torch

from src.model_stats import count_parameters


def test_parameter_counting():
    model = torch.nn.Linear(
        in_features=10,
        out_features=2,
    )

    stats = count_parameters(model)

    assert stats["total"] == 22
    assert stats["trainable"] == 22
    assert stats["frozen"] == 0
    assert stats["trainable_percentage"] == 100.0