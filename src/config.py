"""Central configuration for DocShield experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared across DocShield experiments."""

    # Model and dataset
    model_name: str = "microsoft/layoutlmv3-base"
    dataset_name: str = "nielsr/funsd"

    # Input
    max_length: int = 512

    # Baseline training
    learning_rate: float = 5e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_train_epochs: int = 3
    weight_decay: float = 0.01

    # Reproducibility
    seed: int = 42

    # Outputs
    output_dir: str = "outputs/layoutlmv3-funsd"
    results_dir: str = "results"
    experiment_name: str = "layoutlmv3-funsd-baseline"


CONFIG = ExperimentConfig()