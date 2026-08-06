from pathlib import Path

import pytest

from src.inference import InferenceEngine


def test_missing_model_directory():

    with pytest.raises(FileNotFoundError):

        InferenceEngine(
            model_dir=Path("does_not_exist")
        )