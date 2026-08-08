from pathlib import Path


def test_app_exists():

    app = Path("app.py")

    assert app.exists()


def test_model_directory_exists():

    model_dir = Path(
        "outputs/robust_layoutlmv3"
    )

    assert model_dir.exists()