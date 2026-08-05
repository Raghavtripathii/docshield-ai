from PIL import Image

from src.augmentation import AugmentationPipeline


def test_pipeline_creation():

    pipeline = AugmentationPipeline()

    assert pipeline is not None


def test_probability_zero():

    pipeline = AugmentationPipeline(
        corruption_probability=0.0,
    )

    image = Image.new(
        "RGB",
        (64, 64),
        "white",
    )

    output = pipeline.apply(image)

    assert output == image


def test_probability_one():

    pipeline = AugmentationPipeline(
        corruption_probability=1.0,
        seed=42,
    )

    image = Image.new(
        "RGB",
        (64, 64),
        "white",
    )

    output = pipeline.apply(image)

    assert isinstance(
        output,
        Image.Image,
    )


def test_random_corruption():

    pipeline = AugmentationPipeline()

    corruption = pipeline._random_corruption()

    assert isinstance(
        corruption,
        str,
    )


def test_random_severity():

    pipeline = AugmentationPipeline()

    severity = pipeline._random_severity()

    assert severity in (
        1,
        2,
        3,
    )