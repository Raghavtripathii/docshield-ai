from PIL import Image

from src.augmentation import AugmentationPipeline


def test_custom_severities():

    pipeline = AugmentationPipeline(
        severities=(2,),
        seed=42,
    )

    severity = pipeline._random_severity()

    assert severity == 2


def test_seed_reproducibility():

    pipeline1 = AugmentationPipeline(
        corruption_probability=1.0,
        seed=123,
    )

    pipeline2 = AugmentationPipeline(
        corruption_probability=1.0,
        seed=123,
    )

    image = Image.new(
        "RGB",
        (64, 64),
        "white",
    )

    output1 = pipeline1.apply(image)

    output2 = pipeline2.apply(image)

    assert output1.tobytes() == output2.tobytes()


def test_output_size_preserved():

    pipeline = AugmentationPipeline(
        corruption_probability=1.0,
        seed=42,
    )

    image = Image.new(
        "RGB",
        (128, 256),
        "white",
    )

    output = pipeline.apply(image)

    assert output.size == image.size


def test_output_type():

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


def test_probability_boundary():

    pipeline = AugmentationPipeline(
        corruption_probability=0.0,
    )

    image = Image.new(
        "RGB",
        (64, 64),
        "white",
    )

    output = pipeline.apply(image)

    assert output is image