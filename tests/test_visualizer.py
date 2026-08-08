from PIL import Image

from src.visualizer import PredictionVisualizer


def test_visualizer_returns_image():

    image = Image.new(
        "RGB",
        (400, 200),
        "white",
    )

    visualizer = PredictionVisualizer()

    result = visualizer.draw(
        image=image,
        words=["Invoice"],
        boxes=[[100, 100, 300, 300]],
        labels=["B-QUESTION"],
        scores=[0.95],
    )

    assert isinstance(
        result,
        Image.Image,
    )


def test_visualizer_preserves_size():

    image = Image.new(
        "RGB",
        (400, 200),
        "white",
    )

    visualizer = PredictionVisualizer()

    result = visualizer.draw(
        image=image,
        words=["Invoice"],
        boxes=[[100, 100, 300, 300]],
        labels=["B-QUESTION"],
        scores=[0.95],
    )

    assert result.size == image.size