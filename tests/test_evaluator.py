import pytest

from src.evaluator import Evaluator


class DummyTrainer:

    def __init__(self):
        self.calls = []

    def evaluate(self, eval_dataset):

        self.calls.append(eval_dataset)

        return {
            "eval_loss": 0.5,
            "eval_f1": 0.80,
        }


class DummyCorruptor:

    def gaussian_noise(
        self,
        image,
        severity,
    ):
        return f"{image}-noise-{severity}"

    def gaussian_blur(
        self,
        image,
        severity,
    ):
        return f"{image}-blur-{severity}"

    def brightness(
        self,
        image,
        severity,
    ):
        return f"{image}-brightness-{severity}"

    def jpeg_compression(
        self,
        image,
        severity,
    ):
        return f"{image}-jpeg-{severity}"

    def rotation(
        self,
        image,
        severity,
    ):
        return f"{image}-rotation-{severity}"

    def apply(
        self,
        image,
        corruption,
        severity,
    ):
        return corruption(
            image=image,
            severity=severity,
        )


@pytest.fixture
def trainer():
    return DummyTrainer()


@pytest.fixture
def corruptor():
    return DummyCorruptor()


@pytest.fixture
def dataset():

    return [
        {
            "image": "doc1",
            "label": 0,
        },
        {
            "image": "doc2",
            "label": 1,
        },
    ]


@pytest.fixture
def evaluator(
    trainer,
    corruptor,
    dataset,
):

    return Evaluator(
        trainer=trainer,
        corruptor=corruptor,
        dataset=dataset,
    )


def test_evaluator_creation(
    evaluator,
):
    assert evaluator is not None


def test_available_corruptions(
    evaluator,
):

    corruptions = evaluator.available_corruptions()

    assert "gaussian_noise" in corruptions
    assert "gaussian_blur" in corruptions
    assert "brightness" in corruptions
    assert "jpeg_compression" in corruptions
    assert "rotation" in corruptions


def test_evaluate_clean(
    evaluator,
):

    metrics = evaluator.evaluate_clean()

    assert "eval_loss" in metrics
    assert "eval_f1" in metrics


def test_single_corruption(
    evaluator,
):

    results = evaluator.evaluate(
        corruption_name="gaussian_noise",
        severities=[1],
    )

    assert 1 in results
    assert "eval_f1" in results[1]


def test_multiple_severities(
    evaluator,
):

    results = evaluator.evaluate(
        corruption_name="gaussian_noise",
        severities=[1, 2, 3],
    )

    assert len(results) == 3


def test_evaluate_all(
    evaluator,
):

    benchmark = evaluator.evaluate_all(
        [
            "gaussian_noise",
            "gaussian_blur",
        ],
        severities=[1],
    )

    assert "gaussian_noise" in benchmark
    assert "gaussian_blur" in benchmark


def test_invalid_corruption(
    evaluator,
):

    with pytest.raises(ValueError):
        evaluator.evaluate(
            corruption_name="invalid",
        )


def test_apply_corruption(
    evaluator,
    trainer,
):

    evaluator.evaluate(
        corruption_name="rotation",
        severities=[2],
    )

    dataset = trainer.calls[0]

    assert dataset[0]["image"] == "doc1-rotation-2"


def test_private_method_rejected(
    trainer,
    dataset,
):

    class FakeCorruptor:

        def gaussian_noise(
            self,
            image,
            severity,
        ):
            return image

        def _hidden(
            self,
            image,
            severity,
        ):
            return image

    evaluator = Evaluator(
        trainer=trainer,
        corruptor=FakeCorruptor(),
        dataset=dataset,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            "_hidden",
        )


def test_apply_method_rejected(
    trainer,
    dataset,
):

    class FakeCorruptor:

        def gaussian_noise(
            self,
            image,
            severity,
        ):
            return image

        def apply(
            self,
            image,
            corruption,
            severity,
        ):
            return image

    evaluator = Evaluator(
        trainer=trainer,
        corruptor=FakeCorruptor(),
        dataset=dataset,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            "apply",
        )