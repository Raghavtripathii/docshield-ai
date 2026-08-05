from PIL import Image

from src.augmentation import AugmentationPipeline
from src.robust_dataset import RobustDataset


class DummyDataset:

    def __len__(self):

        return 4

    def __getitem__(
        self,
        index,
    ):

        return {
            "image": Image.new(
                "RGB",
                (64, 64),
                "white",
            ),
            "label": index,
        }


def test_dataset_creation():

    dataset = RobustDataset(
        DummyDataset()
    )

    assert dataset is not None


def test_dataset_length():

    dataset = RobustDataset(
        DummyDataset()
    )

    assert len(dataset) == 4


def test_getitem_returns_dictionary():

    dataset = RobustDataset(
        DummyDataset()
    )

    sample = dataset[0]

    assert isinstance(
        sample,
        dict,
    )


def test_contains_image():

    dataset = RobustDataset(
        DummyDataset()
    )

    sample = dataset[0]

    assert "image" in sample


def test_contains_label():

    dataset = RobustDataset(
        DummyDataset()
    )

    sample = dataset[0]

    assert "label" in sample


def test_augmentation_pipeline():

    pipeline = AugmentationPipeline(
        corruption_probability=1.0,
        seed=42,
    )

    dataset = RobustDataset(
        DummyDataset(),
        augmentation_pipeline=pipeline,
    )

    sample = dataset[0]

    assert isinstance(
        sample["image"],
        Image.Image,
    )


def test_without_augmentation():

    dataset = RobustDataset(
        DummyDataset(),
        augmentation_pipeline=None,
    )

    sample = dataset[0]

    assert isinstance(
        sample["image"],
        Image.Image,
    )