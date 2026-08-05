from __future__ import annotations

from torch.utils.data import Dataset


class RobustDataset(Dataset):

    def __init__(
        self,
        dataset,
        augmentation_pipeline=None,
    ) -> None:

        self.dataset = dataset
        self.augmentation = augmentation_pipeline

    def __len__(self):

        return len(self.dataset)

    def __getitem__(
        self,
        index,
    ):

        sample = dict(self.dataset[index])

        if (
            self.augmentation is not None
            and "image" in sample
        ):
            sample["image"] = self.augmentation.apply(
                sample["image"]
            )

        return sample