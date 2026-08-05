from __future__ import annotations

import random

from configs.benchmark import CORRUPTIONS
from src.corruptions import DocumentCorruptor


class AugmentationPipeline:

    def __init__(
        self,
        corruption_probability: float = 0.5,
        severities=(1, 2, 3),
        seed: int | None = None,
    ) -> None:

        self.corruption_probability = corruption_probability
        self.severities = tuple(severities)
        self.corruptor = DocumentCorruptor()
        self.random = random.Random(seed)

    def _random_corruption(self):

        return self.random.choice(CORRUPTIONS)

    def _random_severity(self):

        return self.random.choice(self.severities)

    def apply(
        self,
        image,
    ):

        if self.random.random() > self.corruption_probability:
            return image

        corruption_name = self._random_corruption()

        severity = self._random_severity()

        corruption = getattr(
            self.corruptor,
            corruption_name,
        )

        kwargs = {
            "image": image,
            "severity": severity,
        }

        if corruption_name == "gaussian_noise":
            kwargs["seed"] = self.random.randint(
                0,
                2**32 - 1,
            )

        return corruption(**kwargs)