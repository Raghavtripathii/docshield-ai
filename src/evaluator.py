from __future__ import annotations

from copy import deepcopy
from typing import Any


class Evaluator:

    def __init__(
        self,
        trainer,
        corruptor,
        dataset,
    ) -> None:

        self.trainer = trainer
        self.corruptor = corruptor
        self.dataset = dataset

    @staticmethod
    def _validate_corruption_name(
        corruptor,
        corruption_name: str,
    ) -> None:

        if corruption_name.startswith("_"):
            raise ValueError(
                f"{corruption_name} is not a public corruption."
            )

        if corruption_name == "apply":
            raise ValueError(
                
            )

        if not hasattr(
            corruptor,
            corruption_name,
        ):
            raise ValueError(
                f"Unknown corruption: {corruption_name}"
            )

    def _copy_dataset(self):

        return deepcopy(self.dataset)

    def _apply_corruption(
        self,
        corruption_name: str,
        severity: int,
    ):

        corruption = getattr(
            self.corruptor,
            corruption_name,
        )

        corrupted_dataset = self._copy_dataset()

        for sample in corrupted_dataset:

            sample["image"] = corruption(
                image=sample["image"],
                severity=severity,
            )

        return corrupted_dataset

    def evaluate(
        self,
        corruption_name: str,
        severities=(1, 2, 3),
    ) -> dict[int, dict[str, Any]]:

        self._validate_corruption_name(
            self.corruptor,
            corruption_name,
        )

        results = {}
        for severity in severities:

            corrupted_dataset = self._apply_corruption(
                corruption_name=corruption_name,
                severity=severity,
            )

            metrics = self.trainer.evaluate(
                eval_dataset=corrupted_dataset,
            )

            results[severity] = metrics

        return results

    def evaluate_clean(
        self,
    ) -> dict[str, Any]:

        return self.trainer.evaluate(
            eval_dataset=self.dataset,
        )

    def evaluate_all(
        self,
        corruptions: list[str],
        severities=(1, 2, 3),
    ) -> dict[str, dict[int, dict[str, Any]]]:

        benchmark = {}

        for corruption in corruptions:

            benchmark[corruption] = self.evaluate(
                corruption_name=corruption,
                severities=severities,
            )

        return benchmark

    def available_corruptions(
        self,
    ) -> list[str]:

        excluded = {
            "apply",
        }

        methods = []

        for name in dir(self.corruptor):

            if name.startswith("_"):
                continue

            if name in excluded:
                continue

            value = getattr(
                self.corruptor,
                name,
            )

            if callable(value):
                methods.append(name)

        return sorted(methods)