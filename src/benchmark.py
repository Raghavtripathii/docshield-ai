from __future__ import annotations

from typing import Any

from configs.benchmark import (
    BENCHMARK_NAME,
    CORRUPTIONS,
    SEVERITIES,
)


class BenchmarkRunner:

    def __init__(
        self,
        evaluator,
    ) -> None:

        self.evaluator = evaluator

    def run(
        self,
    ) -> dict[str, Any]:

        benchmark = {}

        for corruption in CORRUPTIONS:

            benchmark[corruption] = self.evaluator.evaluate(
                corruption_name=corruption,
                severities=SEVERITIES,
            )

        return benchmark

    def available_conditions(
        self,
    ) -> int:

        return (
            len(CORRUPTIONS)
            * len(SEVERITIES)
        )

    def corruption_names(
        self,
    ):

        return list(
            CORRUPTIONS
        )

    def severity_levels(
        self,
    ):

        return list(
            SEVERITIES
        )

    def benchmark_name(
        self,
    ):

        return BENCHMARK_NAME