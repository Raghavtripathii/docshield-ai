from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class RobustnessReporter:

    def __init__(
        self,
        benchmark_results,
    ) -> None:

        self.results = benchmark_results

    def to_dataframe(self):

        rows = []

        for corruption, severities in self.results.items():

            for severity, metrics in severities.items():

                row = {
                    "corruption": corruption,
                    "severity": severity,
                }

                row.update(metrics)

                rows.append(row)

        return pd.DataFrame(rows)

    def save_csv(
        self,
        output_path,
    ):

        df = self.to_dataframe()

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
        )

    def plot_f1(
        self,
        output_path,
    ):

        df = self.to_dataframe()

        plt.figure(figsize=(8, 5))

        for corruption in df["corruption"].unique():

            subset = df[
                df["corruption"] == corruption
            ]

            plt.plot(
                subset["severity"],
                subset["eval_f1"],
                marker="o",
                label=corruption,
            )

        plt.xlabel("Severity")

        plt.ylabel("F1")

        plt.title("Robustness Benchmark")

        plt.grid(True)

        plt.legend()

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()