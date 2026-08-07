from __future__ import annotations

import csv
import io


class PredictionExporter:

    def to_csv(
        self,
        entities: list[dict],
    ) -> str:

        buffer = io.StringIO()

        writer = csv.writer(buffer)

        writer.writerow(
            [
                "Label",
                "Text",
                "Confidence",
            ]
        )

        for entity in entities:

            writer.writerow(
                [
                    entity["label"],
                    entity["text"],
                    f"{entity['score']:.4f}",
                ]
            )

        return buffer.getvalue()