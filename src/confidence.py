from __future__ import annotations


class ConfidenceAnalyzer:

    def summarize(
        self,
        entities: list[dict],
    ) -> dict:

        if not entities:
            return {
                "average": 0.0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }

        scores = [
            entity["score"]
            for entity in entities
        ]

        average = sum(scores) / len(scores)

        high = sum(
            score >= 0.95
            for score in scores
        )

        medium = sum(
            0.80 <= score < 0.95
            for score in scores
        )

        low = sum(
            score < 0.80
            for score in scores
        )

        return {
            "average": average,
            "high": high,
            "medium": medium,
            "low": low,
        }