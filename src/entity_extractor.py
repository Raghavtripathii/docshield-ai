from __future__ import annotations


class EntityExtractor:

    def extract(
        self,
        words: list[str],
        labels: list[str],
        scores: list[float],
    ) -> list[dict]:

        entities = []

        current_label = None
        current_words = []
        current_scores = []

        for word, label, score in zip(
            words,
            labels,
            scores,
        ):

            if label == "O":

                if current_label is not None:
                    entities.append(
                        {
                            "label": current_label,
                            "text": " ".join(current_words),
                            "score": (
                                sum(current_scores)
                                / len(current_scores)
                            ),
                        }
                    )

                current_label = None
                current_words = []
                current_scores = []

                continue

            prefix, entity = label.split("-", 1)

            if prefix == "B":

                if current_label is not None:
                    entities.append(
                        {
                            "label": current_label,
                            "text": " ".join(current_words),
                            "score": (
                                sum(current_scores)
                                / len(current_scores)
                            ),
                        }
                    )

                current_label = entity
                current_words = [word]
                current_scores = [score]

            elif (
                prefix == "I"
                and current_label == entity
            ):

                current_words.append(word)
                current_scores.append(score)

            else:

                if current_label is not None:
                    entities.append(
                        {
                            "label": current_label,
                            "text": " ".join(current_words),
                            "score": (
                                sum(current_scores)
                                / len(current_scores)
                            ),
                        }
                    )

                current_label = entity
                current_words = [word]
                current_scores = [score]

        if current_label is not None:

            entities.append(
                {
                    "label": current_label,
                    "text": " ".join(current_words),
                    "score": (
                        sum(current_scores)
                        / len(current_scores)
                    ),
                }
            )

        return entities