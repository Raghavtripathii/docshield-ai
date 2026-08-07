from __future__ import annotations

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from src.colors import LABEL_COLORS


class PredictionVisualizer:

    def draw(
        self,
        image: Image.Image,
        words,
        boxes,
        labels,
        scores,
    ):

        image = image.copy()

        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(
                "arial.ttf",
                18,
            )
        except OSError:
            font = ImageFont.load_default()

        width, height = image.size

        for word, box, label, score in zip(
            words,
            boxes,
            labels,
            scores,
        ):

            x1 = int(box[0] * width / 1000)
            y1 = int(box[1] * height / 1000)
            x2 = int(box[2] * width / 1000)
            y2 = int(box[3] * height / 1000)

            entity = (
                label.split("-")[-1]
                if "-" in label
                else label
            )

            color = LABEL_COLORS.get(
                entity,
                LABEL_COLORS["OTHER"],
            )

            draw.rectangle(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                outline=color,
                width=3,
            )

            draw.rectangle(
                (
                    x1,
                    y1 - 22,
                    x1 + 120,
                    y1,
                ),
                fill=color,
            )

            draw.text(
                (
                    x1 + 3,
                    y1 - 20,
                ),
                f"{entity} {score:.2f}",
                fill="white",
                font=font,
            )

        return image