from __future__ import annotations
import numpy as np
import easyocr
from PIL import Image


class OCRExtractor:

    def __init__(self) -> None:
        self.reader = easyocr.Reader(
            ["en"],
            gpu=False,
        )

    def extract(
        self,
        image: Image.Image,
    ):

        results = self.reader.readtext(np.array(image))

        words = []
        boxes = []

        width, height = image.size

        for box, text, _ in results:

            if not text.strip():
                continue

            x1 = min(point[0] for point in box)
            y1 = min(point[1] for point in box)
            x2 = max(point[0] for point in box)
            y2 = max(point[1] for point in box)

            boxes.append(
                [
                    int(1000 * x1 / width),
                    int(1000 * y1 / height),
                    int(1000 * x2 / width),
                    int(1000 * y2 / height),
                ]
            )

            words.append(text)

        return words, boxes