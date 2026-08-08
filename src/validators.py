from __future__ import annotations

from PIL import Image


class ImageValidator:

    MAX_PIXELS = 25_000_000

    def validate(
        self,
        image: Image.Image,
    ) -> None:

        if image is None:
            raise ValueError(
                "No image provided."
            )

        if image.mode != "RGB":
            raise ValueError(
                "Image must be RGB."
            )

        width, height = image.size

        if width == 0 or height == 0:
            raise ValueError(
                "Invalid image dimensions."
            )

        if width * height > self.MAX_PIXELS:
            raise ValueError(
                "Image is too large."
            )