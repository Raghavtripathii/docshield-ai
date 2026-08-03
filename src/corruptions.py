from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CorruptionResult:
    image: Image.Image
    corruption: str
    severity: int


class DocumentCorruptor:

    @staticmethod
    def _to_numpy(image: Image.Image) -> np.ndarray:
        return np.array(image.convert("RGB"))

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

    @staticmethod
    def _validate_severity(
        severity: int,
        minimum: int = 1,
        maximum: int = 5,
    ) -> None:
        if not minimum <= severity <= maximum:
            raise ValueError(
                f"Severity must be between {minimum} and {maximum}."
            )

    def apply(
        self,
        image: Image.Image,
        corruption: Callable[..., Image.Image],
        severity: int,
    ) -> CorruptionResult:

        self._validate_severity(severity)

        corrupted = corruption(
            image=image,
            severity=severity,
        )

        return CorruptionResult(
            image=corrupted,
            corruption=corruption.__name__,
            severity=severity,
        )

    def gaussian_noise(
        self,
        image: Image.Image,
        severity: int,
        seed: int | None = None,
    ) -> Image.Image:

        self._validate_severity(severity)

        if seed is not None:
            np.random.seed(seed)

        image_array = self._to_numpy(image).astype(np.float32)

        sigma_levels = {
            1: 5,
            2: 10,
            3: 15,
            4: 25,
            5: 35,
        }

        sigma = sigma_levels[severity]

        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=image_array.shape,
        )

        noisy_image = image_array + noise

        return self._to_pil(noisy_image)

    def gaussian_blur(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:

        self._validate_severity(severity)

        kernel_sizes = {
            1: (3, 3),
            2: (5, 5),
            3: (7, 7),
            4: (9, 9),
            5: (11, 11),
        }

        image_array = self._to_numpy(image)

        blurred = cv2.GaussianBlur(
            image_array,
            kernel_sizes[severity],
            sigmaX=0,
            sigmaY=0,
        )

        return self._to_pil(blurred)

    def brightness(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:
        raise NotImplementedError(
            "Implemented in Commit 34."
        )

    def jpeg_compression(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:
        raise NotImplementedError(
            "Implemented in Commit 35."
        )

    def rotation(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:
        raise NotImplementedError(
            "Implemented in Commit 36."
        )