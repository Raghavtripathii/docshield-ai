from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CorruptionResult:
    """Container for a corrupted document image."""

    image: Image.Image
    corruption: str
    severity: int


class DocumentCorruptor:

    @staticmethod
    def _to_numpy(image: Image.Image) -> np.ndarray:
        """Convert PIL image → NumPy RGB array."""
        return np.array(image.convert("RGB"))

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        """Convert NumPy RGB array → PIL image."""
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
        """
        Apply any corruption using a unified API.
        """

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

    ####################################################################
    # Commit 32
    # Gaussian Noise
    ####################################################################

    def gaussian_noise(
        self,
        image: Image.Image,
        severity: int,
        seed: int | None = None,
    ) -> Image.Image:
        """
        Apply additive Gaussian noise.

        Parameters
        ----------
        image:
            Input PIL image.

        severity:
            Integer from 1 to 5.

        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        PIL.Image.Image
            Noisy image.
        """

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

    ####################################################################
    # Remaining corruptions
    ####################################################################

    def gaussian_blur(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:
        raise NotImplementedError(
            "Implemented in Commit 33."
        )

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