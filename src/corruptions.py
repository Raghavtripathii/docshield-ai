from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from configs.corruption_config import (
    GAUSSIAN_NOISE_SIGMA,
    GAUSSIAN_BLUR_KERNEL,
    BRIGHTNESS_FACTOR,
    JPEG_QUALITY,
    ROTATION_ANGLE,
)


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

        sigma = GAUSSIAN_NOISE_SIGMA[severity]

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

        image_array = self._to_numpy(image)

        blurred = cv2.GaussianBlur(
            image_array,
            GAUSSIAN_BLUR_KERNEL[severity],
            sigmaX=0,
            sigmaY=0,
        )

        return self._to_pil(blurred)

    def brightness(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:

        self._validate_severity(severity)

        image_array = self._to_numpy(image).astype(np.float32)

        factor = BRIGHTNESS_FACTOR[severity]

        darkened = image_array * factor

        return self._to_pil(darkened)

    def jpeg_compression(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:

        self._validate_severity(severity)

        image_array = self._to_numpy(image)

        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY[severity],
        ]

        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR,
            ),
            encode_params,
        )

        if not success:
            raise RuntimeError(
                "Failed to encode JPEG image."
            )

        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        decoded = cv2.cvtColor(
            decoded,
            cv2.COLOR_BGR2RGB,
        )

        return self._to_pil(decoded)

    def rotation(
        self,
        image: Image.Image,
        severity: int,
    ) -> Image.Image:

        self._validate_severity(severity)

        angle = ROTATION_ANGLE[severity]

        image_array = self._to_numpy(image)

        height, width = image_array.shape[:2]

        center = (width / 2, height / 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0,
        )

        rotated = cv2.warpAffine(
            image_array,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        return self._to_pil(rotated)