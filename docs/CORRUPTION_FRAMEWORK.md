# Document Corruption Framework

## Objective

This module provides a unified interface for applying synthetic document corruptions used throughout the robustness experiments.

The framework standardizes how corruptions are applied so every experiment follows the same pipeline.

---

## Supported Corruptions

The following corruption types are supported by the framework.

| Corruption | Commit |
|------------|--------|
| Gaussian Noise | 32 |
| Gaussian Blur | 33 |
| Brightness Degradation | 34 |
| JPEG Compression | 35 |
| Rotation | 36 |

---

## Severity Levels

Each corruption accepts an integer severity level.

| Severity | Meaning |
|----------|---------|
| 1 | Very Low |
| 2 | Low |
| 3 | Medium |
| 4 | High |
| 5 | Very High |

Any value outside this range raises a `ValueError`.

---

## Unified API

Every corruption is accessed through the same interface.

```python
from src.corruptions import DocumentCorruptor

corruptor = DocumentCorruptor()

result = corruptor.apply(
    image,
    corruptor.gaussian_noise,
    severity=3,
)
```

---

## Image Format

The framework automatically converts between:

- PIL Image
- NumPy array

This keeps compatibility with:

- LayoutLMv3
- Hugging Face Datasets
- OpenCV preprocessing

---

## Validation

The framework validates:

- severity range
- image conversion
- placeholder implementations

Unit tests are located in:

```
tests/test_corruptions.py
```

---

## Role in DocShield AI

The corruption framework serves as the common entry point for all robustness experiments.

Later commits implement individual corruption algorithms while preserving the same public API.