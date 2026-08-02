# LayoutLMv3 Baseline Experiment

## Objective

Establish a reproducible clean-document baseline for DocShield AI before introducing document corruption and robustness experiments.

---

## Model

- **Architecture:** LayoutLMv3
- **Checkpoint:** `microsoft/layoutlmv3-base`
- **Task:** Multimodal Token Classification
- **Dataset:** FUNSD

---

## Input Modalities

The model jointly processes:

- document image
- tokenized text
- normalized bounding boxes
- entity labels

---

## Training Protocol

The baseline model is trained using the versioned configuration stored in:

```text
configs/baseline.json
```

Randomness and reproducibility are controlled through the project's centralized reproducibility utilities.

---

## Evaluation

The baseline is evaluated using entity-level metrics:

- Precision
- Recall
- F1-score

Overall evaluation metrics are stored in:

```text
results/baseline_metrics.json
```

---

## Per-Class Performance

Per-class metrics are stored in:

```text
results/baseline_per_class.json
```

| Entity | Precision | Recall | F1 |
|---------|----------:|-------:|---:|
| QUESTION | 0.7728 | 0.8303 | 0.8006 |
| ANSWER | 0.7464 | 0.8335 | 0.7876 |
| HEADER | 0.3932 | 0.3866 | 0.3898 |

---

## Error Analysis

Prediction failures, confidence distributions, confusion patterns, and document-level errors are analyzed in:

```text
notebooks/06_error_analysis.ipynb
```

---

## Role in DocShield

This experiment establishes the clean-document reference baseline.

Future robustness experiments evaluate the impact of document corruptions by comparing their performance against this baseline.