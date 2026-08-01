# LayoutLMv3 Baseline Experiment

## Objective

Establish a reproducible clean-document baseline for DocShield AI before introducing document corruption and robustness interventions.

---

## Model

- Architecture: LayoutLMv3
- Checkpoint: microsoft/layoutlmv3-base
- Task: Multimodal Token Classification
- Dataset: FUNSD

---

## Input Modalities

The model jointly receives:

- document image
- OCR tokens
- normalized bounding boxes
- entity labels

---

## Training Protocol

Training configuration is versioned in:

```
configs/baseline.json
```

Randomness is controlled through the centralized reproducibility utilities.

---

## Evaluation

Metrics include:

- Precision
- Recall
- F1-score
- Accuracy

Primary evaluation metric:

**Entity-level F1**

---

## Results

Overall evaluation metrics are stored in:

```
results/baseline_metrics.json
```

Per-class metrics are stored in:

```
results/baseline_per_class.json
```

---

## Error Analysis

Prediction failures are analyzed in:

```
notebooks/06_error_analysis.ipynb
```

---

## Role in DocShield

This experiment establishes the clean-document reference baseline.

Future robustness experiments compare corrupted-document performance against this baseline.