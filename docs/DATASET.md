# FUNSD Dataset Methodology

## Dataset

DocShield AI uses the FUNSD (Form Understanding in Noisy Scanned Documents) dataset for document understanding experiments.

The dataset contains scanned forms with word-level text, spatial bounding boxes, and semantic entity annotations.

## Task

The primary learning task is multimodal token classification.

Each document combines:

1. document image information,
2. textual word content,
3. two-dimensional spatial layout,
4. semantic entity labels.

This allows the model to learn from visual, textual, and spatial signals simultaneously.

## Dataset Splits

The dataset loader exposes the official FUNSD train and test splits.

The experiments preserve these splits rather than constructing a random replacement split.

Dataset sizes are validated programmatically and explored in `notebooks/01_funsd_eda.ipynb`.

## Input Representation

Each document contains:

- `image` — scanned document image,
- `words` — OCR-transcribed words,
- `bboxes` — normalized word bounding boxes,
- `ner_tags` — word-level entity annotations.

Bounding-box coordinates are validated to remain in the normalized 0–1000 coordinate system.

## Annotation Integrity

DocShield validates that each document contains equal numbers of:

- words,
- bounding boxes,
- entity labels.

Malformed spatial coordinates are rejected by validation utilities.

## Class Distribution

Entity-label frequencies are measured independently for the train and test splits.

Because document entity extraction may contain class imbalance, model evaluation will emphasize entity-level precision, recall, and F1 rather than relying on accuracy alone.

## Sequence Length

Document word counts are explored before tokenization.

Word count is not treated as equivalent to transformer token count because the LayoutLMv3 tokenizer can split individual words into multiple subword tokens.

Actual sequence truncation behavior will therefore be measured after processor-based tokenization.

## Multimodal Processing

DocShield uses LayoutLMv3 with externally supplied FUNSD words and bounding boxes.

OCR is therefore disabled in the processor so that the original dataset annotations remain aligned with their spatial coordinates and labels.

## Robustness Research

The clean FUNSD test split serves as the reference evaluation condition.

Controlled visual corruptions will later be applied to document images to measure how document-understanding performance changes under degraded visual conditions.

The underlying semantic annotations remain fixed unless a transformation explicitly requires corresponding geometric adjustment.

## Reproducibility

Dataset loading, validation, exploratory statistics, label mappings, and preprocessing behavior are implemented as reusable source modules rather than existing only inside notebooks.

Randomized experiments use the project's centralized seed configuration.