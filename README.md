# DocShield AI

## Robust Multimodal Document Intelligence Under Real-World Degradation

DocShield AI is a research-oriented Document AI project investigating
the robustness of multimodal Transformers for Key Information Extraction
(KIE) from scanned documents.

The project fine-tunes LayoutLMv3 on FUNSD and systematically evaluates
how document degradation affects multimodal document understanding.

## Research Questions

1. How accurately can LayoutLMv3 perform KIE on clean documents?
2. How does realistic document degradation affect performance?
3. Which corruption types cause the largest performance loss?
4. Can corruption-aware training improve robustness?
5. Can PEFT/LoRA retain competitive performance while training fewer parameters?

## Planned Experiments

- LayoutLMv3 baseline fine-tuning
- Per-class model evaluation
- Systematic error analysis
- Document corruption benchmark
- Corruption severity analysis
- Corruption-aware training
- Full fine-tuning vs LoRA
- Parameter-efficiency analysis
- Inference latency benchmarking

## Status

Active development. Experimental results are reported only after they
have been reproduced and verified.