# DocShield AI — Development Setup

This document describes the reproducible development and execution
environment used for DocShield AI.

## 1. Development Architecture

DocShield uses a hybrid development workflow.

### Local Windows Environment

The local laptop is used for:

- source-code development
- Git and GitHub
- repository management
- documentation
- configuration
- notebook editing
- lightweight development tasks
- reviewing experiment outputs

### Linux GPU Environment

GPU-backed Linux environments are used for:

- PyTorch model execution
- LayoutLMv3 fine-tuning
- FUNSD preprocessing and model evaluation
- robustness benchmarking
- corruption-aware training
- PEFT/LoRA experiments
- repeated deep-learning inference
- computationally expensive experiments

This separation keeps model training reproducible while avoiding
unnecessary sustained workloads on local hardware.

---

## 2. Python Version

The project uses:

```text
Python 3.11
```

The verified local development environment uses Python 3.11.15.

---

## 3. Local Environment Setup

Create a virtual environment using Python 3.11.

On Windows:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Verify the interpreter:

```powershell
python --version
```

The project should use Python 3.11.x.

---

## 4. Install Dependencies

Install development and machine-learning dependencies with:

```powershell
python -m pip install -r requirements-dev.txt
```

The development requirements include the primary project dependencies
from `requirements.txt`.

---

## 5. Local Windows Application Control

The current Windows development machine enforces an Application Control
policy that prevents some third-party native Python binaries from
executing inside the virtual environment.

Examples observed during environment validation include native
components used by:

- PyTorch
- pandas
- packages depending on pandas

This is an operating-system execution policy rather than a Python
dependency-resolution failure.

The project does not require disabling or weakening this security
policy.

Native ML execution affected by the policy is performed in the Linux
execution environment instead.

---

## 6. Local Development Responsibilities

The local environment remains the primary environment for:

```text
VS Code
Git
GitHub
Python source editing
configuration
documentation
notebook development
experiment design
code review
result inspection
```

Computationally expensive or native-ML-dependent operations are
executed remotely.

---

## 7. GPU Execution Environment

Deep-learning experiments are executed in a Linux environment with an
NVIDIA GPU.

The GPU environment must provide:

```text
Python 3.11
PyTorch
Transformers
Hugging Face Datasets
Accelerate
PEFT
OpenCV
scikit-learn
seqeval
```

Before model execution, verify GPU availability:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Model training should begin only after CUDA availability is confirmed.

---

## 8. Reproducibility

DocShield experiments use:

- controlled random seeds
- centralized experiment configuration
- documented dependencies
- explicit model identifiers
- explicit dataset identifiers
- recorded experiment outputs
- version-controlled source code

Experimental metrics are reported only after the corresponding
experiment has been executed and verified.

---

## 9. Testing

Tests that do not require blocked native ML libraries may be executed
locally.

ML-dependent tests are executed in the Linux environment.

The standard test command is:

```powershell
python -m pytest -q
```

A failed test caused by an operating-system Application Control block
must not be interpreted as a model or source-code failure.

---

## 10. Hardware Safety

Full Transformer fine-tuning and repeated deep-learning experiments are
not executed on the local development laptop.

GPU-intensive workloads including:

- LayoutLMv3 full fine-tuning
- corruption-aware retraining
- PEFT/LoRA training
- repeated robustness evaluation

are executed using remote GPU compute.

This keeps the local machine focused on development rather than
sustained deep-learning workloads.