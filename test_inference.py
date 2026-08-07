from pathlib import Path

from src.inference import InferenceEngine

engine = InferenceEngine()

result = engine.predict_path(
    Path("assets/sample.jpeg")
)

print(result["labels"][:10])
print(result["scores"][:10])