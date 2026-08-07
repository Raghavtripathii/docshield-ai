from PIL import Image

from src.inference import InferenceEngine
from src.ocr import OCRExtractor

image = Image.open("assets/sample.jpeg").convert("RGB")

ocr = OCRExtractor()

words, boxes = ocr.extract(image)

engine = InferenceEngine()

result = engine.predict(
    image=image,
    words=words,
    boxes=boxes,
)

print(result["labels"][:10])
print(result["scores"][:10])