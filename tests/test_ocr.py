from PIL import Image

from src.ocr import OCRExtractor


def test_ocr_returns_lists():

    image = Image.new(
        "RGB",
        (300, 100),
        "white",
    )

    extractor = OCRExtractor()

    words, boxes = extractor.extract(image)

    assert isinstance(words, list)
    assert isinstance(boxes, list)


def test_ocr_words_and_boxes_match():

    image = Image.new(
        "RGB",
        (300, 100),
        "white",
    )

    extractor = OCRExtractor()

    words, boxes = extractor.extract(image)

    assert len(words) == len(boxes)