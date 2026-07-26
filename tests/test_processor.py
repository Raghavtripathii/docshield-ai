from src.processor import build_processor


def test_processor_uses_external_ocr_annotations():
    processor = build_processor()

    assert processor.image_processor.apply_ocr is False


def test_processor_exposes_tokenizer():
    processor = build_processor()

    assert processor.tokenizer is not None


def test_processor_exposes_image_processor():
    processor = build_processor()

    assert processor.image_processor is not None