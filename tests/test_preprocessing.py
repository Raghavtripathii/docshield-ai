import pytest

from src.preprocessing import encode_document


def test_encode_document_rejects_misaligned_annotations():
    class DummyProcessor:
        pass

    with pytest.raises(
        ValueError,
        match="identical lengths",
    ):
        encode_document(
            processor=DummyProcessor(),
            image=None,
            words=["hello", "world"],
            boxes=[[0, 0, 10, 10]],
            labels=[0, 1],
        )