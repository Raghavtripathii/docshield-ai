from PIL import Image

from src.entity_extractor import EntityExtractor


def test_entity_extractor_merges_bio_tokens():

    extractor = EntityExtractor()

    entities = extractor.extract(
        words=[
            "Invoice",
            "Date",
            "10",
            "December",
        ],
        labels=[
            "B-QUESTION",
            "I-QUESTION",
            "B-ANSWER",
            "I-ANSWER",
        ],
        scores=[
            0.95,
            0.90,
            0.98,
            0.96,
        ],
    )

    assert len(entities) == 2

    assert entities[0]["label"] == "QUESTION"
    assert entities[0]["text"] == "Invoice Date"

    assert entities[1]["label"] == "ANSWER"
    assert entities[1]["text"] == "10 December"


def test_entity_extractor_handles_outside_label():

    extractor = EntityExtractor()

    entities = extractor.extract(
        words=[
            "Invoice",
            "random",
            "Date",
        ],
        labels=[
            "B-QUESTION",
            "O",
            "B-QUESTION",
        ],
        scores=[
            0.95,
            0.50,
            0.92,
        ],
    )

    assert len(entities) == 2


def test_entity_extractor_returns_empty_for_empty_input():

    extractor = EntityExtractor()

    entities = extractor.extract(
        words=[],
        labels=[],
        scores=[],
    )

    assert entities == []