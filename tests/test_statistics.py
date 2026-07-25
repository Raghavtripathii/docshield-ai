from src.data import load_funsd
from src.statistics import document_statistics, label_distribution


def test_document_statistics_has_expected_columns():
    dataset = load_funsd()

    frame = document_statistics(dataset)

    expected = {
        "split",
        "document_id",
        "word_count",
        "image_width",
        "image_height",
    }

    assert expected.issubset(frame.columns)


def test_label_distribution_is_not_empty():
    dataset = load_funsd()

    counts = label_distribution(dataset, "train")

    assert sum(counts.values()) > 0