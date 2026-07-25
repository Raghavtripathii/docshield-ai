from src.data import load_funsd
from src.labels import build_label_maps, get_label_list


def test_label_list_exists():
    dataset = load_funsd()

    labels = get_label_list(dataset)

    assert len(labels) > 0


def test_label_maps_are_inverse():
    dataset = load_funsd()

    labels = get_label_list(dataset)
    id2label, label2id = build_label_maps(labels)

    for index, label in id2label.items():
        assert label2id[label] == index