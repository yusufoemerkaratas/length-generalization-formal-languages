from collections import Counter

import pytest
import torch

from flg.data import encode, make_datasets
from flg.languages import LANGUAGES

CFG = {
    "train_range": [1, 40],
    "heldout_lengths": [12, 24, 36],
    "train_size": 2000,
    "val_size": 400,
    "test_size": 400,
    "test_bins": [[41, 80], [161, 200]],
    "test_seed": 1000,
}

NAMES = sorted(LANGUAGES)


@pytest.fixture(scope="module", params=NAMES)
def lang_and_data(request):
    lang = LANGUAGES[request.param]
    return lang, make_datasets(lang, CFG, seed=0)


def all_splits(data):
    yield "train", data["train"]
    yield "val", data["val"]
    for name, examples in data["test"].items():
        yield "test/" + name, examples


def test_labels_are_correct(lang_and_data):
    lang, data = lang_and_data
    for _, examples in all_splits(data):
        for s, label, _ in examples:
            assert lang.is_member(s) == (label == 1), s


def test_splits_are_balanced(lang_and_data):
    lang, data = lang_and_data
    for name, examples in all_splits(data):
        kinds = Counter(kind for _, _, kind in examples)
        half = len(examples) // 2
        assert kinds["pos"] == half, name
        for kind in lang.negative_kinds:
            assert kinds[kind] == half // len(lang.negative_kinds), name


def test_heldout_lengths_never_in_train_or_val(lang_and_data):
    _, data = lang_and_data
    for split in ("train", "val"):
        assert not {len(s) for s, _, _ in data[split]} & set(CFG["heldout_lengths"])


def test_lengths_are_in_their_bin(lang_and_data):
    lang, data = lang_and_data
    test = data["test"]
    assert {len(s) for s, _, _ in test["heldout"]} == set(CFG["heldout_lengths"])
    for lo, hi in CFG["test_bins"]:
        lengths = {len(s) for s, _, _ in test[f"{lo}-{hi}"]}
        assert min(lengths) >= lo and max(lengths) <= hi
    if lang.even_lengths_only:
        for _, examples in all_splits(data):
            assert all(len(s) % 2 == 0 for s, _, _ in examples)


def test_seed_changes_train_but_not_test():
    lang = LANGUAGES["dyck1"]
    a = make_datasets(lang, CFG, seed=0)
    b = make_datasets(lang, CFG, seed=0)
    c = make_datasets(lang, CFG, seed=1)
    assert a == b
    assert a["train"] != c["train"]
    assert a["test"] == c["test"]


def test_encode_pads_on_the_right():
    tokens, lengths, labels = encode([("ab", 1, "pos"), ("b", 0, "count")], "ab")
    assert tokens.tolist() == [[1, 2], [2, 0]]
    assert lengths.tolist() == [2, 1]
    assert torch.equal(labels, torch.tensor([1.0, 0.0]))
