"""Balanced datasets of labelled strings, split by length.

An example is a tuple (string, label, kind) with label 1 for positive strings
and kind one of "pos", "count", "order" (see languages.py).
"""

import random

import torch

from flg.languages import Language


def allowed_lengths(lang: Language, lo: int, hi: int, exclude=()) -> list[int]:
    """Lengths in [lo, hi] that have positive strings, minus `exclude`."""
    step_ok = (lambda n: n % 2 == 0) if lang.even_lengths_only else (lambda n: True)
    return [n for n in range(max(lo, 1), hi + 1) if step_ok(n) and n not in exclude]


def make_split(lang: Language, lengths: list[int], size: int, rng: random.Random) -> list:
    """Make `size` examples, half positive, negatives split evenly across kinds.

    Examples are made in groups. A group has one negative of each kind and the
    same number of positives, so every split is exactly balanced. Group i uses
    length lengths[i % len(lengths)], so all lengths are (almost) equally common.
    """
    kinds = lang.negative_kinds
    group_size = 2 * len(kinds)
    if size % group_size != 0:
        raise ValueError(f"size {size} must be a multiple of {group_size}")
    examples = []
    for i in range(size // group_size):
        n = lengths[i % len(lengths)]
        for kind in kinds:
            examples.append((lang.sample_positive(n, rng), 1, "pos"))
            examples.append((lang.sample_negative(n, rng, kind), 0, kind))
    rng.shuffle(examples)
    return examples


def bin_name(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def make_datasets(lang: Language, cfg: dict, seed: int) -> dict:
    """Build train, val and test splits from the `data` section of a config.

    Train and val depend on `seed`. Test sets use cfg["test_seed"], so every
    model and every seed is evaluated on the same test strings.
    """
    lo, hi = cfg["train_range"]
    heldout = cfg["heldout_lengths"]
    train_lengths = allowed_lengths(lang, lo, hi, exclude=heldout)
    heldout_lengths = allowed_lengths(lang, lo, hi)
    heldout_lengths = [n for n in heldout_lengths if n in heldout]

    rng = random.Random(seed)
    train = make_split(lang, train_lengths, cfg["train_size"], rng)
    val = make_split(lang, train_lengths, cfg["val_size"], rng)

    test_rng = random.Random(cfg["test_seed"])
    test = {
        "train": make_split(lang, train_lengths, cfg["test_size"], test_rng),
        "heldout": make_split(lang, heldout_lengths, cfg["test_size"], test_rng),
    }
    for b_lo, b_hi in cfg["test_bins"]:
        lengths = allowed_lengths(lang, b_lo, b_hi)
        test[bin_name(b_lo, b_hi)] = make_split(lang, lengths, cfg["test_size"], test_rng)
    return {"train": train, "val": val, "test": test}


def encode(examples: list, alphabet: str):
    """Turn examples into padded tensors.

    Token ids: 0 is padding, 1 and 2 are the two symbols of the alphabet.
    Strings are padded on the right; `lengths` holds the real lengths.
    """
    to_id = {c: i + 1 for i, c in enumerate(alphabet)}
    max_len = max(len(s) for s, _, _ in examples)
    tokens = torch.zeros(len(examples), max_len, dtype=torch.long)
    for row, (s, _, _) in enumerate(examples):
        tokens[row, : len(s)] = torch.tensor([to_id[c] for c in s])
    lengths = torch.tensor([len(s) for s, _, _ in examples])
    labels = torch.tensor([label for _, label, _ in examples], dtype=torch.float)
    return tokens, lengths, labels
