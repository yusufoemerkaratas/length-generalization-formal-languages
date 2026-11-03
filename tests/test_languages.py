import itertools
import random

import pytest

from flg.languages import LANGUAGES


# Second definitions of each language, written in a different way than
# is_member, used to check is_member on every short string.
def ref_parity(s):
    return s.count("1") % 2 == 0


def ref_equal_count(s):
    return s.count("a") == s.count("b")


def ref_anbn(s):
    m = len(s) // 2
    return len(s) > 0 and s == "a" * m + "b" * m


def ref_dyck1(s):
    while "()" in s:
        s = s.replace("()", "")
    return s == ""


REFERENCE = {
    "parity": ref_parity,
    "equal_count": ref_equal_count,
    "anbn": ref_anbn,
    "dyck1": ref_dyck1,
}

NAMES = sorted(LANGUAGES)


def lengths(lang, max_len):
    step = 2 if lang.even_lengths_only else 1
    return range(step, max_len + 1, step)


@pytest.mark.parametrize("name", NAMES)
def test_is_member_matches_reference_on_all_short_strings(name):
    lang, ref = LANGUAGES[name], REFERENCE[name]
    for n in range(1, 13):
        for chars in itertools.product(lang.alphabet, repeat=n):
            s = "".join(chars)
            assert lang.is_member(s) == ref(s), s


@pytest.mark.parametrize("name", NAMES)
def test_samples_have_correct_labels_and_lengths(name):
    lang = LANGUAGES[name]
    rng = random.Random(0)
    for n in lengths(lang, 200):
        for _ in range(20):
            pos = lang.sample_positive(n, rng)
            assert len(pos) == n and lang.is_member(pos)
            assert set(pos) <= set(lang.alphabet)
            for kind in lang.negative_kinds:
                neg = lang.sample_negative(n, rng, kind)
                assert len(neg) == n and not lang.is_member(neg)
                assert set(neg) <= set(lang.alphabet)


@pytest.mark.parametrize("name", ["anbn", "dyck1"])
def test_order_negatives_keep_symbol_counts(name):
    # An order negative has the same symbol counts as a positive string of the
    # same length, so a model that only counts symbols cannot tell them apart.
    lang = LANGUAGES[name]
    rng = random.Random(1)
    first = lang.alphabet[0]
    for n in lengths(lang, 200):
        for _ in range(20):
            neg = lang.sample_negative(n, rng, "order")
            assert neg.count(first) == n // 2


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_gives_same_strings(name):
    lang = LANGUAGES[name]

    def draw(seed):
        rng = random.Random(seed)
        out = [lang.sample_positive(40, rng)]
        out += [lang.sample_negative(40, rng, k) for k in lang.negative_kinds]
        return out

    assert draw(7) == draw(7)


@pytest.mark.parametrize("name", ["equal_count", "anbn", "dyck1"])
def test_odd_lengths_are_rejected(name):
    with pytest.raises(ValueError):
        LANGUAGES[name].sample_positive(5, random.Random(0))
