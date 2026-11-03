"""Four formal languages with exact membership tests and samplers.

Every sampler takes a `random.Random` object, so results depend only on its seed.
Negative strings are made by a small edit of a positive string, and each one has
a kind:
  "count": the edit changes how often a symbol occurs.
  "order": symbol counts are the same as in a positive string; only the order is wrong.
"""

import random


class Language:
    name = ""
    alphabet = ""
    # True if every string of odd length is a negative. Such languages are only
    # sampled at even lengths, otherwise the length alone would give the label.
    even_lengths_only = False
    negative_kinds = ()

    def is_member(self, s: str) -> bool:
        raise NotImplementedError

    def sample_positive(self, n: int, rng: random.Random) -> str:
        raise NotImplementedError

    def sample_negative(self, n: int, rng: random.Random, kind: str) -> str:
        raise NotImplementedError

    def _check_length(self, n: int) -> None:
        if n < 1 or (self.even_lengths_only and n % 2 == 1):
            raise ValueError(f"{self.name}: no positive strings of length {n}")


def _flip(s: str, i: int, a: str, b: str) -> str:
    """Replace s[i] by the other symbol of the pair (a, b)."""
    new = b if s[i] == a else a
    return s[:i] + new + s[i + 1:]


def _swap(s: str, i: int, j: int) -> str:
    chars = list(s)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


class Parity(Language):
    """Binary strings with an even number of 1s. A regular language."""

    name = "parity"
    alphabet = "01"
    negative_kinds = ("count",)

    def is_member(self, s):
        # Two-state automaton: the state is the parity of the 1s read so far.
        even = True
        for c in s:
            if c == "1":
                even = not even
        return even

    def sample_positive(self, n, rng):
        self._check_length(n)
        bits = [rng.choice("01") for _ in range(n - 1)]
        last = "1" if bits.count("1") % 2 == 1 else "0"
        return "".join(bits) + last

    def sample_negative(self, n, rng, kind="count"):
        assert kind == "count"
        s = self.sample_positive(n, rng)
        return _flip(s, rng.randrange(n), "0", "1")


class EqualCount(Language):
    """Strings over {a, b} with as many a as b. Not regular; order does not matter."""

    name = "equal_count"
    alphabet = "ab"
    even_lengths_only = True
    negative_kinds = ("count",)

    def is_member(self, s):
        diff = 0
        for c in s:
            diff += 1 if c == "a" else -1
        return diff == 0

    def sample_positive(self, n, rng):
        self._check_length(n)
        chars = ["a"] * (n // 2) + ["b"] * (n // 2)
        rng.shuffle(chars)
        return "".join(chars)

    def sample_negative(self, n, rng, kind="count"):
        assert kind == "count"
        s = self.sample_positive(n, rng)
        return _flip(s, rng.randrange(n), "a", "b")


class AnBn(Language):
    """Strings a^m b^m with m >= 1. Context-free, not regular.

    There is only one positive string per length.
    """

    name = "anbn"
    alphabet = "ab"
    even_lengths_only = True
    negative_kinds = ("count", "order")

    def is_member(self, s):
        i = 0
        while i < len(s) and s[i] == "a":
            i += 1
        num_a = i
        while i < len(s) and s[i] == "b":
            i += 1
        # All symbols read, and the number of b equals the number of a.
        return i == len(s) and len(s) - num_a == num_a

    def sample_positive(self, n, rng):
        self._check_length(n)
        return "a" * (n // 2) + "b" * (n // 2)

    def sample_negative(self, n, rng, kind):
        s = self.sample_positive(n, rng)
        m = n // 2
        if kind == "count":
            # a^(m+1) b^(m-1) or a^(m-1) b^(m+1)
            k = rng.choice([1, -1])
            return "a" * (m + k) + "b" * (m - k)
        if kind == "order":
            # Swap one a (first half) with one b (second half).
            return _swap(s, rng.randrange(m), m + rng.randrange(m))
        raise ValueError(kind)


class Dyck1(Language):
    """Balanced bracket strings over {(, )}. Context-free, not regular."""

    name = "dyck1"
    alphabet = "()"
    even_lengths_only = True
    negative_kinds = ("count", "order")

    def is_member(self, s):
        depth = 0
        for c in s:
            depth += 1 if c == "(" else -1
            if depth < 0:
                return False
        return depth == 0

    def sample_positive(self, n, rng):
        # Shuffle n/2 "(" and n/2 ")", then rotate the string so that it starts
        # right after the point where the running depth is lowest. After the
        # rotation the depth never goes below zero, so the string is balanced.
        # This does not give every balanced string the same probability.
        self._check_length(n)
        chars = ["("] * (n // 2) + [")"] * (n // 2)
        rng.shuffle(chars)
        depth, lowest, start = 0, 0, 0
        for i, c in enumerate(chars):
            depth += 1 if c == "(" else -1
            if depth < lowest:
                lowest, start = depth, i + 1
        return "".join(chars[start:] + chars[:start])

    def sample_negative(self, n, rng, kind):
        s = self.sample_positive(n, rng)
        if kind == "count":
            return _flip(s, rng.randrange(n), "(", ")")
        if kind == "order":
            # Swap an "(" with a later ")". Not every such swap breaks the string,
            # so try random swaps and fall back to one that always does: moving
            # the first ")" to position 0.
            opens = [i for i, c in enumerate(s) if c == "("]
            for _ in range(100):
                i = rng.choice(opens)
                later_closes = [j for j in range(i + 1, n) if s[j] == ")"]
                t = _swap(s, i, rng.choice(later_closes))
                if not self.is_member(t):
                    return t
            return _swap(s, 0, s.index(")"))
        raise ValueError(kind)


LANGUAGES = {lang.name: lang for lang in (Parity(), EqualCount(), AnBn(), Dyck1())}
