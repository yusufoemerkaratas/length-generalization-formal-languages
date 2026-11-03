import torch

from flg.models import LSTMClassifier, TransformerClassifier


def ids(s, alphabet="ab"):
    return torch.tensor([[alphabet.index(c) + 1 for c in s]])


def logit(model, s):
    model.eval()
    with torch.no_grad():
        return model(ids(s), torch.tensor([len(s)])).item()


def test_bidirectional_nope_only_sees_symbol_counts():
    # aaabbb and aababb have the same symbols in a different order. Without
    # positional encoding and with full attention, the output must be equal.
    torch.manual_seed(0)
    model = TransformerClassifier(pe="none", causal=False)
    assert abs(logit(model, "aaabbb") - logit(model, "aababb")) < 1e-5


def test_order_is_visible_with_positions_or_causal_mask():
    torch.manual_seed(0)
    for model in (
        TransformerClassifier(pe="none", causal=True),
        TransformerClassifier(pe="sinusoidal", causal=False),
        TransformerClassifier(pe="learned", causal=False),
    ):
        assert abs(logit(model, "aaabbb") - logit(model, "aababb")) > 1e-5


def test_padding_does_not_change_the_output():
    torch.manual_seed(0)
    models = [
        TransformerClassifier(pe=pe, causal=causal)
        for pe in ("none", "sinusoidal", "learned")
        for causal in (True, False)
    ] + [LSTMClassifier()]
    padded = torch.tensor([[1, 2, 2, 0, 0, 0]])
    for model in models:
        model.eval()
        with torch.no_grad():
            short = model(ids("abb"), torch.tensor([3]))
            long = model(padded, torch.tensor([3]))
        assert torch.allclose(short, long, atol=1e-5), model
