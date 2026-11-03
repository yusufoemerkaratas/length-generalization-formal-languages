"""Sequence classifiers: a small transformer encoder and an LSTM.

Both take padded token ids (0 = padding, 1 and 2 = symbols) and the real
lengths, and return one logit per string.
"""

import math

import torch
from torch import nn

PAD, CLS = 0, 3
VOCAB_SIZE = 4
MAX_POSITIONS = 256


def sinusoidal_table(max_len: int, d: int) -> torch.Tensor:
    """Fixed sine/cosine position vectors as in "Attention Is All You Need"."""
    pos = torch.arange(max_len).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    table = torch.zeros(max_len, d)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq)
    return table


class TransformerClassifier(nn.Module):
    """Transformer encoder with a choice of positional encoding and attention.

    pe: "none", "sinusoidal" or "learned".
    causal=True: each position attends only to itself and earlier positions;
        the answer is read at the last real token.
    causal=False: full attention over the string plus a [CLS] token put in
        front; the answer is read at [CLS]. With pe="none" this model sees
        the string only as a multiset of symbols.
    """

    def __init__(self, pe="none", causal=True, d_model=64, n_layers=2, n_heads=4, d_ff=128):
        super().__init__()
        if pe not in ("none", "sinusoidal", "learned"):
            raise ValueError(pe)
        self.pe, self.causal = pe, causal
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        if pe == "learned":
            self.pos_embed = nn.Embedding(MAX_POSITIONS, d_model)
        elif pe == "sinusoidal":
            self.register_buffer("pos_table", sinusoidal_table(MAX_POSITIONS, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tokens, lengths):
        if not self.causal:
            cls = torch.full((tokens.shape[0], 1), CLS, dtype=tokens.dtype)
            tokens = torch.cat([cls, tokens], dim=1)
            lengths = lengths + 1
        seq_len = tokens.shape[1]
        if seq_len > MAX_POSITIONS:
            raise ValueError(f"string longer than {MAX_POSITIONS}")

        x = self.embed(tokens)
        positions = torch.arange(seq_len)
        if self.pe == "learned":
            x = x + self.pos_embed(positions)
        elif self.pe == "sinusoidal":
            x = x + self.pos_table[positions]

        if self.causal:
            # Padding comes after the real tokens, so with a causal mask the
            # real tokens never see it and no padding mask is needed.
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            h = self.encoder(x, mask=mask)
            h = self.norm(h)
            out = h[torch.arange(len(tokens)), lengths - 1]
        else:
            padding = torch.arange(seq_len).unsqueeze(0) >= lengths.unsqueeze(1)
            h = self.encoder(x, src_key_padding_mask=padding)
            out = self.norm(h)[:, 0]
        return self.head(out).squeeze(-1)


class LSTMClassifier(nn.Module):
    """One-layer LSTM; the answer is read at the last real token."""

    def __init__(self, d_model=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.lstm = nn.LSTM(d_model, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, tokens, lengths):
        h, _ = self.lstm(self.embed(tokens))
        out = h[torch.arange(len(tokens)), lengths - 1]
        return self.head(out).squeeze(-1)


def build_model(cfg: dict) -> nn.Module:
    """Build a model from a config entry such as
    {"type": "transformer", "pe": "none", "causal": true}."""
    if cfg["type"] == "transformer":
        return TransformerClassifier(pe=cfg["pe"], causal=cfg["causal"])
    if cfg["type"] == "lstm":
        return LSTMClassifier()
    raise ValueError(cfg["type"])
