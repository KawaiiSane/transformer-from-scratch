"""Smoke tests for Phase 1 modules and full model forward pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import config
from models.attention import (
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask,
    scaled_dot_product_attention,
)
from models.encoder import Encoder
from models.positional_encoding import PositionalEncoding, plot_positional_encoding
from models.transformer import Transformer


def test_scaled_dot_product_attention():
    Q = torch.randn(2, 4, 8, 32)
    K = torch.randn(2, 4, 8, 32)
    V = torch.randn(2, 4, 8, 32)
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 4, 8, 32)
    assert w.shape == (2, 4, 8, 8)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5)


def test_causal_mask():
    mask = create_causal_mask(5)
    Q = K = V = torch.randn(1, 1, 5, 8)
    _, w = scaled_dot_product_attention(Q, K, V, mask=mask)
    upper = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    assert (w[0, 0][upper] < 1e-6).all()


def test_positional_encoding():
    pe = PositionalEncoding(config.D_MODEL, max_len=50)
    x = torch.randn(2, 10, config.D_MODEL)
    out = pe(x)
    assert out.shape == x.shape


def test_multi_head_attention():
    mha = MultiHeadAttention(config.D_MODEL, config.N_HEADS)
    x = torch.randn(2, 6, config.D_MODEL)
    out, w = mha(x, x, x)
    assert out.shape == x.shape
    assert w.shape == (2, config.N_HEADS, 6, 6)


def test_encoder():
    enc = Encoder(
        vocab_size=20,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        n_layers=2,
        max_seq_len=config.MAX_SEQ_LEN,
        pad_idx=0,
    )
    src = torch.randint(1, 20, (2, 10))
    out, attn = enc(src)
    assert out.shape == (2, 10, config.D_MODEL)
    assert len(attn) == 2


def test_transformer_forward():
    model = Transformer(
        src_vocab_size=30,
        tgt_vocab_size=30,
        pad_idx=0,
        sos_idx=1,
        eos_idx=2,
    )
    src = torch.randint(3, 30, (2, 8))
    tgt = torch.randint(3, 30, (2, 8))
    logits, attn = model(src, tgt)
    assert logits.shape == (2, 8, 30)
    assert "encoder" in attn


if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_causal_mask()
    test_positional_encoding()
    test_multi_head_attention()
    test_encoder()
    test_transformer_forward()
    print("All smoke tests passed.")
