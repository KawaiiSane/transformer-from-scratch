"""
Encoder — Paper Section 3.1

The encoder maps an input sequence to a continuous representation (memory)
that the decoder attends to via cross-attention.

SHAPES:
    Input:  (batch, src_seq_len)  — token ids
    Output: (batch, src_seq_len, d_model)

Each EncoderLayer:
    x → SelfAttn → Add&Norm → FFN → Add&Norm

PAPER REF: Section 3.1, Figure 1 (left stack)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, create_padding_mask
from models.feed_forward import PositionwiseFeedForward
from models.positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Single encoder layer: Self-Attention + Feed-Forward with residual connections.

    Uses Post-LayerNorm (paper original): LayerNorm(x + Sublayer(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len) padding mask

        Returns:
            output: (batch, src_len, d_model)
            attn_weights: (batch, n_heads, src_len, src_len)
        """
        # Sub-layer 1: Multi-head self-attention
        attn_out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))  # residual + norm

        # Sub-layer 2: Position-wise feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class Encoder(nn.Module):
    """Full encoder stack: Embedding + Positional Encoding + N layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int,
        pad_idx: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            src: (batch, src_len) token ids

        Returns:
            memory: (batch, src_len, d_model)
            all_attn_weights: list of (batch, n_heads, src_len, src_len)
        """
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_idx)

        # Embed and scale (paper: multiply embeddings by sqrt(d_model))
        x = self.embedding(src) * math.sqrt(self.d_model)  # (B, L, d_model)
        x = self.pos_encoding(x)

        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            all_attn_weights.append(attn_weights)

        return x, all_attn_weights
