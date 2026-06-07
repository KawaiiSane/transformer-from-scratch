"""
Decoder — Paper Section 3.1

The decoder generates the output sequence one token at a time, attending to:
1. Previously generated tokens (masked self-attention)
2. Encoder output (cross-attention)

SHAPES:
    Input:  tgt (batch, tgt_len), memory (batch, src_len, d_model)
    Output: (batch, tgt_len, d_model)

PAPER REF: Section 3.1, Figure 1 (right stack)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention, create_causal_mask, create_padding_mask
from models.feed_forward import PositionwiseFeedForward
from models.positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Single decoder layer:
        Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm
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
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, tgt_len, d_model)
            memory: (batch, src_len, d_model) — encoder output
            tgt_mask: Combined causal + padding mask for self-attention
            memory_mask: Padding mask for cross-attention keys

        Returns:
            output, self_attn_weights, cross_attn_weights
        """
        # Masked self-attention (causal: can't look at future tokens)
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention: Q from decoder, K/V from encoder memory
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x, memory, memory, mask=memory_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """Full decoder stack."""

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
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            tgt: (batch, tgt_len) target token ids
            memory: (batch, src_len, d_model)

        Returns:
            output, self_attn_weights_list, cross_attn_weights_list
        """
        if tgt_mask is None:
            tgt_pad_mask = create_padding_mask(tgt, self.pad_idx)
            causal_mask = create_causal_mask(tgt.size(1), device=tgt.device)
            tgt_mask = tgt_pad_mask & causal_mask

        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        self_attn_weights_list = []
        cross_attn_weights_list = []

        for layer in self.layers:
            x, self_w, cross_w = layer(x, memory, tgt_mask, memory_mask)
            self_attn_weights_list.append(self_w)
            cross_attn_weights_list.append(cross_w)

        return x, self_attn_weights_list, cross_attn_weights_list
