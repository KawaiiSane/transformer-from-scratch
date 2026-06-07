"""
Scaled Dot-Product Attention & Multi-Head Attention — Paper Section 3.2

This module implements attention entirely from PyTorch primitives.
Do NOT use torch.nn.MultiheadAttention or torch.nn.Transformer.

SHAPES (Scaled Dot-Product Attention):
    Q: (batch, n_heads, seq_len_q, d_k)
    K: (batch, n_heads, seq_len_k, d_k)
    V: (batch, n_heads, seq_len_v, d_v)   where seq_len_v == seq_len_k

    scores:      (batch, n_heads, seq_len_q, seq_len_k)
    attn_weights:(batch, n_heads, seq_len_q, seq_len_k)  — after softmax
    output:      (batch, n_heads, seq_len_q, d_v)

MATH:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

INTUITION:
    Each query vector asks "how relevant is each key?" via dot product.
    Softmax converts scores to a probability distribution over values.
    Scaling by sqrt(d_k) prevents dot products from growing large (softmax saturation).

COMPLEXITY:
    O(L_q × L_k × d_k) per head. Multi-head multiplies by H but splits d_model.

INTERVIEW:
    Q: Why scale by sqrt(d_k)?
    A: Dot products of random vectors with dimension d_k have variance ~d_k.
       Dividing by sqrt(d_k) keeps variance ~1, preserving useful softmax gradients.

PAPER REF: Section 3.2
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries  — (batch, n_heads, seq_len_q, d_k)
        K: Keys     — (batch, n_heads, seq_len_k, d_k)
        V: Values   — (batch, n_heads, seq_len_v, d_v)
        mask: Optional boolean/float mask broadcastable to scores.
              Use True/1 for positions to KEEP, False/0 for positions to MASK.
              Masked positions receive -inf before softmax.
        dropout: Optional dropout applied to attention weights.

    Returns:
        output:       (batch, n_heads, seq_len_q, d_v)
        attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Step 1: Raw attention scores via matrix multiply
    # Q @ K^T → (batch, n_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply mask (padding or causal)
    if mask is not None:
        # Convert keep-mask to additive mask: 0 for keep, -inf for mask
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 3: Softmax over keys (last dimension)
    attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, L_q, L_k)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Step 4: Weighted sum of values
    output = torch.matmul(attn_weights, V)  # (batch, heads, L_q, d_v)

    return output, attn_weights


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask for encoder/decoder sequences.

    Args:
        seq: Token ids, shape (batch, seq_len)
        pad_idx: Index of the padding token

    Returns:
        mask: (batch, 1, 1, seq_len) — True = real token, False = pad
    """
    # True where token is NOT padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Create a causal (look-ahead) mask for decoder self-attention.

    Position i may only attend to positions j <= i.

    Returns:
        mask: (1, 1, seq_len, seq_len) — lower-triangular boolean matrix
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention — Paper Section 3.2

    Projects input into H parallel attention heads, runs scaled dot-product
    attention on each, then concatenates and projects back to d_model.

    SHAPES:
        Input:  (batch, seq_len, d_model)
        Output: (batch, seq_len, d_model)
        Weights:(batch, n_heads, seq_len, seq_len) or (batch, n_heads, L_q, L_k)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # Separate linear projections for Q, K, V, and output (paper: W_Q, W_K, W_V, W_O)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask:  Broadcastable to (batch, 1, seq_len_q, seq_len_k)

        Returns:
            output:       (batch, seq_len_q, d_model)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        # Linear projections
        Q = self._split_heads(self.W_q(query))   # (B, H, L_q, d_k)
        K = self._split_heads(self.W_k(key))     # (B, H, L_k, d_k)
        V = self._split_heads(self.W_v(value))   # (B, H, L_v, d_v)

        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )

        # Merge heads and apply output projection
        concat = self._merge_heads(attn_output)  # (B, L_q, d_model)
        output = self.W_o(concat)

        return output, attn_weights
