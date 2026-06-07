"""
Position-wise Feed-Forward Network — Paper Section 3.3

Applied identically to each position (same weights, independent per token).

SHAPES:
    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)

MATH:
    FFN(x) = max(0, x W1 + b1) W2 + b2

INTUITION:
    After attention mixes information across positions, the FFN processes each
    position independently — acting like a small MLP per token. This adds
    non-linearity and increases model capacity.

COMPLEXITY:
    O(seq_len × d_model × d_ff) — linear in sequence length.

INTERVIEW:
    Q: Why is the FFN applied position-wise?
    A: Attention handles cross-token interaction; FFN adds depth and non-linearity
       to each position's representation without further mixing.

PAPER REF: Section 3.3
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Two linear layers with ReLU activation and dropout."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # (B, L, d_model) → (B, L, d_ff) → (B, L, d_model)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)
