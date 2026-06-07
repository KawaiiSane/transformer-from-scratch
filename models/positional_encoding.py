"""
Positional Encoding — Paper Section 3.5

Since self-attention treats all positions equally (permutation-invariant),
we must inject information about token order. The original paper uses fixed
sinusoidal encodings rather than learned embeddings.

SHAPES:
    Input:  (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, d_model)   # embedding + positional encoding

MATH:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

INTUITION:
    Each dimension oscillates at a different frequency. Low dimensions change
    slowly (capture coarse position); high dimensions change quickly (fine position).
    The model can learn to attend by relative position via linear combinations of
    these sin/cos waves (see paper: PE(pos+k) is a linear function of PE(pos)).

COMPLEXITY:
    O(max_len × d_model) precomputation, O(seq_len × d_model) per forward pass.
    Negligible compared to attention's O(L² × d).

INTERVIEW:
    Q: Why sin/cos instead of learned positional embeddings?
    A: Sinusoidal encodings generalize to unseen sequence lengths and encode
       relative positions via trigonometric identities. Learned embeddings work
       too (used in BERT/GPT) but the original paper chose fixed encodings.

PAPER REF: Section 3.5
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe shape: (1, max_len, d_model) — batch dim of 1 for broadcasting
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # div_term shape: (d_model/2,) — one frequency per sin/cos pair
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # register_buffer: saved with model state but NOT updated by optimizer
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings, shape (batch_size, seq_len, d_model)

        Returns:
            Embeddings with positional information added, same shape as x.
        """
        seq_len = x.size(1)
        # Slice PE to match current sequence length and broadcast over batch
        x = x + self.pe[:, :seq_len, :]  # (batch, seq_len, d_model)
        return self.dropout(x)


def plot_positional_encoding(
    d_model: int = 128,
    max_len: int = 100,
    title: str = "Sinusoidal Positional Encodings",
) -> plt.Figure:
    """
    Visualize the precomputed PE matrix as a heatmap.

    Returns:
        Matplotlib Figure (caller can save or display).
    """
    module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    pe = module.pe.squeeze(0).numpy()  # (max_len, d_model)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", origin="lower")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="PE value")
    fig.tight_layout()
    return fig


def plot_positional_encoding_waves(
    d_model: int = 128,
    max_len: int = 100,
    dims: tuple[int, ...] = (0, 1, 2, 3),
) -> plt.Figure:
    """Plot sin/cos waves for selected embedding dimensions."""
    module = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    pe = module.pe.squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    positions = range(max_len)
    for dim in dims:
        ax.plot(positions, pe[:, dim], label=f"dim {dim}")
    ax.set_xlabel("Position")
    ax.set_ylabel("PE value")
    ax.set_title("Positional encoding waves (selected dimensions)")
    ax.legend()
    fig.tight_layout()
    return fig
