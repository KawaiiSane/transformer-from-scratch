"""
Matplotlib visualization helpers for Transformer attention and training.

Used by notebooks, train.py exports, and the HTML showcase generator.
"""

from __future__ import annotations

import io
import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_heatmap(
    weights: np.ndarray | list,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str = "Attention Weights",
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """
    Plot a single attention weight matrix as a heatmap.

    Args:
        weights: (seq_len_q, seq_len_k) attention probabilities
    """
    weights = np.array(weights)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


def plot_multi_head_attention(
    weights: np.ndarray,
    n_heads: int,
    title_prefix: str = "Head",
) -> plt.Figure:
    """Plot all attention heads in a grid."""
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        im = ax.imshow(weights[h], cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{title_prefix} {h}")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig


def plot_training_loss(
    train_loss: list[float],
    val_loss: list[float] | None = None,
    title: str = "Training Loss",
) -> plt.Figure:
    """Plot training (and optional validation) loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train", color="#2563eb")
    if val_loss:
        ax.plot(epochs, val_loss, label="Validation", color="#dc2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_positional_encoding_matrix(
    pe: np.ndarray,
    title: str = "Positional Encodings",
) -> plt.Figure:
    """Plot PE matrix as heatmap. pe shape: (max_len, d_model)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", origin="lower")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """Convert matplotlib figure to base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 120) -> None:
    """Save figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
