"""
Tiny decoder-only Transformer (GPT-style) — Bonus module.

Unlike the full encoder-decoder model, this uses ONLY a stack of decoder
blocks with causal self-attention — the architecture behind GPT.

This is a simplified educational version for understanding decoder-only LMs.

PAPER REF: Decoder stack from Section 3.1, adapted for language modeling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.attention import create_causal_mask, create_padding_mask
from models.decoder import DecoderLayer
from models.positional_encoding import PositionalEncoding


class GPTDecoder(nn.Module):
    """
    Decoder-only Transformer for next-token prediction.

    SHAPES:
        Input:  (batch, seq_len) token ids
        Output: (batch, seq_len, vocab_size) logits
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        max_seq_len: int = 128,
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
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) input token ids

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        tgt_mask = create_padding_mask(x, self.pad_idx) & create_causal_mask(
            x.size(1), device=x.device
        )

        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoding(h)

        # Use self-attention only: pass h as both query input and fake "memory"
        for layer in self.layers:
            h, _, _ = layer(h, h, tgt_mask, tgt_mask)

        return self.output_projection(h)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 20) -> torch.Tensor:
        """Autoregressive generation from a prompt."""
        self.eval()
        generated = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 2).all():  # assumes eos_idx=2
                break
        return generated
