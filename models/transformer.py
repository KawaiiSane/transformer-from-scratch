"""
Full Transformer (Encoder-Decoder) — Paper Section 3, Figure 1

Wires together encoder, decoder, and the final linear projection layer.

SHAPES:
    src: (batch, src_len)
    tgt: (batch, tgt_len)
    output logits: (batch, tgt_len, tgt_vocab_size)

TRAINING: Teacher forcing — decoder input is shifted target sequence.
INFERENCE: Autoregressive greedy/beam decoding.

PAPER REF: Section 3, Figure 1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import create_causal_mask, create_padding_mask
from models.decoder import Decoder
from models.encoder import Encoder


class Transformer(nn.Module):
    """Encoder-Decoder Transformer for sequence-to-sequence tasks."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        max_seq_len: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, pad_idx, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, pad_idx, dropout
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Padding mask for encoder: (batch, 1, 1, src_len)"""
        return create_padding_mask(src, self.pad_idx)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Combined padding + causal mask: (batch, 1, tgt_len, tgt_len)"""
        tgt_pad_mask = create_padding_mask(tgt, self.pad_idx)
        causal_mask = create_causal_mask(tgt.size(1), device=tgt.device)
        return tgt_pad_mask & causal_mask

    def _make_memory_mask(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        """Cross-attention mask: (batch, 1, tgt_len, src_len)"""
        src_pad_mask = create_padding_mask(src, self.pad_idx)  # (B, 1, 1, src_len)
        return src_pad_mask.expand(-1, -1, tgt_len, -1)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        src_mask = self._make_src_mask(src)
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        tgt_mask = self._make_tgt_mask(tgt)
        memory_mask = self._make_memory_mask(src, tgt.size(1))
        return self.decoder(tgt, memory, tgt_mask, memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """
        Forward pass with teacher forcing.

        Args:
            src: (batch, src_len) source token ids
            tgt: (batch, tgt_len) target token ids (with SOS prepended)

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
            attention_dict: encoder, decoder_self, decoder_cross weights
        """
        memory, enc_attn = self.encode(src)
        dec_out, dec_self_attn, dec_cross_attn = self.decode(tgt, memory, src)
        logits = self.output_projection(dec_out)

        attention = {
            "encoder": enc_attn,
            "decoder_self": dec_self_attn,
            "decoder_cross": dec_cross_attn,
        }
        return logits, attention

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Greedy autoregressive decoding.

        Args:
            src: (1, src_len) single source sequence

        Returns:
            generated: (1, generated_len) token ids including SOS and EOS
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        memory, enc_attn = self.encode(src)

        # Start with SOS token
        generated = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)

        all_self_attn = []
        all_cross_attn = []

        for _ in range(max_len - 1):
            dec_out, self_attn, cross_attn = self.decode(generated, memory, src)
            logits = self.output_projection(dec_out[:, -1:, :])  # last position only
            next_token = logits.argmax(dim=-1)  # (batch, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if return_attention:
                all_self_attn.append([w[:, :, -1:, :] for w in self_attn])
                all_cross_attn.append([w[:, :, -1:, :] for w in cross_attn])

            # Stop at EOS or PAD, or after reasonable length
            if (next_token == self.eos_idx).all() or (next_token == self.pad_idx).all():
                break
            if generated.size(1) >= src.size(1) + 8:
                break

        attn = None
        if return_attention:
            attn = {
                "encoder": enc_attn,
                "decoder_self": all_self_attn,
                "decoder_cross": all_cross_attn,
            }

        return generated, attn

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        max_len: int,
        beam_size: int = 3,
    ) -> torch.Tensor:
        """
        Beam search decoding (bonus feature).

        Returns:
            best sequence: (1, seq_len)
        """
        self.eval()
        device = src.device

        memory, _ = self.encode(src)
        memory = memory.expand(beam_size, -1, -1)

        src_expanded = src.expand(beam_size, -1)

        # Each beam: (score, token_sequence)
        beams = [(0.0, [self.sos_idx])]

        for _ in range(max_len - 1):
            candidates = []
            for score, seq in beams:
                if seq[-1] == self.eos_idx:
                    candidates.append((score, seq))
                    continue

                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                dec_out, _, _ = self.decode(tgt, memory[:1], src)
                logits = self.output_projection(dec_out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                topk_log_probs, topk_ids = log_probs.topk(beam_size)
                for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    candidates.append((score + lp, seq + [tid]))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            if all(seq[-1] == self.eos_idx for _, seq in beams):
                break

        best_seq = beams[0][1]
        return torch.tensor([best_seq], dtype=torch.long, device=device)
