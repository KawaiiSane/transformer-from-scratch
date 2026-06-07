"""
Training loop for the educational Transformer.

Usage:
    python train.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import config
from dataset import TRANSLATION_PAIRS, get_dataloaders
from models.transformer import Transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Cross-entropy with teacher forcing.

    logits: (batch, tgt_len, vocab_size)
    tgt:    (batch, tgt_len) — we predict token t+1 from position t
    """
    # Shift: input to decoder is tgt[:, :-1], predict tgt[:, 1:]
    logits = logits[:, :-1, :].contiguous()
    labels = tgt[:, 1:].contiguous()

    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=pad_idx,
    )


def attention_to_json(attention: dict, src_tokens: list[str], tgt_tokens: list[str]) -> dict:
    """Convert attention tensors to JSON-serializable lists (first batch, head 0)."""
    result = {}

    if "encoder" in attention and attention["encoder"]:
        enc = attention["encoder"][-1][0, 0].detach().cpu().numpy().tolist()
        result["encoder_self"] = {
            "weights": enc,
            "row_labels": src_tokens,
            "col_labels": src_tokens,
            "title": "Encoder Self-Attention (Layer -1, Head 0)",
        }

    if "decoder_cross" in attention and attention["decoder_cross"]:
        cross = attention["decoder_cross"][-1][0, 0].detach().cpu().numpy().tolist()
        result["decoder_cross"] = {
            "weights": cross,
            "row_labels": tgt_tokens,
            "col_labels": src_tokens,
            "title": "Decoder Cross-Attention (Layer -1, Head 0)",
        }

    if "decoder_self" in attention and attention["decoder_self"]:
        self_attn = attention["decoder_self"][-1][0, 0].detach().cpu().numpy().tolist()
        result["decoder_self"] = {
            "weights": self_attn,
            "row_labels": tgt_tokens,
            "col_labels": tgt_tokens,
            "title": "Decoder Self-Attention (Layer -1, Head 0)",
        }

    return result


def train() -> None:
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, src_vocab, tgt_vocab, train_pairs, val_pairs = get_dataloaders()
    src_vocab.save(config.VOCAB_PATH.replace("vocab.json", "src_vocab.json"))
    tgt_vocab.save(config.VOCAB_PATH.replace("vocab.json", "tgt_vocab.json"))

    # Combined vocab metadata for inference
    with open(config.VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "src_token2idx": src_vocab.token2idx,
                "tgt_token2idx": tgt_vocab.token2idx,
                "src_idx2token": {str(k): v for k, v in src_vocab.idx2token.items()},
                "tgt_idx2token": {str(k): v for k, v in tgt_vocab.idx2token.items()},
            },
            f,
            indent=2,
        )

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        pad_idx=src_vocab.pad_idx,
        sos_idx=src_vocab.sos_idx,
        eos_idx=src_vocab.eos_idx,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        n_layers=config.N_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"Training on {len(train_pairs)} pairs, validating on {len(val_pairs)} pairs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits, _ = model(src, tgt)
            loss = compute_loss(logits, tgt, src_vocab.pad_idx)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits, _ = model(src, tgt)
                total_val_loss += compute_loss(logits, tgt, src_vocab.pad_idx).item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "d_model": config.D_MODEL,
                        "n_heads": config.N_HEADS,
                        "d_ff": config.D_FF,
                        "n_layers": config.N_LAYERS,
                        "max_seq_len": config.MAX_SEQ_LEN,
                        "dropout": config.DROPOUT,
                    },
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                },
                config.CHECKPOINT_PATH,
            )

    with open(config.TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Capture attention snapshot on a sample sentence
    model.eval()
    sample_src = "hello"
    src_ids = [src_vocab.sos_idx] + src_vocab.encode(sample_src) + [src_vocab.eos_idx]
    src_ids = src_ids[: config.MAX_SEQ_LEN]
    src_ids += [src_vocab.pad_idx] * (config.MAX_SEQ_LEN - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    tgt_text = dict(TRANSLATION_PAIRS).get(sample_src, "bonjour")
    tgt_ids = [tgt_vocab.sos_idx] + tgt_vocab.encode(tgt_text) + [tgt_vocab.eos_idx]
    tgt_ids = tgt_ids[: config.MAX_SEQ_LEN]
    tgt_ids += [tgt_vocab.pad_idx] * (config.MAX_SEQ_LEN - len(tgt_ids))
    tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, attention = model(src_tensor, tgt_tensor)

    src_tokens = [config.SOS_TOKEN] + sample_src.split() + [config.EOS_TOKEN]
    tgt_tokens = [config.SOS_TOKEN] + tgt_text.split() + [config.EOS_TOKEN]

    snapshot = {
        "sample": {"src": sample_src, "tgt": tgt_text},
        "attention": attention_to_json(attention, src_tokens, tgt_tokens),
        "translations": [],
    }

    # Record translations for all pairs
    for en, fr in TRANSLATION_PAIRS:
        s_ids = [src_vocab.sos_idx] + src_vocab.encode(en) + [src_vocab.eos_idx]
        s_ids = s_ids[: config.MAX_SEQ_LEN]
        s_ids += [src_vocab.pad_idx] * (config.MAX_SEQ_LEN - len(s_ids))
        s = torch.tensor([s_ids], dtype=torch.long, device=device)
        generated, _ = model.greedy_decode(s, config.MAX_SEQ_LEN)
        pred = tgt_vocab.decode(generated[0].tolist())
        snapshot["translations"].append({"src": en, "target": fr, "predicted": pred})

    with open(config.ATTENTION_SNAPSHOTS_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {config.CHECKPOINT_PATH}")
    print(f"Run: python visualizations/generate_showcase.py")


if __name__ == "__main__":
    train()
