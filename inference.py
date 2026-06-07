"""
Inference script — greedy and beam search decoding.

Usage:
    python inference.py --src "hello"
    python inference.py --src "thank you" --beam 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import config
from dataset import Vocabulary
from models.transformer import Transformer


def load_model(checkpoint_path: str = config.CHECKPOINT_PATH, device: torch.device | None = None):
    """Load trained model and vocabularies."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run `python train.py` first."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_path = config.VOCAB_PATH

    with open(vocab_path, encoding="utf-8") as f:
        vocab_data = json.load(f)

    src_vocab = Vocabulary(
        token2idx=vocab_data["src_token2idx"],
        idx2token={int(k): v for k, v in vocab_data["src_idx2token"].items()},
    )
    tgt_vocab = Vocabulary(
        token2idx=vocab_data["tgt_token2idx"],
        idx2token={int(k): v for k, v in vocab_data["tgt_idx2token"].items()},
    )

    model = Transformer(
        src_vocab_size=ckpt["src_vocab_size"],
        tgt_vocab_size=ckpt["tgt_vocab_size"],
        pad_idx=src_vocab.pad_idx,
        sos_idx=src_vocab.sos_idx,
        eos_idx=src_vocab.eos_idx,
        **ckpt["config"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, src_vocab, tgt_vocab, device


def translate(
    model: Transformer,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    text: str,
    device: torch.device,
    beam_size: int = 0,
    return_attention: bool = False,
):
    """Translate a single English sentence to French."""
    src_ids = [src_vocab.sos_idx] + src_vocab.encode(text) + [src_vocab.eos_idx]
    src_ids = src_ids[: config.MAX_SEQ_LEN]
    src_ids += [src_vocab.pad_idx] * (config.MAX_SEQ_LEN - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    if beam_size > 0:
        generated = model.beam_search_decode(src_tensor, config.MAX_SEQ_LEN, beam_size)
        attn = None
    else:
        generated, attn = model.greedy_decode(
            src_tensor, config.MAX_SEQ_LEN, return_attention=return_attention
        )

    translation = tgt_vocab.decode(generated[0].tolist())
    return translation, attn


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English to French")
    parser.add_argument("--src", type=str, default="hello", help="Source sentence")
    parser.add_argument("--beam", type=int, default=0, help="Beam size (0 = greedy)")
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH)
    args = parser.parse_args()

    model, src_vocab, tgt_vocab, device = load_model(args.checkpoint)
    translation, _ = translate(
        model, src_vocab, tgt_vocab, args.src, device, beam_size=args.beam
    )

    print(f"English:  {args.src}")
    print(f"French:   {translation}")
    if args.beam > 0:
        print(f"(beam search, beam_size={args.beam})")


if __name__ == "__main__":
    main()
