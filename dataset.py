"""
Toy English→French dataset and word-level vocabulary.

The corpus is intentionally tiny so the model trains in minutes on CPU.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import config

# Built-in translation pairs (source → target)
TRANSLATION_PAIRS = [
    ("hello", "bonjour"),
    ("thank you", "merci"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("how are you", "comment allez vous"),
    ("see you later", "au revoir"),
    ("i love you", "je t aime"),
    ("goodbye", "au revoir"),
]

SPECIAL_TOKENS = [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.UNK_TOKEN]


@dataclass
class Vocabulary:
    """Word-level vocabulary with special tokens."""

    token2idx: dict[str, int]
    idx2token: dict[int, str]

    @property
    def pad_idx(self) -> int:
        return self.token2idx[config.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[config.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[config.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[config.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)

    def encode(self, text: str) -> list[int]:
        tokens = text.lower().strip().split()
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for idx in ids:
            if idx in (self.pad_idx, self.sos_idx):
                continue
            if idx == self.eos_idx:
                break
            token = self.idx2token.get(idx, config.UNK_TOKEN)
            if token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "token2idx": self.token2idx,
                    "idx2token": {str(k): v for k, v in self.idx2token.items()},
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        idx2token = {int(k): v for k, v in data["idx2token"].items()}
        return cls(token2idx=data["token2idx"], idx2token=idx2token)


def build_vocab(texts: list[str]) -> Vocabulary:
    """Build vocabulary from a list of text strings."""
    token2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for text in texts:
        for token in text.lower().split():
            if token not in token2idx:
                token2idx[token] = len(token2idx)
    idx2token = {v: k for k, v in token2idx.items()}
    return Vocabulary(token2idx=token2idx, idx2token=idx2token)


class TranslationDataset(Dataset):
    """Dataset of (source_ids, target_ids) padded to max_len."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = config.MAX_SEQ_LEN,
    ) -> None:
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.pairs[idx]

        # Encode: [SOS, ...tokens..., EOS, PAD...]
        src_ids = [self.src_vocab.sos_idx] + self.src_vocab.encode(src_text) + [self.src_vocab.eos_idx]
        tgt_ids = [self.tgt_vocab.sos_idx] + self.tgt_vocab.encode(tgt_text) + [self.tgt_vocab.eos_idx]

        src_ids = src_ids[: self.max_len]
        tgt_ids = tgt_ids[: self.max_len]

        src_ids += [self.src_vocab.pad_idx] * (self.max_len - len(src_ids))
        tgt_ids += [self.tgt_vocab.pad_idx] * (self.max_len - len(tgt_ids))

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def get_dataloaders(
    val_size: int = 1,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    repeat_train: int = 10,
) -> tuple[DataLoader, DataLoader, Vocabulary, Vocabulary, list, list]:
    """
    Split pairs into train/val and return dataloaders + vocabs.

    Returns:
        train_loader, val_loader, src_vocab, tgt_vocab, train_pairs, val_pairs
    """
    pairs = list(TRANSLATION_PAIRS)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:] * repeat_train  # repeat for stable tiny-data training

    all_src = [p[0] for p in pairs]
    all_tgt = [p[1] for p in pairs]
    src_vocab = build_vocab(all_src)
    tgt_vocab = build_vocab(all_tgt)

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, src_vocab, tgt_vocab, train_pairs, val_pairs
