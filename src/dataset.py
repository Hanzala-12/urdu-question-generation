"""TSV -> tensors: dataset and padded-batch collation.

``QGDataset`` reads a ``source<TAB>target`` file (produced by
``src.data_prep``) and tokenises each side with the trained SentencePiece
model.  ``collate_batch`` pads a list of examples into rectangular tensors.

Tensor conventions
------------------
* source ids:  ``[B, S]``  padded with ``PAD_ID``
* target ids:  ``[B, T]``  ``<s> w1 w2 ... wn </s>`` then padded
* the decoder is trained to predict ``target[:, 1:]`` from ``target[:, :-1]``
* ``src_lengths`` (LongTensor ``[B]``) lets the encoder pack the sequence
* ``src_key_padding_mask`` (BoolTensor ``[B, S]``) is ``True`` on PAD, for
  the attention layer to ignore.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.config import BOS_ID, EOS_ID, PAD_ID, CFG


def load_pairs(tsv_path: str | Path) -> list[tuple[str, str]]:
    """Read a ``source<TAB>target`` file into a list of tuples."""
    pairs: list[tuple[str, str]] = []
    with Path(tsv_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


class QGDataset(Dataset):
    """Question-generation pairs, tokenised to id lists on the fly."""

    def __init__(
        self,
        tsv_path: str | Path,
        sp,  # sentencepiece.SentencePieceProcessor
        max_src_len: int = CFG.train.max_src_len,
        max_tgt_len: int = CFG.train.max_tgt_len,
    ) -> None:
        self.pairs = load_pairs(tsv_path)
        self.sp = sp
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src_text, tgt_text = self.pairs[idx]
        # SentencePiece already knows <ans> / </ans> as single symbols.
        src_ids = self.sp.encode(src_text, out_type=int)[: self.max_src_len]
        tgt_ids = self.sp.encode(tgt_text, out_type=int)[: self.max_tgt_len - 2]
        tgt_ids = [BOS_ID, *tgt_ids, EOS_ID]
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def _pad(seqs: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Stack variable-length 1-D tensors into ``[B, max_len]``."""
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_batch(batch: list[dict]) -> dict:
    """Pad a list of ``QGDataset`` items into batched tensors."""
    batch = sorted(batch, key=lambda ex: ex["src"].size(0), reverse=True)
    src = _pad([ex["src"] for ex in batch], PAD_ID)
    tgt = _pad([ex["tgt"] for ex in batch], PAD_ID)
    src_lengths = torch.tensor([ex["src"].size(0) for ex in batch], dtype=torch.long)
    return {
        "src": src,                              # [B, S]
        "tgt": tgt,                              # [B, T]
        "src_lengths": src_lengths,              # [B]
        "src_key_padding_mask": src.eq(PAD_ID),  # [B, S]  True where PAD
        "src_texts": [ex["src_text"] for ex in batch],
        "tgt_texts": [ex["tgt_text"] for ex in batch],
    }
