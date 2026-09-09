"""Central configuration for the Urdu question-generation project.

Every tunable number lives here so the notebook, training script, evaluation
and front end all agree. Import it as:

    from src.config import CFG

and read fields like ``CFG.hidden_size``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repo root, which is the parent of this file's folder)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_TSV = DATA_DIR / "train.tsv"
VALID_TSV = DATA_DIR / "valid.tsv"
WIKI_TSV = DATA_DIR / "wiki.tsv"
SPM_CORPUS = DATA_DIR / "sp_corpus.txt"

SPM_PREFIX = ARTIFACTS_DIR / "ur_sp"           # -> ur_sp.model / ur_sp.vocab
SPM_MODEL = ARTIFACTS_DIR / "ur_sp.model"
BEST_CKPT = ARTIFACTS_DIR / "best.pt"          # weights only, for the front end
LAST_CKPT = ARTIFACTS_DIR / "last.pt"          # weights + optimiser, for resume

# ---------------------------------------------------------------------------
# Special tokens.  Ids are fixed at tokenizer-training time (see spm_train.py).
# SentencePiece decodes an unknown piece as U+2047 (DOUBLE QUESTION MARK).
# ---------------------------------------------------------------------------
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"
UNK_SURFACE = "⁇"

# ---------------------------------------------------------------------------
# Data preparation (Task 1)
# ---------------------------------------------------------------------------
# Urdu full stop (U+06D4), Urdu question mark (U+061F), exclamation mark.
SENT_DELIMS = "۔؟!"
MAX_SRC_TOKENS = 60      # drop pairs whose source has more whitespace tokens
MAX_TGT_TOKENS = 25      # drop pairs whose target has more whitespace tokens
DEBUG_SUBSET = 10_000    # size of the quick "does the loss fall?" subset


@dataclass(frozen=True)
class ModelConfig:
    """Architecture — a from-scratch RNN encoder-decoder with attention."""

    vocab_size: int = 8000
    emb_dim: int = 256
    hidden_size: int = 512          # per direction in the encoder
    enc_layers: int = 2
    dec_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    rnn_cell: str = "lstm"          # "lstm" or "gru"
    attention: str = "bahdanau"     # "bahdanau" (additive) or "luong" (dot)
    tie_output_embedding: bool = False


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation schedule."""

    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    teacher_forcing: float = 0.5    # prob. of feeding the gold token at step t
    label_smoothing: float = 0.0
    lr_patience: int = 1            # ReduceLROnPlateau patience (epochs)
    lr_factor: float = 0.5
    seed: int = 1234
    num_workers: int = 2
    max_src_len: int = 80           # hard cap after tokenisation (subword)
    max_tgt_len: int = 40


@dataclass(frozen=True)
class DecodeConfig:
    """Inference settings shared by evaluation and the front end."""

    beam_size: int = 5
    length_penalty: float = 0.6     # alpha in score / (len ** alpha); GNMT-style
    max_decode_len: int = 40
    n_eval_samples: int = 60        # rows written to results/samples.tsv (>= 50)
    human_eval_n: int = 50
    human_eval_seed: int = 7


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)


CFG = Config()
