"""Task 1 - Data preparation.

Turn the UQA question-answering corpus into sentence-level
(source, target) pairs for question generation:

    source :  an Urdu sentence with the answer wrapped in <ans> ... </ans>
    target :  the question whose answer is that span

Run it as a script to build every split:

    python -m src.data_prep                 # train.tsv, valid.tsv, wiki.tsv
    python -m src.data_prep --max-rows 2000 # quick smoke test

Design notes for the viva
-------------------------
* We only keep the *sentence* that contains the answer, not the whole
  paragraph.  A from-scratch RNN cannot learn to read 300 tokens; the
  sentence-level setting (Du et al., 2017) is what makes the task learnable.
* The HuggingFace copy of UQA currently uses a *flat* schema
  (``answer`` / ``answer_start`` / ``is_impossible``).  Older SQuAD-style
  dumps use a *nested* ``answers`` dict.  ``normalise_row`` accepts both.
* We trust the dataset's character offsets but *verify* every one: if the
  slice ``sentence[rel:rel+len(answer)]`` is not exactly the answer, the
  row is dropped.  The count of such drops is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

from src.config import (
    ANS_CLOSE,
    ANS_OPEN,
    DATA_DIR,
    FIGURES_DIR,
    MAX_SRC_TOKENS,
    MAX_TGT_TOKENS,
    RESULTS_DIR,
    SENT_DELIMS,
    WIKI_TSV,
    TRAIN_TSV,
    VALID_TSV,
)

# HuggingFace dataset ids (see the assignment manual).
UQA_ID = "uqa/UQA"
WIKI_UQA_ID = "uqa/Wiki-UQA"


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, sentence)`` with character offsets into ``text``.

    A sentence ends at an Urdu full stop (U+06D4), an Urdu question mark
    (U+061F) or ``!``.  The delimiter is kept on the sentence it closes.
    """
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_DELIMS:
            yield start, i + 1, text[start : i + 1]
            start = i + 1
    if start < len(text):
        yield start, len(text), text[start:]


# ---------------------------------------------------------------------------
# Schema handling
# ---------------------------------------------------------------------------
def normalise_row(row: dict) -> Optional[tuple[str, str, str, int]]:
    """Return ``(context, question, answer_text, answer_start)`` or ``None``.

    Handles both the flat UQA schema and the nested SQuAD-style schema.
    Rows that are unanswerable or have no usable offset return ``None``.
    """
    context = row.get("context") or ""
    question = row.get("question") or ""
    if not context or not question:
        return None

    # Nested SQuAD-style: {"answers": {"text": [...], "answer_start": [...]}}
    answers = row.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        if not texts or not starts:
            return None
        return context, question, texts[0], int(starts[0])

    # Flat UQA schema.
    if row.get("is_impossible"):
        return None
    answer = row.get("answer")
    start = row.get("answer_start")
    if not answer or start is None or int(start) < 0:
        return None
    return context, question, answer, int(start)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def _collapse_ws(s: str) -> str:
    """Normalise all runs of whitespace to a single space."""
    return " ".join(s.split())


def make_pair(
    row: dict,
    max_src: int = MAX_SRC_TOKENS,
    max_tgt: int = MAX_TGT_TOKENS,
) -> Optional[tuple[str, str]] | str:
    """Build one ``(source, target)`` pair.

    Returns the pair, or a short string reason code when the row is
    unusable (so the caller can tally *why* rows were dropped):
    ``"unanswerable"``, ``"no-sentence"``, ``"offset-mismatch"``,
    ``"too-long"``.
    """
    norm = normalise_row(row)
    if norm is None:
        return "unanswerable"
    context, question, a_text, a_start = norm

    context = unicodedata.normalize("NFC", context)
    a_text = unicodedata.normalize("NFC", a_text).strip()
    question = unicodedata.normalize("NFC", question)
    if not a_text:
        return "unanswerable"

    for s, e, sent in split_sentences(context):
        if not (s <= a_start < e):
            continue
        rel = a_start - s
        window = sent[rel : rel + len(a_text)]
        if window != a_text:
            # Offsets can drift by a few chars; try a small local search.
            local = sent.find(a_text)
            if local == -1:
                return "offset-mismatch"
            rel = local
        src = (
            sent[:rel]
            + f" {ANS_OPEN} "
            + a_text
            + f" {ANS_CLOSE} "
            + sent[rel + len(a_text) :]
        )
        src = _collapse_ws(src)
        tgt = _collapse_ws(question)
        if not tgt:
            return "unanswerable"
        if len(src.split()) > max_src or len(tgt.split()) > max_tgt:
            return "too-long"
        return src, tgt

    return "no-sentence"


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------
def build_split(rows: Iterable[dict], out_path: Path) -> dict:
    """Write ``out_path`` (tab-separated ``source\\ttarget``) and return stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reasons: Counter[str] = Counter()
    src_lens: list[int] = []
    tgt_lens: list[int] = []
    n_raw = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            n_raw += 1
            result = make_pair(row)
            if isinstance(result, str):
                reasons[result] += 1
                continue
            src, tgt = result
            # Guard against stray control chars breaking the TSV.
            src = src.replace("\t", " ").replace("\r", " ")
            tgt = tgt.replace("\t", " ").replace("\r", " ")
            f.write(f"{src}\t{tgt}\n")
            src_lens.append(len(src.split()))
            tgt_lens.append(len(tgt.split()))

    kept = len(src_lens)
    stats = {
        "file": str(out_path.relative_to(out_path.parents[1])),
        "raw_rows": n_raw,
        "answerable_rows": n_raw - reasons["unanswerable"],
        "pairs_after_filter": kept,
        "dropped": dict(reasons),
        "mean_src_len": round(sum(src_lens) / kept, 2) if kept else 0.0,
        "mean_tgt_len": round(sum(tgt_lens) / kept, 2) if kept else 0.0,
        "max_src_len": max(src_lens, default=0),
        "max_tgt_len": max(tgt_lens, default=0),
    }
    print(f"  {out_path.name}: {kept} pairs kept from {n_raw} rows  {dict(reasons)}")
    return stats | {"_src_lens": src_lens, "_tgt_lens": tgt_lens}


def plot_length_hists(stats_by_split: dict, out_path: Path) -> None:
    """Source/target length histograms for every split (Figure 2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(stats_by_split)
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 7), squeeze=False)
    for col, name in enumerate(splits):
        st = stats_by_split[name]
        axes[0][col].hist(st["_src_lens"], bins=30, color="#3b6ea5")
        axes[0][col].set_title(f"{name} - source tokens")
        axes[0][col].axvline(MAX_SRC_TOKENS, color="crimson", ls="--", lw=1)
        axes[1][col].hist(st["_tgt_lens"], bins=25, color="#a5673b")
        axes[1][col].set_title(f"{name} - target tokens")
        axes[1][col].axvline(MAX_TGT_TOKENS, color="crimson", ls="--", lw=1)
    for ax in axes.ravel():
        ax.set_xlabel("whitespace tokens")
        ax.set_ylabel("pairs")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows per split (quick smoke test)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # Keep the cell output clean: no download progress bars, warnings only.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    import datasets
    from datasets import load_dataset

    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bars()

    print("loading UQA and Wiki-UQA from HuggingFace ...")
    uqa = load_dataset(UQA_ID)
    wiki = load_dataset(WIKI_UQA_ID)

    # Wiki-UQA ships as a single split under some revisions.
    wiki_rows = wiki["test"] if "test" in wiki else wiki[list(wiki)[0]]

    def limit(ds):
        return ds.select(range(min(len(ds), args.max_rows))) if args.max_rows else ds

    jobs = [
        ("train", limit(uqa["train"]), TRAIN_TSV),
        ("valid", limit(uqa["validation"]), VALID_TSV),
        ("wiki", limit(wiki_rows), WIKI_TSV),
    ]

    stats_by_split: dict[str, dict] = {}
    for name, rows, path in jobs:
        print(f"Building {name} ...")
        stats_by_split[name] = build_split(rows, path)

    # Persist stats for Table 1 (without the big per-row length lists).
    slim = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in stats_by_split.items()
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_stats.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_plot:
        plot_length_hists(stats_by_split, FIGURES_DIR / "length_hist.png")

    # Compact preview: one pair per split, source truncated.
    print("\nSample pairs (source -> target):")
    for path in (TRAIN_TSV, VALID_TSV, WIKI_TSV):
        if not path.exists():
            continue
        src, tgt = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
        src = src if len(src) <= 100 else src[:100] + " ..."
        print(f"  [{path.stem:5}] {src}\n          -> {tgt}")


if __name__ == "__main__":
    main()
