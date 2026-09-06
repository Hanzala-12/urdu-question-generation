"""Task 2 - train the SentencePiece subword tokenizer.

    python -m src.spm_train

Reads ``data/train.tsv`` (so the tokenizer only ever sees training text),
writes ``artifacts/ur_sp.model`` / ``.vocab`` and a few worked examples to
``results/tokenizer_examples.txt`` for the blog.

Choices, for the viva
---------------------
* ``model_type="unigram"`` - the SentencePiece default; picks a vocabulary
  that maximises the likelihood of the corpus under a unigram LM. Good for
  morphologically rich languages because it keeps frequent whole words and
  splits rare ones into reusable stems + affixes.
* ``vocab_size=8000`` - fixed by the manual.
* ``character_coverage=1.0`` - keep every Urdu character; the script is
  small enough that we do not need to drop rare glyphs.
* ``<ans>`` / ``</ans>`` are ``user_defined_symbols`` so a full tag is
  always exactly one token and never split.
* ids are pinned: pad=0, unk=1, bos=2, eos=3 (see ``src/config.py``).
"""

from __future__ import annotations

import argparse

import sentencepiece as spm

from src.config import (
    ANS_CLOSE,
    ANS_OPEN,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    RESULTS_DIR,
    SPM_CORPUS,
    SPM_MODEL,
    SPM_PREFIX,
    TRAIN_TSV,
    UNK_ID,
)
from src.dataset import load_pairs


def build_corpus() -> int:
    """Write one sentence per line (sources then targets) to SPM_CORPUS."""
    pairs = load_pairs(TRAIN_TSV)
    SPM_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with SPM_CORPUS.open("w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(src + "\n" + tgt + "\n")
    print(f"corpus: {SPM_CORPUS}  ({2 * len(pairs)} lines from {len(pairs)} pairs)")
    return len(pairs)


def train(vocab_size: int = 8000) -> None:
    SPM_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(SPM_CORPUS),
        model_prefix=str(SPM_PREFIX),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        user_defined_symbols=[ANS_OPEN, ANS_CLOSE],
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        # Let a small corpus (local smoke test) settle for a smaller vocab
        # instead of hard-failing; the full-data run always reaches 8000.
        hard_vocab_limit=False,
    )
    print(f"wrote {SPM_MODEL}")


def report() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_MODEL))
    print(f"vocab size: {sp.get_piece_size()}")
    # Check the tags in the spaced context data_prep actually produces
    # (an isolated "<ans>" fragments because of SentencePiece's dummy
    # prefix; that never happens mid-sentence).
    probe = f"کتاب {ANS_OPEN} علی {ANS_CLOSE} ہے"
    pieces = sp.encode(probe, out_type=str)
    for tag in (ANS_OPEN, ANS_CLOSE):
        assert pieces.count(tag) == 1, f"{tag} not a single piece in {pieces}"
        print(f"  {tag!r} -> id {sp.piece_to_id(tag)} (single token in context, good)")

    pairs = load_pairs(TRAIN_TSV)[:5]
    lines = ["Five tokenised training examples\n" + "=" * 34 + "\n"]
    for src, tgt in pairs:
        s_pieces = sp.encode(src, out_type=str)
        t_pieces = sp.encode(tgt, out_type=str)
        rt = sp.decode(sp.encode(tgt, out_type=int)) == " ".join(tgt.split())
        block = (
            f"SRC  {src}\n"
            f"     {' '.join(s_pieces)}   ({len(s_pieces)} pieces)\n"
            f"TGT  {tgt}\n"
            f"     {' '.join(t_pieces)}   ({len(t_pieces)} pieces)\n"
            f"     round-trip ok: {rt}\n"
        )
        print("\n" + block)
        lines.append(block)

    lines.append(
        "\nNote on Urdu morphology: the unigram model keeps common function\n"
        "words whole (کے، میں، سے) and breaks inflected or compound forms into\n"
        "a stem plus affix pieces (e.g. plural/oblique endings and the\n"
        "izafat 'ی'). Rare proper nouns fragment into character-level pieces,\n"
        "which is where most <unk>-like behaviour and copy errors come from.\n"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tokenizer_examples.txt").write_text("".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS_DIR / 'tokenizer_examples.txt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-size", type=int, default=8000,
                    help="8000 for the real run; smaller only for local smoke tests")
    args = ap.parse_args()
    build_corpus()
    train(vocab_size=args.vocab_size)
    report()


if __name__ == "__main__":
    main()
