"""Task 4 - evaluation.

    python -m src.evaluate                       # valid + wiki, greedy + beam
    python -m src.evaluate --split valid --beam-max 3000

Reports, per split and per decoding strategy:

* BLEU-4   - sacrebleu, corpus level, on detokenised text
* ROUGE-L  - rouge-score, mean F-measure
* PPL      - exp(mean teacher-forced cross-entropy) of best.pt
* <unk>%   - share of generated subword ids that are <unk>

Writes ``results/metrics.json``, ``results/samples.tsv`` (>= 50 rows:
source, reference, greedy, beam), ``results/figures/attention.png`` and a
draft ``results/tables.md``.

Sanity band from the manual: BLEU-4 ~ 6-13 on UQA-valid, lower on
Wiki-UQA.  Near 0 => a bug.  Above 30 => train/validation leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BEST_CKPT,
    CFG,
    FIGURES_DIR,
    PAD_ID,
    RESULTS_DIR,
    UNK_ID,
    VALID_TSV,
    WIKI_TSV,
)
from src.dataset import QGDataset, collate_batch, load_pairs
from src.decode import beam_search, greedy_decode
from src.train import run_epoch
from src.utils import build_model_from_checkpoint, load_checkpoint, load_sp, pick_device


def _bleu(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(hyps, [refs]).score


class _WhitespaceTokenizer:
    """rouge_score's default tokenizer keeps only [a-z0-9] after lowercasing,
    which deletes every Urdu character. Split on whitespace instead."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def _rouge_l(hyps: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=_WhitespaceTokenizer()
    )
    total = sum(
        scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)
    )
    return total / max(len(refs), 1)


@torch.no_grad()
def _decode_split(model, sp, pairs, device, beam_max: int):
    """Return greedy hyps, beam hyps, refs, and the <unk> id-rate for greedy."""
    ds = QGDataset.__new__(QGDataset)          # lightweight: reuse tokeniser only
    ds.pairs, ds.sp = pairs, sp
    ds.max_src_len, ds.max_tgt_len = CFG.train.max_src_len, CFG.train.max_tgt_len
    loader = DataLoader(ds, batch_size=CFG.train.batch_size,
                        shuffle=False, collate_fn=collate_batch)

    greedy_hyps: list[str] = []
    refs: list[str] = []
    n_unk = n_tok = 0
    order_src: list[str] = []
    for batch in loader:
        src = batch["src"].to(device)
        seqs, _ = greedy_decode(
            model, src, batch["src_lengths"],
            batch["src_key_padding_mask"].to(device),
        )
        for ids, ref_text, src_text in zip(seqs, batch["tgt_texts"], batch["src_texts"]):
            hyp_text = sp.decode(ids)
            greedy_hyps.append(hyp_text)
            refs.append(ref_text)
            order_src.append(src_text)
            # <unk> rate the way the manual's starter measures it: SentencePiece
            # renders <unk> as U+2047, over whitespace tokens of the output.
            n_unk += hyp_text.count("⁇")
            n_tok += len(hyp_text.split())

    beam_hyps: list[str] = []
    n_beam = len(pairs) if beam_max <= 0 else min(beam_max, len(pairs))
    for src_text, _ in pairs[:n_beam]:
        ids = sp.encode(src_text, out_type=int)[: CFG.train.max_src_len]
        src = torch.tensor([ids], dtype=torch.long, device=device)
        seq, _ = beam_search(model, src, torch.tensor([len(ids)]), src.eq(PAD_ID))
        beam_hyps.append(sp.decode(seq))

    return {
        "greedy": greedy_hyps,
        "beam": beam_hyps,
        "refs": refs,
        "src": order_src,
        "unk_rate": n_unk / max(n_tok, 1),
        "n_beam": n_beam,
    }


def evaluate_split(model, sp, tsv_path: Path, device, beam_max: int,
                   max_eval: int | None = None) -> dict:
    pairs = load_pairs(tsv_path)
    decode_pairs = pairs[:max_eval] if max_eval else pairs
    print(f"{tsv_path.name}: {len(pairs)} pairs "
          f"(decoding {len(decode_pairs)}, beam {min(beam_max or len(decode_pairs), len(decode_pairs))})")
    out = _decode_split(model, sp, decode_pairs, device, beam_max)

    # Perplexity: teacher-forced CE over the whole split.
    ds = QGDataset(tsv_path, sp)
    loader = DataLoader(ds, batch_size=CFG.train.batch_size,
                        shuffle=False, collate_fn=collate_batch)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    ce = run_epoch(model, loader, crit, device)
    ppl = float(torch.exp(torch.tensor(min(ce, 20.0))))

    g, b, refs = out["greedy"], out["beam"], out["refs"]
    metrics = {
        "n_pairs": len(pairs),
        "n_greedy_scored": len(refs),
        "perplexity": round(ppl, 3),
        "unk_pct": round(100 * out["unk_rate"], 3),
        "greedy": {
            "bleu4": round(_bleu(g, refs), 2),
            "rougeL": round(_rouge_l(g, refs), 4),
        },
        "beam": {
            "k": CFG.decode.beam_size,
            "n_scored": out["n_beam"],
            "bleu4": round(_bleu(b, refs[: out["n_beam"]]), 2),
            "rougeL": round(_rouge_l(b, refs[: out["n_beam"]]), 4),
        },
    }
    return metrics, out


def _write_samples(out_valid: dict, path: Path, n: int) -> None:
    n = min(n, len(out_valid["beam"]))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("source\treference\tgreedy\tbeam\n")
        for i in range(n):
            row = [
                out_valid["src"][i], out_valid["refs"][i],
                out_valid["greedy"][i], out_valid["beam"][i],
            ]
            f.write("\t".join(c.replace("\t", " ").replace("\n", " ") for c in row) + "\n")
    print(f"wrote {path}  ({n} rows)")


def _write_human_eval(out_valid: dict, path: Path, n: int, seed: int) -> None:
    """Fixed-seed sample for the two-member human evaluation (Section 3.2)."""
    import csv
    import random

    pool = len(out_valid["greedy"])
    idx = sorted(random.Random(seed).sample(range(pool), min(n, pool)))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "source", "reference", "greedy",
                    "m1_fluency", "m1_relevance", "m1_answerability",
                    "m2_fluency", "m2_relevance", "m2_answerability"])
        for j in idx:
            w.writerow([j, out_valid["src"][j], out_valid["refs"][j],
                        out_valid["greedy"][j], "", "", "", "", "", ""])
    print(f"wrote {path}  ({len(idx)} rows) - fill m1_/m2_ columns with 1/0")


def _urdu_font():
    """An Arabic-script FontProperties (+ a reshaper), or (None, identity).

    Matplotlib's default font has no Urdu glyphs, so labels render as boxes
    unless we point it at a font that covers the Arabic block and reshape
    the text for right-to-left display.
    """
    import glob

    import matplotlib.font_manager as fm

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshape = lambda s: get_display(arabic_reshaper.reshape(s))
    except Exception:
        return None, (lambda s: s)

    # Prefer a proper Naskh/Sans text face; Kufi is decorative and drops glyphs.
    patterns = [
        r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/**/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/**/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/**/*NotoNaskhArabic*.ttf",
        "/usr/share/fonts/**/*NotoSansArabic*.ttf",
        "/usr/share/fonts/**/*Amiri*.ttf",
        "/usr/share/fonts/**/*[Nn]askh*.ttf",
        "/System/Library/Fonts/**/*Arab*.ttf",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return fm.FontProperties(fname=hits[0]), reshape
    return None, reshape


def _attention_figure(model, sp, source_text: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.decode import generate

    question, attn, src_pieces = generate(model, sp, source_text, mode="greedy")
    gen_pieces = [sp.id_to_piece(i) for i in sp.encode(question, out_type=int)]
    src_pieces = [p.replace("\u2581", "") or " " for p in src_pieces]
    gen_pieces = [p.replace("\u2581", "") or " " for p in gen_pieces]
    attn = attn[: len(gen_pieces), : len(src_pieces)]

    font, reshape = _urdu_font()
    fig, ax = plt.subplots(figsize=(max(7, len(src_pieces) * 0.45),
                                    max(3.5, len(gen_pieces) * 0.5)))
    im = ax.imshow(attn.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_pieces)))
    ax.set_yticks(range(len(gen_pieces)))
    if font is not None:
        ax.set_xticklabels([reshape(p) for p in src_pieces], rotation=90,
                           fontsize=9, fontproperties=font)
        ax.set_yticklabels([reshape(p) for p in gen_pieces], fontsize=9,
                           fontproperties=font)
    else:  # no Arabic font: index the axes, print the legend
        print("source pieces   :", " ".join(f"{i}:{p}" for i, p in enumerate(src_pieces)))
        print("generated pieces:", " ".join(f"{i}:{p}" for i, p in enumerate(gen_pieces)))
    ax.set_xlabel("source pieces")
    ax.set_ylabel("generated pieces")
    ax.set_title("Decoder attention (Bahdanau)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}   (source: {source_text[:60]}...)")


def _write_tables(all_metrics: dict, path: Path) -> None:
    stats_path = RESULTS_DIR / "data_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    lines = ["# Results tables\n"]

    lines.append("## Table 1 - Dataset statistics\n")
    lines.append("| | Train | Validation | Wiki-UQA |")
    lines.append("|---|---|---|---|")
    def g(split, key):
        return stats.get(split, {}).get(key, "")
    lines.append(f"| Rows in raw dataset | {g('train','raw_rows')} | {g('valid','raw_rows')} | {g('wiki','raw_rows')} |")
    lines.append(f"| Answerable rows | {g('train','answerable_rows')} | {g('valid','answerable_rows')} | {g('wiki','answerable_rows')} |")
    lines.append(f"| Pairs after length filter | {g('train','pairs_after_filter')} | {g('valid','pairs_after_filter')} | {g('wiki','pairs_after_filter')} |")
    lines.append(f"| Mean source / target length | {g('train','mean_src_len')} / {g('train','mean_tgt_len')} | {g('valid','mean_src_len')} / {g('valid','mean_tgt_len')} | {g('wiki','mean_src_len')} / {g('wiki','mean_tgt_len')} |\n")

    m = CFG.model
    lines.append("## Table 2 - Model configuration\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Encoder / decoder | {m.enc_layers}-layer BiLSTM / {m.dec_layers}-layer LSTM |")
    lines.append(f"| Embedding / hidden | {m.emb_dim} / {m.hidden_size} |")
    lines.append(f"| Attention | {m.attention} |")
    lines.append(f"| Vocabulary | {m.vocab_size} |")
    npar = all_metrics.get("_n_params", "")
    lines.append(f"| Trainable parameters | {npar} |")
    lines.append(f"| Optimiser | Adam lr={CFG.train.lr}, ReduceLROnPlateau |")
    meta_path = RESULTS_DIR / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    wall = f"{meta['wall_clock_s'] / 60:.1f} min" if "wall_clock_s" in meta else "-"
    gpu = meta.get("gpu", "-")
    ep = meta.get("epochs", CFG.train.epochs)
    bs = meta.get("batch_size", CFG.train.batch_size)
    lines.append(f"| Batch / epochs / wall-clock / GPU | {bs} / {ep} / {wall} / {gpu} |\n")

    lines.append("## Table 3 - Automatic metrics\n")
    lines.append("| Split | Decoding | BLEU-4 | ROUGE-L | PPL | <unk>% |")
    lines.append("|---|---|---|---|---|---|")
    for split in ("valid", "wiki"):
        if split not in all_metrics:
            continue
        md = all_metrics[split]
        lines.append(f"| {split} | greedy | {md['greedy']['bleu4']} | {md['greedy']['rougeL']} | {md['perplexity']} | {md['unk_pct']} |")
        lines.append(f"| {split} | beam (k={md['beam']['k']}) | {md['beam']['bleu4']} | {md['beam']['rougeL']} | {md['perplexity']} | {md['unk_pct']} |")
    lines.append("")
    lines.append("## Table 4 - Human evaluation (50 samples)\n")
    lines.append("| | Fluency | Relevance | Answerability |")
    lines.append("|---|---|---|---|")
    lines.append("| Member 1 (% yes) | | | |")
    lines.append("| Member 2 (% yes) | | | |")
    lines.append("| Cohen's kappa | | | |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["valid", "wiki", "both"], default="both")
    ap.add_argument("--beam-max", type=int, default=3000,
                    help="cap on examples scored with beam search (0 = all)")
    ap.add_argument("--max-eval", type=int, default=None,
                    help="cap on examples decoded for BLEU/ROUGE (perplexity is always full)")
    ap.add_argument("--ckpt", default=str(BEST_CKPT))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    sp = load_sp()
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model, _ = build_model_from_checkpoint(ckpt)
    model.to(device).eval()
    print(f"loaded {args.ckpt} onto {device}")

    targets = {"valid": VALID_TSV, "wiki": WIKI_TSV}
    if args.split != "both":
        targets = {args.split: targets[args.split]}

    all_metrics: dict = {}
    out_valid = None
    for name, path in targets.items():
        metrics, out = evaluate_split(model, sp, path, device, args.beam_max,
                                      max_eval=args.max_eval)
        all_metrics[name] = metrics
        print(json.dumps({name: metrics}, indent=2, ensure_ascii=False))
        if name == "valid":
            out_valid = out

    all_metrics["_n_params"] = ckpt.get("extra", {}).get("n_params")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {RESULTS_DIR / 'metrics.json'}")

    if out_valid is not None:
        _write_samples(out_valid, RESULTS_DIR / "samples.tsv", CFG.decode.n_eval_samples)
        _write_human_eval(out_valid, RESULTS_DIR / "human_eval.csv",
                          CFG.decode.human_eval_n, CFG.decode.human_eval_seed)
        # A readable example for the attention heat-map: a short source whose
        # greedy question is short and ends in the Urdu question mark.
        pick = next(
            (s for s, g in zip(out_valid["src"], out_valid["greedy"])
             if len(s.split()) <= 16 and 3 <= len(g.split()) <= 9 and g.strip().endswith("؟")),
            out_valid["src"][0],
        )
        _attention_figure(model, sp, pick, FIGURES_DIR / "attention.png")
    _write_tables(all_metrics, RESULTS_DIR / "tables.md")


if __name__ == "__main__":
    main()
