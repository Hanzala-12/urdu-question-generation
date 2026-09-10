"""Task 5 - front end.

A small Gradio UI: paste an Urdu sentence, type the answer span, and see the
question the model generates with greedy and beam decoding, plus the
decoder's attention over the source.

    python app/app.py            # then open the printed local URL

The model + tokenizer are loaded once at startup from ``artifacts/``.
If ``artifacts/best.pt`` is missing, see the README for how to fetch it
from the training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow "python app/app.py" without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import ANS_CLOSE, ANS_OPEN, BEST_CKPT, CFG, SPM_MODEL  # noqa: E402
from src.decode import generate  # noqa: E402
from src.utils import (  # noqa: E402
    build_model_from_checkpoint,
    load_checkpoint,
    load_sp,
    pick_device,
)

DEVICE = pick_device()
_MODEL = None
_SP = None


def _load():
    global _MODEL, _SP
    if _MODEL is not None:
        return
    if not Path(SPM_MODEL).exists() or not Path(BEST_CKPT).exists():
        raise FileNotFoundError(
            f"Missing {SPM_MODEL} or {BEST_CKPT}. Run training (see README) "
            "or download the released checkpoint into artifacts/."
        )
    _SP = load_sp()
    ckpt = load_checkpoint(BEST_CKPT, map_location=DEVICE)
    model, _ = build_model_from_checkpoint(ckpt)
    _MODEL = model.to(DEVICE).eval()


def _mark_answer(sentence: str, answer: str) -> tuple[str, str | None]:
    """Wrap the first occurrence of ``answer`` in the sentence with <ans> tags."""
    sentence = " ".join(sentence.split())
    answer = " ".join(answer.split())
    if not answer:
        return sentence, "No answer span given - running on the raw sentence."
    idx = sentence.find(answer)
    if idx == -1:
        return sentence, f"'{answer}' not found in the sentence - running on raw text."
    marked = (
        sentence[:idx]
        + f" {ANS_OPEN} {answer} {ANS_CLOSE} "
        + sentence[idx + len(answer):]
    )
    return " ".join(marked.split()), None


from src.evaluate import _urdu_font  # noqa: E402

_FONT, _RESHAPE = _urdu_font()
_TAG = {ANS_OPEN: "«", ANS_CLOSE: "»"}


def _labels(pieces):
    out = [_TAG.get(p.replace("▁", ""), p.replace("▁", "")) or " " for p in pieces]
    return [_RESHAPE(p) for p in out]


def _heatmap(attn, src_pieces, gen_pieces):
    attn = attn[: len(gen_pieces), : len(src_pieces)]
    fig, ax = plt.subplots(
        figsize=(max(5, len(src_pieces) * 0.45), max(2.5, len(gen_pieces) * 0.45))
    )
    ax.imshow(attn.numpy(), aspect="auto", cmap="viridis")
    kw = {"fontproperties": _FONT} if _FONT is not None else {}
    ax.set_xticks(range(len(src_pieces)))
    ax.set_xticklabels(_labels(src_pieces), rotation=90, fontsize=8, **kw)
    ax.set_yticks(range(len(gen_pieces)))
    ax.set_yticklabels(_labels(gen_pieces), fontsize=8, **kw)
    ax.set_xlabel("source  (« » = <ans> tags)")
    ax.set_ylabel("generated")
    fig.tight_layout()
    return fig


def run(sentence: str, answer: str, mode: str, beam_size: int):
    _load()
    source, note = _mark_answer(sentence, answer)
    question, attn, src_pieces = generate(
        _MODEL, _SP, source, mode="beam" if mode.startswith("beam") else "greedy",
        beam_size=int(beam_size), device=DEVICE,
    )
    gen_pieces = [_SP.id_to_piece(i) for i in _SP.encode(question, out_type=int)]
    fig = _heatmap(attn, src_pieces, gen_pieces)
    status = note or "ok"
    return source, question, fig, status


with gr.Blocks(title="Urdu Question Generation") as demo:
    gr.Markdown(
        "# Urdu Question Generation\n"
        "From-scratch BiLSTM encoder-decoder with Bahdanau attention. "
        "Enter a sentence and the answer span; the model writes the question."
    )
    with gr.Row():
        with gr.Column():
            sent = gr.Textbox(label="Urdu sentence", lines=3,
                              placeholder="دریائے سندھ تقریباً 3180 کلومیٹر طویل ہے۔")
            ans = gr.Textbox(label="Answer span (exact substring)",
                             placeholder="3180 کلومیٹر")
            mode = gr.Radio(["greedy", f"beam (k={CFG.decode.beam_size})"],
                            value="greedy", label="Decoding")
            beam = gr.Slider(2, 8, value=CFG.decode.beam_size, step=1,
                             label="Beam width (beam mode)")
            go = gr.Button("Generate question", variant="primary")
        with gr.Column():
            marked_out = gr.Textbox(label="Marked source fed to the model")
            q_out = gr.Textbox(label="Generated question")
            status_out = gr.Textbox(label="Status")
    attn_out = gr.Plot(label="Attention (generated x source)")

    go.click(run, [sent, ans, mode, beam], [marked_out, q_out, attn_out, status_out])
    gr.Examples(
        [["دریائے سندھ تقریباً 3180 کلومیٹر طویل ہے۔", "3180 کلومیٹر", "greedy", 5]],
        [sent, ans, mode, beam],
    )


if __name__ == "__main__":
    demo.launch()
