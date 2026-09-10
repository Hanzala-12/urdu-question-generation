"""Task 5 - front end (Streamlit).

    streamlit run app/app.py

Paste an Urdu sentence, mark the answer span, and read the question the model
generates with greedy and beam decoding, alongside the decoder's attention
over the source. The model + tokenizer load once from ``artifacts/`` (see the
README for how to fetch ``best.pt`` from the release).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.config import ANS_CLOSE, ANS_OPEN, BEST_CKPT, CFG, SPM_MODEL
from src.decode import generate
from src.evaluate import _urdu_font
from src.utils import (
    build_model_from_checkpoint,
    load_checkpoint,
    load_sp,
    pick_device,
)

st.set_page_config(page_title="Urdu Question Generation", page_icon="؟", layout="centered")

DEVICE = pick_device()
_FONT, _RESHAPE = _urdu_font()
_TAG = {ANS_OPEN: "«", ANS_CLOSE: "»"}

EXAMPLES = [
    ("دریائے سندھ تقریباً 3180 کلومیٹر طویل ہے۔", "3180 کلومیٹر"),
    ("قائد اعظم محمد علی جناح 1876 میں کراچی میں پیدا ہوئے۔", "کراچی"),
    ("ماؤنٹ ایورسٹ دنیا کی سب سے بلند پہاڑی ہے، جو نیپال میں واقع ہے۔", "نیپال"),
]


@st.cache_resource(show_spinner="Loading model …")
def _load():
    if not Path(SPM_MODEL).exists() or not Path(BEST_CKPT).exists():
        st.error(
            f"Missing {SPM_MODEL} or {BEST_CKPT}. Download `best.pt` from the "
            "v1.0 release into `artifacts/` (see README)."
        )
        st.stop()
    sp = load_sp()
    ckpt = load_checkpoint(BEST_CKPT, map_location=DEVICE)
    model, _ = build_model_from_checkpoint(ckpt)
    return sp, model.to(DEVICE).eval(), ckpt.get("extra", {})


def _mark(sentence: str, answer: str) -> tuple[str, str | None]:
    sentence, answer = " ".join(sentence.split()), " ".join(answer.split())
    if not answer:
        return sentence, "No answer span given — running on the raw sentence."
    i = sentence.find(answer)
    if i == -1:
        return sentence, f"'{answer}' not found in the sentence — running on raw text."
    marked = sentence[:i] + f" {ANS_OPEN} {answer} {ANS_CLOSE} " + sentence[i + len(answer):]
    return " ".join(marked.split()), None


def _heatmap(attn, src_ids, gen_text, sp):
    src_pieces = [_TAG.get(sp.id_to_piece(i), sp.id_to_piece(i).replace("▁", "")) or " " for i in src_ids]
    gen_pieces = [sp.id_to_piece(i).replace("▁", "") or " " for i in sp.encode(gen_text, out_type=int)]
    a = attn[: len(gen_pieces), : len(src_pieces)]
    fig, ax = plt.subplots(figsize=(max(6, len(src_pieces) * 0.42), max(3, len(gen_pieces) * 0.42)))
    im = ax.imshow(a.numpy(), aspect="auto", cmap="viridis")
    kw = {"fontproperties": _FONT} if _FONT is not None else {}
    ax.set_xticks(range(len(src_pieces))); ax.set_xticklabels([_RESHAPE(p) for p in src_pieces], rotation=90, fontsize=8, **kw)
    ax.set_yticks(range(len(gen_pieces))); ax.set_yticklabels([_RESHAPE(p) for p in gen_pieces], fontsize=8, **kw)
    ax.set_xlabel("source  (« » = <ans> tags)"); ax.set_ylabel("generated")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


sp, model, meta = _load()

st.title("Urdu Question Generation")
st.caption(
    "From-scratch 2-layer BiLSTM encoder – decoder with Bahdanau attention "
    f"({meta.get('n_params', 35_505_472):,} parameters). Mark the answer in an "
    "Urdu sentence; the model writes the question."
)

with st.sidebar:
    st.subheader("Decoding")
    mode = st.radio("Strategy", ["greedy", "beam"], horizontal=True)
    beam_k = st.slider("Beam width", 2, 8, CFG.decode.beam_size, disabled=(mode == "greedy"))
    st.divider()
    st.subheader("Examples")
    for j, (s, a) in enumerate(EXAMPLES):
        if st.button(s[:38] + "…", key=f"ex{j}", use_container_width=True):
            st.session_state.update(sentence=s, answer=a)

sentence = st.text_area("Urdu sentence", key="sentence", height=110,
                        value=st.session_state.get("sentence", EXAMPLES[0][0]))
answer = st.text_input("Answer span (exact substring)", key="answer",
                       value=st.session_state.get("answer", EXAMPLES[0][1]))
go = st.button("Generate question", type="primary", use_container_width=True)

if go:
    source, note = _mark(sentence, answer)
    if note:
        st.warning(note)
    ids = sp.encode(source, out_type=int)[: CFG.train.max_src_len]
    st.markdown("**Marked source**")
    st.code(source, language=None)

    greedy_q, g_attn, _ = generate(model, sp, source, mode="greedy", device=DEVICE)
    beam_q, b_attn, _ = generate(model, sp, source, mode="beam", beam_size=int(beam_k), device=DEVICE)
    if mode == "beam":
        main_q, main_attn, label = beam_q, b_attn, f"beam search, k = {int(beam_k)}"
    else:
        main_q, main_attn, label = greedy_q, g_attn, "greedy decoding"

    st.markdown("### Generated question")
    st.markdown(f"## {main_q}")
    st.caption(label)

    st.markdown("### Attention")
    st.pyplot(_heatmap(main_attn, ids, main_q, sp), use_container_width=False)

    with st.expander("Compare decoding strategies"):
        st.table({
            "decoding": ["greedy", f"beam (k={int(beam_k)})"],
            "generated question": [greedy_q, beam_q],
        })
else:
    st.info("Set a sentence and answer span, then press **Generate question**.")
