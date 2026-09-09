"""Greedy and beam-search decoding, shared by evaluation and the front end.

* ``greedy_decode`` runs a whole batch at once - fast, used for bulk eval.
* ``beam_search`` decodes one source at a time with the beam as the batch
  dimension - used for the "beam" column in results and in the UI.

Both return the generated **text** and the attention matrix
(``[gen_len, src_len]``) for the chosen hypothesis, so the front end can
draw a heat-map.
"""

from __future__ import annotations

import torch

from src.config import BOS_ID, EOS_ID, PAD_ID, CFG
from src.model import Seq2Seq


@torch.no_grad()
def greedy_decode(
    model: Seq2Seq,
    src: torch.Tensor,               # [B, S]  (already tokenised + padded)
    src_lengths: torch.Tensor,       # [B]
    pad_mask: torch.Tensor,          # [B, S]
    max_len: int = CFG.decode.max_decode_len,
) -> tuple[list[list[int]], torch.Tensor]:
    """Return ``(list_of_id_lists, attn)`` with attn ``[B, max_len, S]``."""
    model.eval()
    device = src.device
    B = src.size(0)
    enc_outputs, enc_keys, state = model.encode(src, src_lengths)
    context = torch.zeros(B, 2 * model.cfg.hidden_size, device=device)
    input_id = torch.full((B,), BOS_ID, dtype=torch.long, device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    tokens: list[list[int]] = [[] for _ in range(B)]
    attn_steps = []
    for _ in range(max_len):
        logits, state, context, attn = model.decoder.step(
            input_id, state, context, enc_outputs, enc_keys, pad_mask
        )
        input_id = logits.argmax(-1)                      # [B]
        attn_steps.append(attn)
        for b in range(B):
            if not finished[b]:
                tokens[b].append(int(input_id[b]))
        finished |= input_id.eq(EOS_ID)
        if bool(finished.all()):
            break

    attn = torch.stack(attn_steps, dim=1)                 # [B, gen_len, S]
    tokens = [_trim_eos(t) for t in tokens]
    return tokens, attn


@torch.no_grad()
def beam_search(
    model: Seq2Seq,
    src: torch.Tensor,               # [1, S]
    src_lengths: torch.Tensor,       # [1]
    pad_mask: torch.Tensor,          # [1, S]
    beam_size: int = CFG.decode.beam_size,
    max_len: int = CFG.decode.max_decode_len,
    length_penalty: float = CFG.decode.length_penalty,
) -> tuple[list[int], torch.Tensor]:
    """Decode a single source. Return ``(id_list, attn[gen_len, S])``."""
    model.eval()
    device = src.device
    assert src.size(0) == 1, "beam_search decodes one source at a time"
    S = src.size(1)
    k = beam_size

    enc_outputs, enc_keys, state = model.encode(src, src_lengths)
    # Expand everything to the beam width.
    enc_outputs = enc_outputs.expand(k, S, -1).contiguous()
    enc_keys = enc_keys.expand(k, S, -1).contiguous()
    pad_mask_k = pad_mask.expand(k, S).contiguous()
    state = (state[0].repeat(1, k, 1).contiguous(),
             state[1].repeat(1, k, 1).contiguous())
    context = torch.zeros(k, 2 * model.cfg.hidden_size, device=device)

    seqs = torch.full((k, 1), BOS_ID, dtype=torch.long, device=device)
    scores = torch.full((k,), float("-inf"), device=device)
    scores[0] = 0.0
    attn_hist = torch.zeros(k, 0, S, device=device)
    finished: list[tuple[float, list[int], torch.Tensor]] = []

    for _ in range(max_len):
        logits, state, context, attn = model.decoder.step(
            seqs[:, -1], state, context, enc_outputs, enc_keys, pad_mask_k
        )
        logp = torch.log_softmax(logits, dim=-1)          # [k, V]
        cand = scores.unsqueeze(1) + logp                 # [k, V]
        flat = cand.view(-1)
        top_scores, top_idx = flat.topk(k)
        beam_idx = torch.div(top_idx, logp.size(-1), rounding_mode="floor")
        token_idx = top_idx % logp.size(-1)

        seqs = torch.cat([seqs[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores
        state = (state[0][:, beam_idx, :].contiguous(),
                 state[1][:, beam_idx, :].contiguous())
        context = context[beam_idx]
        attn_hist = torch.cat(
            [attn_hist[beam_idx], attn[beam_idx].unsqueeze(1)], dim=1
        )

        for b in range(k):
            if int(token_idx[b]) == EOS_ID:
                seq = seqs[b, 1:-1].tolist()
                lp = (len(seq) + 1) ** length_penalty
                finished.append((scores[b].item() / lp, seq, attn_hist[b, :-1]))
                scores[b] = float("-inf")          # do not extend a finished beam
        if len(finished) >= k:
            break

    # Also consider the beams still running - otherwise a single short beam
    # that hit EOS early wins over better, longer, unfinished hypotheses.
    for b in range(k):
        if scores[b].item() != float("-inf"):
            seq = seqs[b, 1:].tolist()
            lp = (len(seq) + 1) ** length_penalty
            finished.append((scores[b].item() / lp, seq, attn_hist[b]))

    finished.sort(key=lambda x: x[0], reverse=True)
    _, best_seq, best_attn = finished[0]
    best_seq = _trim_eos(best_seq)
    return best_seq, best_attn[: len(best_seq)]


def _trim_eos(ids: list[int]) -> list[int]:
    out: list[int] = []
    for i in ids:
        if i == EOS_ID:
            break
        if i not in (PAD_ID, BOS_ID):
            out.append(i)
    return out


def ids_to_text(sp, ids: list[int]) -> str:
    return sp.decode(_trim_eos(ids))


# --- convenience wrappers used by the front end -----------------------------
@torch.no_grad()
def generate(
    model: Seq2Seq,
    sp,
    source_text: str,
    *,
    mode: str = "greedy",
    beam_size: int = CFG.decode.beam_size,
    device: torch.device | None = None,
) -> tuple[str, torch.Tensor, list[str]]:
    """High-level: raw source string -> (question, attn, source_pieces)."""
    device = device or next(model.parameters()).device
    ids = sp.encode(source_text, out_type=int)[: CFG.train.max_src_len]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(ids)], dtype=torch.long)
    pad_mask = src.eq(PAD_ID)
    if mode == "beam":
        seq, attn = beam_search(model, src, src_lengths, pad_mask, beam_size=beam_size)
    else:
        seqs, attn_b = greedy_decode(model, src, src_lengths, pad_mask)
        seq, attn = seqs[0], attn_b[0, : len(seqs[0])]
    return sp.decode(seq), attn.cpu(), [sp.id_to_piece(i) for i in ids]
