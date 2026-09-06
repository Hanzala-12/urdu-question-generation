"""Task 3 - a from-scratch RNN encoder-decoder with attention.

Built only from ``nn.Embedding``, ``nn.LSTM``, ``nn.Linear``, ``nn.Dropout``.
No pretrained weights, no Transformer blocks, no off-the-shelf seq2seq.

    Encoder :  embedding -> 2-layer bidirectional LSTM
    Bridge  :  linear map from the final encoder state to the decoder's
               initial hidden/cell state
    Attention: Bahdanau (additive) - score = v . tanh(W_e h_enc + W_d s_dec)
    Decoder :  embedding + input-feeding -> 2-layer LSTM -> attention ->
               linear readout over the vocabulary

Shapes (B batch, S source len, T target len, H hidden, E embedding):
    encoder outputs      [B, S, 2H]
    decoder hidden/cell   [layers, B, H]
    context vector        [B, 2H]
    step logits           [B, vocab]
    step attention        [B, S]
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import BOS_ID, PAD_ID, Config, ModelConfig


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, m: ModelConfig) -> None:
        super().__init__()
        assert m.rnn_cell == "lstm", "only the LSTM path is implemented"
        self.hidden_size = m.hidden_size
        self.dec_layers = m.dec_layers
        self.embedding = nn.Embedding(m.vocab_size, m.emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(m.dropout)
        self.rnn = nn.LSTM(
            m.emb_dim,
            m.hidden_size,
            num_layers=m.enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=m.dropout if m.enc_layers > 1 else 0.0,
        )
        # Merge the final forward+backward states into the decoder's init.
        self.bridge_h = nn.Linear(2 * m.hidden_size, m.hidden_size)
        self.bridge_c = nn.Linear(2 * m.hidden_size, m.hidden_size)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        emb = self.dropout(self.embedding(src))                     # [B, S, E]
        packed = pack_padded_sequence(
            emb, src_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        outputs, (h, c) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=src.size(1)
        )                                                          # [B, S, 2H]

        # h, c: [enc_layers * 2, B, H] -> take the last layer's two directions.
        last_h = torch.cat([h[-2], h[-1]], dim=-1)                 # [B, 2H]
        last_c = torch.cat([c[-2], c[-1]], dim=-1)                 # [B, 2H]
        dec_h0 = torch.tanh(self.bridge_h(last_h))                 # [B, H]
        dec_c0 = torch.tanh(self.bridge_c(last_c))                 # [B, H]
        # Replicate as the starting state for every decoder layer.
        dec_h0 = dec_h0.unsqueeze(0).repeat(self.dec_layers, 1, 1)  # [L, B, H]
        dec_c0 = dec_c0.unsqueeze(0).repeat(self.dec_layers, 1, 1)
        return outputs, (dec_h0.contiguous(), dec_c0.contiguous())


# ---------------------------------------------------------------------------
# Bahdanau (additive) attention
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int) -> None:
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def project_keys(self, enc_outputs: torch.Tensor) -> torch.Tensor:
        """W_enc . h_enc for every source position - computed once per batch."""
        return self.W_enc(enc_outputs)                             # [B, S, H]

    def forward(
        self,
        dec_state: torch.Tensor,        # [B, H]
        enc_outputs: torch.Tensor,      # [B, S, 2H]
        enc_keys: torch.Tensor,         # [B, S, H]  (from project_keys)
        pad_mask: torch.Tensor,         # [B, S]  True where PAD
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.W_dec(dec_state).unsqueeze(1)                 # [B, 1, H]
        scores = self.v(torch.tanh(enc_keys + query)).squeeze(-1)  # [B, S]
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                    # [B, S]
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # [B,2H]
        return context, weights


# ---------------------------------------------------------------------------
# Decoder (single time step)
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, m: ModelConfig) -> None:
        super().__init__()
        enc_dim = 2 * m.hidden_size
        self.embedding = nn.Embedding(m.vocab_size, m.emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(m.dropout)
        self.rnn = nn.LSTM(
            m.emb_dim + enc_dim,           # input feeding: [emb ; prev context]
            m.hidden_size,
            num_layers=m.dec_layers,
            batch_first=True,
            dropout=m.dropout if m.dec_layers > 1 else 0.0,
        )
        self.attn = BahdanauAttention(enc_dim, m.hidden_size)
        self.out = nn.Linear(m.hidden_size + enc_dim + m.emb_dim, m.vocab_size)

    def step(
        self,
        input_id: torch.Tensor,                       # [B]
        state: tuple[torch.Tensor, torch.Tensor],     # ([L,B,H], [L,B,H])
        prev_context: torch.Tensor,                   # [B, 2H]
        enc_outputs: torch.Tensor,                    # [B, S, 2H]
        enc_keys: torch.Tensor,                       # [B, S, H]
        pad_mask: torch.Tensor,                       # [B, S]
    ) -> tuple[torch.Tensor, tuple, torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(input_id))              # [B, E]
        rnn_in = torch.cat([emb, prev_context], dim=-1).unsqueeze(1)  # [B,1,E+2H]
        rnn_out, state = self.rnn(rnn_in, state)
        dec_state = rnn_out.squeeze(1)                            # [B, H]
        context, attn = self.attn(dec_state, enc_outputs, enc_keys, pad_mask)
        readout = torch.cat([dec_state, context, emb], dim=-1)
        logits = self.out(self.dropout(readout))                 # [B, vocab]
        return logits, state, context, attn


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, src, src_lengths):
        enc_outputs, state = self.encoder(src, src_lengths)
        enc_keys = self.decoder.attn.project_keys(enc_outputs)
        return enc_outputs, enc_keys, state

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        pad_mask: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training / scoring pass over a whole target sequence.

        Returns ``(logits, attn)`` with logits ``[B, T-1, vocab]`` aligned to
        ``tgt[:, 1:]`` and attention ``[B, T-1, S]``.
        """
        B, T = tgt.size()
        enc_outputs, enc_keys, state = self.encode(src, src_lengths)
        context = src.new_zeros((B, 2 * self.cfg.hidden_size), dtype=enc_outputs.dtype)
        input_id = tgt[:, 0]                                      # <s>
        logits_seq, attn_seq = [], []
        for t in range(1, T):
            logits, state, context, attn = self.decoder.step(
                input_id, state, context, enc_outputs, enc_keys, pad_mask
            )
            logits_seq.append(logits)
            attn_seq.append(attn)
            use_gold = random.random() < teacher_forcing
            input_id = tgt[:, t] if use_gold else logits.argmax(-1)
        return torch.stack(logits_seq, dim=1), torch.stack(attn_seq, dim=1)


def build_model(cfg: Config | ModelConfig) -> Seq2Seq:
    m = cfg.model if isinstance(cfg, Config) else cfg
    return Seq2Seq(m)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":  # tiny smoke test, no data needed
    from src.config import CFG

    torch.manual_seed(0)
    model = build_model(CFG)
    print(f"trainable parameters: {count_parameters(model):,}")
    B, S, T = 4, 12, 9
    src = torch.randint(4, CFG.model.vocab_size, (B, S))
    src[:, -3:] = PAD_ID
    src_lengths = torch.tensor([12, 11, 10, 9])
    tgt = torch.randint(4, CFG.model.vocab_size, (B, T))
    tgt[:, 0] = BOS_ID
    logits, attn = model(src, src_lengths, src.eq(PAD_ID), tgt, teacher_forcing=1.0)
    print("logits", tuple(logits.shape), "attn", tuple(attn.shape))
    assert logits.shape == (B, T - 1, CFG.model.vocab_size)
    assert attn.shape == (B, T - 1, S)
    print("ok")
