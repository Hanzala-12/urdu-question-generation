# EXPLAIN — how every part works (viva notes)

Read this alongside the code. Each section: **what it does**, **why it is
like that**, and **questions you should be ready for**. Both members should
be able to answer every one without notes.

---

## `src/config.py`

**What:** one place for every path, special-token id, and hyperparameter.
Dataclasses `ModelConfig`, `TrainConfig`, `DecodeConfig` bundled into `CFG`.

**Why:** the notebook, training script, evaluation and the front end must
all agree on vocab size, hidden size and the pad/bos/eos ids. Hard-coding
them in four files is how leakage and shape bugs happen.

**Be ready for:**
- *Why is pad id 0?* SentencePiece lets us pin ids at training time; 0 for
  pad means `nn.Embedding(..., padding_idx=0)` and `CrossEntropyLoss(
  ignore_index=0)` line up with the tokenizer for free.
- *What is `SENT_DELIMS`?* Urdu full stop `U+06D4`, Urdu question mark
  `U+061F`, and `!` — the sentence boundaries used in Task 1.

---

## `src/data_prep.py` — Task 1

**What:** downloads UQA + Wiki-UQA, keeps answerable rows, finds the
*sentence* containing the answer using the character offset, wraps the
answer in `<ans> … </ans>`, drops pairs longer than 60 (source) / 25
(target) whitespace tokens, and writes `data/{train,valid,wiki}.tsv`.
Also emits length histograms and `results/data_stats.json` (Table 1).

**Why:**
- *Sentence, not paragraph.* A from-scratch RNN on ~90k pairs cannot learn
  to read a 300-token context. Restricting the input to the answer
  sentence is the setup of Du et al. (2017) and is what makes the task
  learnable.
- *Verify every offset.* `normalise_row` handles both the flat UQA schema
  (`answer` / `answer_start` / `is_impossible`) and the older nested
  SQuAD-style `answers` dict. We then check `sentence[rel:rel+len]` really
  equals the answer; mismatches (Unicode/offset drift) are counted and
  dropped rather than trusted.
- *NFC normalisation* keeps Urdu combining marks consistent so offsets and
  later tokenisation agree.

**Be ready for:**
- *What if the answer straddles two sentences?* The single-sentence offset
  check fails and the pair is dropped — reported under `offset-mismatch` /
  `no-sentence`.
- *Why 60 / 25?* From the manual; the histograms (`length_hist.png`) show
  both cover the large majority of pairs, so the cut mostly removes
  outliers that would blow up padding.
- *Could `<ans>` tags shift the token count?* Yes — they are counted; a
  source at exactly 60 real tokens plus tags can be filtered. Acceptable.

---

## `src/spm_train.py` — Task 2

**What:** concatenates training sources + targets into `sp_corpus.txt`,
trains a SentencePiece **unigram** model (`vocab_size=8000`,
`character_coverage=1.0`), registers `<ans>` / `</ans>` as
`user_defined_symbols`, pins pad/unk/bos/eos = 0/1/2/3. Prints five
tokenised examples and a round-trip check.

**Why:**
- *Unigram vs BPE.* Unigram picks the vocabulary that maximises corpus
  likelihood under a unigram LM; it tends to keep frequent whole words and
  split only rare forms, which suits Urdu's rich inflection.
- *User-defined symbols* guarantee a full tag is one token and is never
  broken, so the model always sees a clean answer boundary.
- *Trained on train only* — never on validation/Wiki — so there is no
  tokeniser-level leakage.

**Be ready for:**
- *Why `character_coverage=1.0`?* Urdu uses a compact script; we can afford
  to keep every character rather than map rares to `<unk>`.
- *What does `<unk>` look like at decode time?* SentencePiece renders it as
  `U+2047` (`⁇`); the `<unk>%` metric counts those.
- *Show an example of morphology splitting.* See
  `results/tokenizer_examples.txt` — e.g. an oblique plural ending or the
  izafat `ی` coming off as its own piece.

---

## `src/dataset.py`

**What:** `QGDataset` reads a TSV and tokenises each side on `__getitem__`;
targets get `<s> … </s>`. `collate_batch` sorts a batch by source length
(descending), pads to `PAD_ID`, and returns `src`, `tgt`, `src_lengths`,
and `src_key_padding_mask` (True on pad).

**Why:**
- *Sort + lengths* so the encoder can `pack_padded_sequence` with
  `enforce_sorted=True` — the LSTM then never computes over pad steps.
- *`tgt[:, :-1]` in, `tgt[:, 1:]` out* is standard teacher-forced shift:
  predict token *t+1* from tokens `≤ t`.

**Be ready for:**
- *Why a bool pad mask separately from lengths?* Lengths drive RNN packing;
  the mask drives attention (`masked_fill(-inf)` before softmax).

---

## `src/model.py` — Task 3

**What:** `Encoder` (embed → 2-layer BiLSTM → linear bridge to the decoder
state), `BahdanauAttention` (additive scoring, pad-masked softmax),
`Decoder` (embed + previous context → 2-layer LSTM → attention → linear
readout over `[dec_state ; context ; input_emb]`), and `Seq2Seq` which
runs the teacher-forced loop and returns `logits [B,T-1,V]` and
`attn [B,T-1,S]`.

**Why:**
- *Bidirectional encoder* — a question word often depends on material on
  both sides of the answer span.
- *Bridge with `tanh`* — encoder hidden is `2H` (both directions); the
  decoder is `H`. A learned linear + `tanh` gives a sane starting state
  instead of zeros.
- *Input feeding* (concat previous context into the next decoder input) —
  lets the decoder know what it already attended to; standard since Luong
  et al. (2015).
- *Readout over `[state ; context ; emb]`* — the projection sees the RNN
  state, what it just read from the source, and the token it just emitted.
- *Attention masking* — pad positions get `-inf` before softmax so they
  receive exactly zero weight.

**Be ready for:**
- *Derive the additive score.* `e_{t,i} = vᵀ tanh(W_e h_i + W_d s_t)`, then
  `α_{t} = softmax(e_t)`, `c_t = Σ_i α_{t,i} h_i`.
- *Where does teacher forcing happen?* In `Seq2Seq.forward`: with
  probability `teacher_forcing` the next input is the gold token, else the
  model's own `argmax`. Eval always uses gold (`=1.0`) so the loss/PPL is
  comparable across runs.
- *Parameter count?* Printed by `count_parameters` and recorded in Table 2.
- *Why LSTM not GRU?* Either is allowed; LSTM's separate cell state is a
  small, safe default. `config.py` has the switch; only the LSTM path is
  implemented here.

---

## `src/train.py` — Task 3

**What:** builds loaders, model, `Adam`, `ReduceLROnPlateau`,
`CrossEntropyLoss(ignore_index=PAD_ID)`. `run_epoch` does one pass
(train if given an optimizer, else eval with full teacher forcing).
Per epoch: log `train_loss, val_loss, val_ppl, lr` to
`results/loss_log.csv`, save `last.pt` always and `best.pt` when val loss
improves. Ends by plotting `loss_curve.png`. `--debug` = 10k pairs, 1
epoch (the manual's debug gate).

**Why:**
- *Token-weighted epoch loss* — accumulate `loss * n_tokens` and divide by
  total non-pad tokens, so batches with more real tokens count more.
- *`best.pt` is weights only* — smaller, and the front end never needs the
  optimizer. `last.pt` carries optimizer + epoch for `--resume`.
- *Grad clip 1.0* — RNN gradients can spike; clipping the global norm
  keeps training stable.
- *PPL = exp(mean CE)* — reported per epoch as an interpretable number.

**Be ready for:**
- *Why does the debug gate exist?* If the loss does not drop on 10k pairs
  in one epoch, something is wrong (bad masking, wrong shift, LR) — cheaper
  to catch in 3 minutes than after an hour.
- *What's your teacher-forcing ratio and why not 1.0?* `0.5` — some
  exposure to its own predictions reduces train/inference mismatch
  (exposure bias) without making early training unstable.

---

## `src/decode.py`

**What:** `greedy_decode` (whole batch, argmax each step, stop at `</s>`)
and `beam_search` (one source, beam as the batch dim, log-prob sum with a
length penalty, keep first `k` finished hypotheses). Both also return the
attention matrix for the chosen hypothesis. `generate` wraps raw text →
`(question, attn, source_pieces)` for the front end.

**Why:**
- *Greedy* commits to the argmax each step, following the attention
  distribution — it stays anchored to the source. *Beam* keeps `k`
  hypotheses and approximately maximises the **whole-sequence** log-prob.
- *Length penalty* `score / (len^α)`, α = 0.6 — beam compares sequences of
  different lengths; without normalisation it prefers very short ones.
- We also rank the still-running beams at the end, so a beam that hit `</s>`
  early cannot beat a better, longer, unfinished one.

**Be ready for:**
- *Beam mechanics:* at each step score every (beam × vocab) extension,
  take the global top-k, remember which beam each came from, carry that
  beam's LSTM state/context forward.
- *Our result: beam BLEU < greedy BLEU.* This is **beam-search
  degradation** on a weak model. Beam maximises `P(q | source)`; an
  under-trained model puts high probability on a few generic question
  templates that are fluent almost regardless of the source, so beam finds
  and recycles them (you can see identical beam outputs for different
  sources in `results/samples.tsv`). Greedy avoids it by staying local.
  Documented in Koehn & Knowles (2017, §3.3), Stahlberg & Byrne (2019),
  Cohen & Beck (2019). This is the answer to the manual's §4.7 question
  "where does beam search hurt?".
- *Sanity check:* `beam_search(..., k=1)` returns exactly the greedy output —
  proof the beam machinery itself is correct.

---

## `src/evaluate.py` — Task 4

**What:** loads `best.pt`, decodes UQA-valid and Wiki-UQA with greedy and
beam, computes **BLEU-4** (sacrebleu, corpus), **ROUGE-L** (mean F),
**perplexity** (teacher-forced CE via `run_epoch`), **`<unk>`%** (share of
generated ids equal to `UNK_ID`). Writes `metrics.json`, `samples.tsv`
(≥ 50: source, reference, greedy, beam), `figures/attention.png`, and a
draft `tables.md`.

**Why:**
- *Corpus BLEU on detokenised text* — matches how sacrebleu is meant to be
  used and avoids tokeniser-dependent scores.
- *Beam capped (`--beam-max`)* — beam is one-at-a-time; scoring a few
  thousand is enough for a stable number. State the cap in the report.

**Be ready for:**
- *Sanity band:* BLEU-4 ≈ 6–13 on UQA-valid, lower on Wiki-UQA. ~0 ⇒ bug;
  > 30 ⇒ train/val leakage.
- *Why does Wiki-UQA score lower?* It is out-of-domain and human-written,
  while training data is translated SQuAD — discuss what that says about
  translated corpora.

---

## `app/app.py` — Task 5

**What:** Streamlit UI (`streamlit run app/app.py`). Enter a sentence + the
exact answer substring; the app wraps the first match in `<ans>` tags, runs
greedy or beam decoding, and shows the question plus an attention heat-map
(generated × source pieces). Model + tokenizer load once via
`@st.cache_resource`, from `artifacts/`.

**Be ready for:**
- *What if the answer string isn't in the sentence?* The app warns and
  runs on the raw text.
- *What does the heat-map show?* For each generated piece, the softmax
  weights over source pieces — you should see the question word attend to
  the answer span and its head noun.
