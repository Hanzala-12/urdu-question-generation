# Urdu Question Generation (seq2seq, from scratch)

Given an Urdu sentence with the answer marked as `<ans> … </ans>`, generate the
question that the answer responds to.

> Example — `دریائے سندھ تقریباً <ans> 3180 کلومیٹر </ans> طویل ہے۔`
> → `دریائے سندھ کی لمبائی کتنی ہے؟`

An RNN encoder–decoder with attention, built from primitives (`nn.Embedding`,
`nn.LSTM`, `nn.Linear`). No pretrained weights, no Transformers, no off-the-shelf
seq2seq. Own SentencePiece subword vocabulary (8k).

## Architecture

| Part | Choice |
|---|---|
| Encoder | 2-layer **bidirectional** LSTM, embedding 256, hidden 512, dropout 0.3 |
| Bridge | linear map from the final encoder state to the decoder's initial state |
| Attention | Bahdanau (additive): `score = v·tanh(Wₑ·h_enc + W_d·s_dec)`, masked over pad |
| Decoder | 2-layer LSTM with input-feeding, linear readout over the 8k vocab |
| Training | teacher forcing, padding-masked cross-entropy, Adam, grad-clip 1.0, `ReduceLROnPlateau` |
| Decoding | greedy **and** beam search (k=5) |

## Two ways to read this project

| You want… | Look at |
|---|---|
| **One self-contained notebook** — every step inline, run top to bottom on Kaggle | [`notebooks/urdu_qg_standalone.ipynb`](notebooks/urdu_qg_standalone.ipynb) |
| **Modular `.py` code** — the same pipeline as a package, with a thin driver notebook | [`src/`](src/) + [`notebooks/train_urdu_qg.ipynb`](notebooks/train_urdu_qg.ipynb) |

Both produce the same artifacts and identical results — they are the same logic, one
flattened into a notebook and one split into modules.

## Repository layout

```
notebooks/
  urdu_qg_standalone.ipynb  — self-contained: data → tokenizer → model → train → eval → tables
  train_urdu_qg.ipynb       — thin wrapper: clones the repo and runs the src/ modules
src/            data_prep, spm_train, dataset, model, train, decode, evaluate
app/            app.py               — Gradio front end
results/        metrics.json, samples.tsv, human_eval.csv, tables.md, figures/
blog/           medium_blog.md, linkedin_post.md
docs/EXPLAIN.md per-module notes
```

## Setup (local — evaluation and the front end)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Pipeline

| Step | Command | Where |
|---|---|---|
| 1. Data prep (Task 1) | `python -m src.data_prep` | local or Kaggle (needs internet) |
| 2. Tokenizer (Task 2) | `python -m src.spm_train` | local or Kaggle |
| 3. Debug gate | `python -m src.train --debug` | 10k pairs, 1 epoch — loss must fall |
| 4. Train (Task 3) | `python -m src.train --epochs 15` | **Kaggle GPU** (~1–2 h) |
| 5. Evaluate (Task 4) | `python -m src.evaluate --split both` | local or Kaggle |
| 6. Front end (Task 5) | `python app/app.py` | local |

On Kaggle, either notebook runs steps 1–5 and zips `artifacts/` + `results/` (+ the
TSVs) to `/kaggle/working/outputs.zip`. Download it, unpack into the repo root, then
run the front end locally.

`artifacts/best.pt` (trained weights, ~140 MB) is produced by step 4 and is too large
for the repo — download it from the
[v1.0 release](https://github.com/Hanzala-12/urdu-question-generation/releases/tag/v1.0)
and place it at `artifacts/best.pt` before running evaluation or the front end.

## Results

Trained 15 epochs on a Tesla T4 (~81 min); best checkpoint at epoch 10
(validation loss 3.53, perplexity 34). Full detail in
[`results/tables.md`](results/tables.md) and [`results/metrics.json`](results/metrics.json).

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>`% |
|---|---|---|---|---|---|
| UQA valid | greedy | 5.58 | 0.260 | 34.2 | 0.0 |
| UQA valid | beam k=5 | 1.02 | 0.126 | 34.2 | 0.0 |
| Wiki-UQA | greedy | 3.57 | 0.227 | 50.8 | 0.0 |
| Wiki-UQA | beam k=5 | 0.32 | 0.076 | 50.8 | 0.0 |

Greedy and beam are scored on the same 3,000 validation examples; perplexity is
on the full split. Greedy sits just below the manual's expected 6–13 band —
reasonable for a 35 M-parameter model trained from scratch, and well clear of
"≈0 = bug" / "&gt;30 = leakage".

**Beam search hurts here, and that is expected.** Beam approximately maximises
the total sequence probability `P(question | source)`. An under-trained model
puts high probability on a few fluent, generic question templates that score
well almost regardless of the source, so wide-beam search finds and recycles
them; greedy avoids this because it commits token-by-token following the
attention distribution. This *beam-search degradation* on weak models is well
documented — larger beams lowering BLEU (Koehn & Knowles 2017, §3.3), the exact
search optimum being degenerate (Stahlberg & Byrne 2019), the effect growing
with beam width (Cohen & Beck 2019). It is why production NMT uses small beams
with length normalisation. Discussed further in the blog (§4.7).

![Front end](results/figures/frontend.png)

## Write-ups

- Medium blog: _TBD_
- LinkedIn post: _TBD_

## Dataset

[UQA: Corpus for Urdu Question Answering](https://huggingface.co/datasets/uqa/UQA)
(Arif, Farid, Athar & Raza, LREC-COLING 2024) and
[Wiki-UQA](https://huggingface.co/datasets/uqa/Wiki-UQA) as an out-of-domain test.
Licensed CC-BY-4.0.

## References

- Arif, Farid, Athar & Raza (2024). *UQA: A Corpus for Urdu Question Answering.* LREC-COLING.
- Du, Shao & Cardie (2017). *Learning to Ask: Neural Question Generation for Reading Comprehension.* ACL. — sentence-level QG setting.
- Bahdanau, Cho & Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR. — additive attention.
- Luong, Pham & Manning (2015). *Effective Approaches to Attention-based NMT.* EMNLP. — input feeding.
- Sennrich, Haddow & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL. — subword vocabularies.
- Koehn & Knowles (2017). *Six Challenges for Neural Machine Translation.* WNMT.
- Stahlberg & Byrne (2019). *On NMT Search Errors and Model Errors: Cat Got Your Tongue?* EMNLP.
- Cohen & Beck (2019). *Empirical Analysis of Beam Search Performance Degradation in Neural Sequence Models.* ICML.
- Meister, Cotterell & Vieira (2020). *If Beam Search is the Answer, What Was the Question?* EMNLP.

## Author

Muhammad Hanzala ([@Hanzala-12](https://github.com/Hanzala-12))
