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

See [`results/tables.md`](results/tables.md) and [`results/metrics.json`](results/metrics.json).

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>`% |
|---|---|---|---|---|---|
| UQA valid | greedy | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| UQA valid | beam k=5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Wiki-UQA | greedy | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Wiki-UQA | beam k=5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

![Front end](results/figures/frontend.png)

## Write-ups

- Medium blog: _TBD_
- LinkedIn post: _TBD_

## Dataset

[UQA: Corpus for Urdu Question Answering](https://huggingface.co/datasets/uqa/UQA)
(Arif, Farid, Athar & Raza, LREC-COLING 2024) and
[Wiki-UQA](https://huggingface.co/datasets/uqa/Wiki-UQA) as an out-of-domain test.
Licensed CC-BY-4.0.

## Author

Muhammad Hanzala ([@Hanzala-12](https://github.com/Hanzala-12))
