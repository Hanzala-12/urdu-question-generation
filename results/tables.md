# Results tables

## Table 1 — Dataset statistics

| | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124745 | 16824 | 210 |
| Answerable rows | 83018 | 11169 | 209 |
| Pairs after length filter | 75236 | 10043 | 177 |
| Mean source / target length | 32.59 / 11.92 | 33.23 / 12.29 | 31.69 / 11.42 |

## Table 2 — Model configuration

| Field | Value |
|---|---|
| Encoder / decoder | 2-layer BiLSTM / 2-layer LSTM |
| Layers / embedding / hidden | 2 / 256 / 512 |
| Attention | Bahdanau (additive) |
| Vocabulary size | 8000 |
| Trainable parameters | 35,505,472 |
| Optimiser / lr / schedule | Adam / 0.001 / ReduceLROnPlateau (x0.5, patience 1) |
| Batch / epochs / wall-clock / GPU | 64 / 15 / 84.5 min / Tesla T4 |

## Table 3 — Automatic metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | <unk> % |
|---|---|---|---|---|---|
| UQA valid | greedy | 5.15 | 0.2555 | 34.174 | 0.0 |
| UQA valid | beam(k=5) | 1.02 | 0.1262 | 34.174 | 0.0 |
| Wiki-UQA | greedy | 3.57 | 0.2273 | 50.774 | 0.0 |
| Wiki-UQA | beam(k=5) | 0.32 | 0.0763 | 50.774 | 0.0 |

## Table 4 — Human evaluation (50 samples)

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) | | | |
| Member 2 (% yes) | | | |
| Cohen's kappa | | | |
