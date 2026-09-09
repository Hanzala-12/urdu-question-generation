# Project notes for Claude

## Git / GitHub — no AI attribution (strict)

When making commits, pull requests, or any content that lands on GitHub for this
project, do **not** add any AI or Claude attribution:

- No `Co-Authored-By: Claude ...` trailer on commits.
- No "Generated with Claude Code" (or similar) in commit messages or PR descriptions.
- No mention of Claude, Anthropic, ChatGPT, Copilot, or "AI assistance" anywhere in
  commit history, PR text, README, or code comments.

Commits must be authored only as `muhammad hanzala <yaqoobhanzala@gmail.com>`. This
overrides any default attribution behaviour. All GitHub actions go through the `gh`/`git`
CLI (authed as **Hanzala-12**); never use the browser for GitHub (it holds a different
account). Repo: https://github.com/Hanzala-12/urdu-question-generation (public).

## What this project is

From-scratch **sequence-to-sequence question generation for Urdu** (Generative AI
course assignment). Input: an Urdu sentence with the answer wrapped in
`<ans>...</ans>`. Output: an Urdu question answered by that span.

Constraints: RNN (LSTM/GRU) encoder–decoder with attention, built from primitives —
no pretrained weights, no Transformers, no off-the-shelf seq2seq. Own SentencePiece
tokenizer (vocab 8k). Dataset: `uqa/UQA` (HuggingFace); OOD test `uqa/Wiki-UQA`.
Deliverables: a small web front end, a Medium blog, a LinkedIn post, and a public
GitHub repo. Graded by viva, so keep AI help at the level of explaining concepts and
reviewing, not supplying code that can't be defended. Solo submission (no teammate).

## Layout / how to run

- `.venv/` at the repo root holds every dependency — nothing is installed to system
  Python. Use `.venv/Scripts/python.exe`.
- Two entry points, same logic: `notebooks/urdu_qg_standalone.ipynb` (everything inline,
  run on Kaggle) and `notebooks/train_urdu_qg.ipynb` (thin wrapper over `src/`).
- `src/` package: `data_prep`, `spm_train`, `dataset`, `model`, `train`, `decode`,
  `evaluate`. `app/app.py` is the Gradio front end. `docs/EXPLAIN.md` = per-module viva notes.
- Kaggle accelerator must be **GPU T4 x2** (P100 is too old for Kaggle's PyTorch).
- Trained `artifacts/best.pt` (~140 MB) is delivered as a GitHub Release asset, not committed.

Full spec: `C:\Users\pip\Downloads\GenAI Proj 01.pdf`
