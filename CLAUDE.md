# Project notes for Claude

## Git / GitHub — no AI attribution (strict)

When making commits, pull requests, or any content that lands on GitHub for this
project, do **not** add any AI or Claude attribution:

- No `Co-Authored-By: Claude ...` trailer on commits.
- No "Generated with Claude Code" (or similar) in commit messages or PR descriptions.
- No mention of Claude, Anthropic, ChatGPT, Copilot, or "AI assistance" anywhere in
  commit history, PR text, README, or code comments.

Commits must be authored as the human team members only. This overrides any default
attribution behaviour.

## What this project is

From-scratch **sequence-to-sequence question generation for Urdu** (Generative AI
course assignment). Input: an Urdu sentence with the answer wrapped in
`<ans>...</ans>`. Output: an Urdu question answered by that span.

Constraints: RNN (LSTM/GRU) encoder–decoder with attention, built from primitives —
no pretrained weights, no Transformers, no off-the-shelf seq2seq. Own SentencePiece
tokenizer (vocab 8k). Dataset: `uqa/UQA` (HuggingFace); OOD test `uqa/Wiki-UQA`.
Deliverables include a small web front end, a Medium blog, a LinkedIn post, and a
public GitHub repo with commits from both group members. Graded by viva — both
members must be able to explain every part, so keep AI help at the level of
explaining concepts and reviewing, not supplying code that can't be defended.

Full spec: `C:\Users\pip\Downloads\GenAI Proj 01.pdf`
