# Teaching an RNN to ask questions in Urdu — from scratch

*Building a sequence-to-sequence question generator for Urdu with no pretrained
weights, no Transformers, and a vocabulary learned from the data.*

---

## The problem

Give a model an Urdu sentence with the answer marked inside it, and ask it to
produce the question that answer responds to. This is what a teacher does when
turning a paragraph into practice questions:

> Input:  `دریائے سندھ تقریباً <ans> 3180 کلومیٹر </ans> طویل ہے۔`
> Output: `دریائے سندھ کی لمبائی کتنی ہے؟`  ("How long is the Indus?")

It is a classic sequence-to-sequence task — variable-length in, variable-length
out, no fixed alignment — the same family as translation and summarisation. The
constraint that made it interesting: **build the whole thing from primitives.**
An RNN encoder–decoder with attention, `nn.Embedding` / `nn.LSTM` / `nn.Linear`
only. No pretrained anything.

## Data: only the sentence, not the paragraph

The dataset is **UQA**, a translation of SQuAD 2.0 into Urdu that keeps the
character offset of every answer. It has ~142k context–question–answer rows.

A from-scratch RNN cannot learn to read a 300-token paragraph, so — following
Du et al. (2017) — we throw the paragraph away and keep only the **sentence that
contains the answer**. Concretely: split the context on the Urdu full stop
(`U+06D4`), Urdu question mark (`U+061F`) and `!`; find the sentence covering the
answer's character offset; wrap the span in `<ans> … </ans>`; use that sentence
as the source and the `question` field as the target. Drop pairs longer than 60
(source) / 25 (target) whitespace tokens.

| | Train | Validation | Wiki-UQA (OOD) |
|---|---|---|---|
| Raw rows | 124,745 | 16,824 | 210 |
| Answerable | 83,018 | 11,169 | 209 |
| Pairs after length filter | **75,236** | **10,043** | **177** |

About 40% of rows are unanswerable (dropped), and the length filter removes
outliers where a single "sentence" was really two clauses glued together.

## Tokenizer: subwords for a morphologically rich language

We learn an 8k **SentencePiece unigram** vocabulary on the training text only.
`<ans>` and `</ans>` are registered as user-defined symbols so a tag is always
exactly one token; `<pad> <unk> <s> </s>` are pinned to ids 0–3.

Unigram (rather than BPE) tends to keep frequent whole words and split only the
rarer inflected forms. On Urdu that shows up clearly:

- function words stay whole — `کے`, `میں`, `سے`
- inflected / compound forms break into stem + affix — `مقابل وں`
  (competitions: stem + oblique-plural `وں`), `پر ورش` (upbringing),
  `گلوکار ہ` (female singer: stem + `ہ`)
- rare proper nouns fragment into near-character pieces — `ڈسٹ نی` (Destiny)

That last point matters later: the model's copy errors and truncations almost
always happen on these fragmented names.

## Model

Nothing exotic — the point was to implement each piece by hand.

| Part | Choice |
|---|---|
| Encoder | 2-layer **bidirectional** LSTM, embedding 256, hidden 512, dropout 0.3 |
| Bridge | linear + `tanh` from the final encoder state to the decoder's initial state |
| Attention | Bahdanau (additive): `eₜᵢ = vᵀ tanh(Wₑ hᵢ + W_d sₜ)`, softmax over source positions, padding masked to `-∞` |
| Decoder | 2-layer LSTM with input feeding (previous context vector concatenated to the input), linear readout over `[sₜ ; cₜ ; embₜ]` |

**35.5M trainable parameters.** Training: teacher forcing (ratio 0.5),
padding-masked cross-entropy, Adam (lr 1e-3) with `ReduceLROnPlateau`, gradient
clipping at 1.0, 15 epochs, batch 64. About **81 minutes on a single Tesla T4.**

![Training and validation loss](../results/figures/loss_curve.png)

Validation loss bottoms out at **epoch 10** (loss 3.53, perplexity 34) and then
the model starts to overfit — training loss keeps falling while validation
creeps back up. We keep the epoch-10 checkpoint.

## Results

Greedy and beam (k=5) are scored on the same 3,000 validation examples;
perplexity is on the full split.

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | `<unk>`% |
|---|---|---|---|---|---|
| UQA valid | greedy | **5.58** | 0.260 | 34.2 | 0.0 |
| UQA valid | beam k=5 | 1.02 | 0.126 | 34.2 | 0.0 |
| Wiki-UQA | greedy | 3.57 | 0.227 | 50.8 | 0.0 |
| Wiki-UQA | beam k=5 | 0.32 | 0.076 | 50.8 | 0.0 |

Du et al. (2017) reached BLEU-4 ≈ 12 on English SQuAD with a bigger model and
more data. Our greedy 5.58 sits just below the 6–13 band you would expect for a
35M model trained from scratch — clearly not "≈0 = a bug", and nowhere near
">30 = train/validation leakage". The `<unk>` rate is 0 because
`character_coverage=1.0` gives every rare glyph its own piece.

### Three real outputs

1. **Answer `911`.** Reference: *"ڈچی آف نارمنڈی کی بنیاد کب رکھی گئی تھی؟"*
   Model: *"نورمنڈی ڈچ کب شروع ہوا؟"* — different wording, same question. The
   question word `کب` ("when") is right for a date.
2. **Answer `1185`.** Reference: *"نارمنز نے ڈیرراچیم پر کب حملہ کیا؟"*
   Model: *"کس سال میں بازنطینیوں نے را پر حملہ کیا؟"* — "in which year… attacked",
   correct question type, keeps the verb "attacked".
3. **Answer `بوہیمنڈ` (a name).** Reference: *"رابرٹ کا بیٹا کون تھا؟"*
   Model: *"کون سا کے بیٹے کا کمانڈر کون تھا؟"* — `کون` ("who") and `بیٹے` ("son")
   are both there, but the sentence is garbled.

### Five failure cases

| # | Model output (abridged) | Failure type |
|---|---|---|
| 1 | *"نپولین نے کتنے مردوں کی قیادت کی؟"* (answer was `30,000`) | **hallucinated entity** — "Napoleon" is not in the source |
| 2 | *"…شوپن نے کب واپس لے لی؟"* (a date question) | **hallucinated entity** — "Chopin" appears from nowhere |
| 3 | *"…وفاداری کے وفاداری کے وفاداری کے…"* | **repetition** — the decoder loops on a phrase |
| 4 | *"را  کہاں سے آئے تھے؟"* | **copied fragment** — `را` is half of a fragmented proper noun; also the wrong question word |
| 5 | *"زیادہ تر زیادہ تر لوگ زیادہ تر لوگ کس قسم کی…"* | **copied phrase + repetition** — "most" is lifted straight from the source and repeated |

The pattern: the model reliably gets the **question word** right (especially
`کب` / `کس سال` for dates and `کتنے` for quantities), and it degrades on
**content** — hallucinating names, repeating, or copying fragmented tokens.

### Attention

![Attention heat-map](../results/figures/attention.png)

For each generated piece the heat-map shows the softmax weights over the source
pieces. The generated content words concentrate sharply on the matching source
words, and the `«` `»` (the `<ans>` tags) receive almost no weight — the model
learned they are markers, not content.

## Discussion

**Which question words work?** Temporal (`کب`, `کس سال`) and quantity (`کتنے`)
are the most reliable — they are frequent in the data and cued strongly by a
numeric answer span. `کون` ("who") is hit-or-miss, and the model rarely gets the
finer distinctions (`کہاں` vs `کس ملک میں`) right.

**Where does beam search help, and where does it hurt?** It hurts — badly. Beam
BLEU-4 is 1.02 against greedy's 5.58. Beam approximately maximises the whole-
sequence probability `P(question | source)`, and an under-trained model puts
high probability on a handful of fluent, generic question templates that score
well *almost regardless of the source*. So wide-beam search finds and recycles
them: several different source sentences all produce
*"فرانس کے خطے کو کب نام دیا گیا تھا؟"*. Greedy avoids this because it commits
token by token, following the attention distribution, so it stays anchored to
the answer. This is the well-documented **beam-search degradation** on weak
models — larger beams lowering BLEU (Koehn & Knowles, 2017), the exact search
optimum being degenerate (Stahlberg & Byrne, 2019). A `k=1` beam returns exactly
the greedy output, which confirms the search itself is implemented correctly.

**Why does Wiki-UQA drop?** BLEU falls from 5.58 to 3.57 and perplexity rises
from 34 to 51. Wiki-UQA is out of domain *and* written by humans, whereas the
training questions are **translated** from English SQuAD. Translated text has a
narrower, more regular phrasing distribution; the model fits that distribution
and is then surprised by natural Urdu question style. It is a concrete reminder
that a model trained on translated data learns the translation's quirks as much
as the language.

## Wrap-up

A 35M-parameter RNN, trained from scratch for 81 minutes, learns the *shape* of
Urdu question generation — the right question word, attention that lands on the
answer — but not yet the *content*. The honest bottleneck is model capacity and
data quality, not the architecture. Code, notebook and trained weights:
**https://github.com/Hanzala-12/urdu-question-generation**
