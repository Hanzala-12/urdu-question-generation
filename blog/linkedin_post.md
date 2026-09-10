# LinkedIn post

*(Attach `results/figures/frontend.png` and `results/figures/attention.png` as images. Replace `[BLOG LINK]` with the Medium URL after publishing.)*

---

For our Generative AI course we built a **sequence-to-sequence question generator
for Urdu — completely from scratch.** No pretrained weights, no Transformers, no
off-the-shelf seq2seq. Just an RNN encoder–decoder with attention, built from
`nn.Embedding` / `nn.LSTM` / `nn.Linear`, and an 8k SentencePiece vocabulary
learned from the data.

Given an Urdu sentence with the answer marked, the model writes the question:

`دریائے سندھ تقریباً <ans> 3180 کلومیٹر </ans> طویل ہے۔`  →  `دریائے سندھ کی لمبائی کتنی ہے؟`

A few things I learned:

• A 2-layer BiLSTM encoder + Bahdanau-attention decoder (35.5M params) trains in
~80 minutes on a single free T4 GPU. Validation loss bottoms out at epoch 10,
then overfits.

• **Beam search made it worse, not better** (BLEU 1.0 vs greedy's 5.6). On a
weak model, beam finds fluent-but-generic question templates and recycles them;
greedy stays anchored to the answer via attention. This "beam-search
degradation" is documented (Koehn & Knowles 2017) — a nice reminder that more
search ≠ better output.

• The model reliably gets the **question word** right and fails on **content** —
hallucinating names, repeating phrases, copying fragmented tokens. On fresh
sentences through the front end:
   – *"… جو <ans> نیپال </ans> میں واقع ہے۔"* → *"ماؤنٹ ایورسٹ کہاں واقع ہے؟"* ("Where is Everest located?") ✔
   – *"… <ans> قائد اعظم محمد علی جناح </ans> 1876 میں …"* → the long name fragments into gibberish

• Attention heat-maps show it genuinely learned to look at the answer span while
generating.

Full write-up: [BLOG LINK]
Code + notebook + trained weights: https://github.com/Hanzala-12/urdu-question-generation

#NLP #DeepLearning #MachineLearning #Urdu #Seq2Seq #GenerativeAI
