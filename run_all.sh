#!/usr/bin/env bash
# One-shot pipeline for the Kaggle GPU notebook.
#   !git clone --depth 1 https://github.com/Hanzala-12/urdu-question-generation.git
#   !bash urdu-question-generation/run_all.sh
#
# set -e: stop at the first failing step so a data bug never wastes the
# 1-2 h training slot.
set -euo pipefail

cd "$(dirname "$0")"
echo "== repo: $(pwd)  |  $(git rev-parse --short HEAD) =="

python -c "import torch; print('torch', torch.__version__, '| GPU:',
 torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - enable GPU!')"

pip -q install sentencepiece sacrebleu rouge-score

echo "== Task 1: data preparation =="
python -m src.data_prep

echo "== Task 2: SentencePiece tokenizer =="
python -m src.spm_train

echo "== Debug gate: 10k pairs, 1 epoch (loss must fall) =="
python -m src.train --debug

echo "== Task 3: full training =="
python -m src.train --epochs 15 --batch-size 64

echo "== Task 4: evaluation (UQA-valid + Wiki-UQA, greedy + beam) =="
python -m src.evaluate --split both --beam-max 3000

echo "== Bundle outputs for download =="
python - <<'PY'
import shutil, pathlib, json
out = pathlib.Path("/kaggle/working/outputs")
for sub in ("artifacts", "results"):
    shutil.copytree(sub, out / sub, dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/outputs", "zip", out)
print("wrote /kaggle/working/outputs.zip")
print(json.dumps(json.load(open("results/metrics.json")), indent=2, ensure_ascii=False))
PY

echo "== DONE =="
