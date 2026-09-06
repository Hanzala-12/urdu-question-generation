"""Task 3 - training loop.

    python -m src.train                     # full run (uses src/config.py)
    python -m src.train --debug             # 10k-pair subset, 1 epoch (debug gate)
    python -m src.train --epochs 15 --batch-size 64

The debug gate comes straight from the manual: if the loss does not fall
after one epoch on a 10k-pair subset, the bug is in the code - fix it
before spending an hour on the full run.

Checkpoints
-----------
* ``artifacts/best.pt`` - model weights only, lowest validation loss.
  This is what evaluation and the front end load.
* ``artifacts/last.pt`` - weights + optimizer + epoch, for resuming.

Per-epoch metrics are appended to ``results/loss_log.csv`` and plotted to
``results/figures/loss_curve.png``.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import (
    BEST_CKPT,
    CFG,
    FIGURES_DIR,
    LAST_CKPT,
    PAD_ID,
    RESULTS_DIR,
    TRAIN_TSV,
    VALID_TSV,
)
from src.dataset import QGDataset, collate_batch
from src.model import build_model, count_parameters
from src.utils import load_sp, pick_device, save_checkpoint, set_seed

LOSS_LOG = RESULTS_DIR / "loss_log.csv"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    teacher_forcing: float = 0.5,
    grad_clip: float = 1.0,
) -> float:
    """One pass over ``loader``.  Returns token-weighted mean loss."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lengths = batch["src_lengths"]
        pad_mask = batch["src_key_padding_mask"].to(device)
        gold = tgt[:, 1:]                                    # [B, T-1]
        n_tokens = gold.ne(PAD_ID).sum().item()

        with torch.set_grad_enabled(training):
            logits, _ = model(
                src, src_lengths, pad_mask, tgt,
                teacher_forcing=teacher_forcing if training else 1.0,
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), gold.reshape(-1)
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def train(
    *,
    epochs: int = CFG.train.epochs,
    batch_size: int = CFG.train.batch_size,
    limit: int | None = None,
    device_str: str | None = None,
    resume: bool = False,
) -> dict:
    set_seed(CFG.train.seed)
    device = pick_device(device_str)

    sp = load_sp()
    train_ds: object = QGDataset(TRAIN_TSV, sp)
    valid_ds: object = QGDataset(VALID_TSV, sp)
    if limit:
        train_ds = Subset(train_ds, range(min(limit, len(train_ds))))
        valid_ds = Subset(valid_ds, range(min(limit // 5 or 1, len(valid_ds))))
    print(f"device: {device} | epochs: {epochs} | batch: {batch_size} | "
          f"train pairs: {len(train_ds):,} | valid pairs: {len(valid_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=CFG.train.num_workers, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
    )

    model = build_model(CFG).to(device)
    n_params = count_parameters(model)
    print(f"trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.train.lr, weight_decay=CFG.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=CFG.train.lr_factor, patience=CFG.train.lr_patience
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID, label_smoothing=CFG.train.label_smoothing
    )

    start_epoch = 1
    if resume and LAST_CKPT.exists():
        ck = torch.load(LAST_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"resumed from epoch {start_epoch - 1}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh log for a normal run; append only when resuming, so the debug
    # gate's single epoch does not pollute the real training curve.
    log_mode = "a" if (resume and LOSS_LOG.exists()) else "w"
    log_f = LOSS_LOG.open(log_mode, newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if log_mode == "w":
        writer.writerow(["epoch", "train_loss", "val_loss", "val_ppl", "lr", "seconds"])

    best_val = math.inf
    history: list[dict] = []
    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        tr = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, teacher_forcing=CFG.train.teacher_forcing,
            grad_clip=CFG.train.grad_clip,
        )
        va = run_epoch(model, valid_loader, criterion, device)
        secs = time.time() - t0
        ppl = math.exp(min(va, 20))
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va)
        print(
            f"epoch {epoch:2d} | train {tr:.4f} | val {va:.4f} | "
            f"ppl {ppl:7.2f} | lr {lr:.2e} | {secs:.0f}s"
        )
        writer.writerow([epoch, f"{tr:.5f}", f"{va:.5f}", f"{ppl:.3f}", f"{lr:.2e}",
                         f"{secs:.1f}"])
        log_f.flush()
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "ppl": ppl})

        save_checkpoint(LAST_CKPT, model, CFG.model, optimizer=optimizer, epoch=epoch)
        if va < best_val:
            best_val = va
            save_checkpoint(BEST_CKPT, model, CFG.model,
                            extra={"val_loss": va, "epoch": epoch, "n_params": n_params})
            print(f"           saved new best -> {BEST_CKPT.name}")

    log_f.close()
    _plot_curve()
    print(f"\nbest val loss: {best_val:.4f}  (ppl {math.exp(min(best_val, 20)):.2f})")
    return {"best_val_loss": best_val, "history": history, "n_params": n_params}


def _plot_curve() -> None:
    if not LOSS_LOG.exists():
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, tr, va = [], [], []
    with LOSS_LOG.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"]))
            tr.append(float(r["train_loss"]))
            va.append(float(r["val_loss"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, tr, "o-", label="train")
    ax.plot(epochs, va, "s-", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'loss_curve.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only this many training pairs")
    ap.add_argument("--debug", action="store_true",
                    help="shortcut for --limit 10000 --epochs 1 (the debug gate)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    epochs = 1 if args.debug else args.epochs
    limit = 10_000 if args.debug else args.limit
    train(epochs=epochs, batch_size=args.batch_size, limit=limit,
          device_str=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
