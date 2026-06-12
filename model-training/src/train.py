"""Training script with tqdm logging and mid-epoch checkpoint recovery.

Usage:
    python src/train.py              # full training, auto-resumes if resume.pth exists
    python src/train.py --smoke-test  # 1 epoch per phase (quick sanity check)
    python src/train.py --restart     # ignore any existing resume.pth, start fresh
"""
from __future__ import annotations

import sys
import argparse
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from tqdm import tqdm

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from config import (
    DEVICE, CHECKPOINT_DIR, LOG_PATH,
    PHASE1_EPOCHS, PHASE2_EPOCHS, PATIENCE, NUM_CLASSES,
)
from src.dataset import get_dataloaders
from src.model import (
    build_model, freeze_for_phase1, unfreeze_for_phase2,
    get_optimizer_phase1, get_optimizer_phase2, count_trainable_params,
)

RESUME_PATH = CHECKPOINT_DIR / "resume.pth"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    phase: int,
    epoch: int,
    global_epoch: int,
    best_val_f1: float,
    patience_counter: int,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "phase":            phase,
        "epoch":            epoch,
        "global_epoch":     global_epoch,
        "best_val_f1":      best_val_f1,
        "patience_counter": patience_counter,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
    }, RESUME_PATH)


def load_resume(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Epoch runner with tqdm batch progress
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    training: bool,
    desc: str = "",
) -> tuple[float, float, list[int], list[int]]:
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_seen = 0

    bar = tqdm(loader, desc=desc, leave=False, ncols=90,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  loss={postfix}")

    with torch.set_grad_enabled(training):
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * len(images)
            n_seen     += len(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            running_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
            bar.set_postfix_str(f"loss={batch_loss:.3f}  acc={running_acc*100:.1f}%")

    avg_loss = total_loss / len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return avg_loss, acc, all_preds, all_labels


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

LOG_FIELDS = [
    "epoch", "phase", "train_loss", "val_loss",
    "train_acc", "val_acc", "val_f1", "lr", "duration_s",
]


def open_log(append: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    f = open(LOG_PATH, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
    if not append:
        writer.writeheader()
    return f, writer


def log_epoch(writer, csv_file, row: dict) -> None:
    writer.writerow(row)
    csv_file.flush()


def print_epoch_summary(
    global_epoch: int, total_epochs: int, phase: int,
    t_loss: float, v_loss: float,
    t_acc: float, v_acc: float,
    v_f1: float, lr: float, dur: float,
    is_best: bool,
) -> None:
    best_marker = " ★ best" if is_best else ""
    print(
        f"  Epoch {global_epoch:02d}/{total_epochs:02d} │ Ph{phase} │ "
        f"loss {t_loss:.3f}→{v_loss:.3f} │ "
        f"acc {t_acc*100:.1f}→{v_acc*100:.1f}% │ "
        f"F1 {v_f1:.3f} │ "
        f"lr {lr:.1e} │ "
        f"{dur:.1f}s"
        f"{best_marker}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(smoke_test: bool = False, restart: bool = False) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    phase1_epochs = 1 if smoke_test else PHASE1_EPOCHS
    phase2_epochs = 1 if smoke_test else PHASE2_EPOCHS
    total_epochs  = phase1_epochs + phase2_epochs

    print(f"\n{'═'*60}")
    print(f"  CNN Defect Detector – Training")
    print(f"  Device  : {device}")
    print(f"  Phase 1 : {phase1_epochs} epoch(s)  │  Phase 2 : {phase2_epochs} epoch(s)")
    print(f"{'═'*60}\n")

    train_loader, val_loader, _ = get_dataloaders()

    # Weighted loss to compensate for class imbalance in the training set.
    train_labels = [
        train_loader.dataset.dataset.samples[i][1]
        for i in train_loader.dataset.indices
    ]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_labels,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    model = build_model(device)

    # ── Determine start state ────────────────────────────────────────────────
    resume_phase        = 1
    resume_epoch_start  = 1
    global_epoch        = 0
    best_val_f1         = 0.0
    patience_counter    = 0
    appending_log       = False

    if not restart and RESUME_PATH.exists():
        ckpt = load_resume(RESUME_PATH)
        model.load_state_dict(ckpt["model_state"])
        resume_phase     = ckpt["phase"]
        resume_epoch_start = ckpt["epoch"] + 1       # next epoch to run
        global_epoch     = ckpt["global_epoch"]
        best_val_f1      = ckpt["best_val_f1"]
        patience_counter = ckpt["patience_counter"]
        appending_log    = True
        print(f"  ↻  Resuming from Phase {resume_phase}, epoch {resume_epoch_start} "
              f"(best F1 so far: {best_val_f1:.4f})\n")
    else:
        if RESUME_PATH.exists():
            RESUME_PATH.unlink()
        print("  Starting fresh.\n")

    csv_file, writer = open_log(append=appending_log)

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if resume_phase == 1:
        print(f"{'─'*60}")
        print(f"  Phase 1 – Head-only  ({phase1_epochs} epoch(s))")
        freeze_for_phase1(model)
        print(f"  Trainable params: {count_trainable_params(model):,}\n")

        optimizer1 = get_optimizer_phase1(model)
        # dummy scheduler for uniform save_resume signature
        dummy_sched = torch.optim.lr_scheduler.LambdaLR(optimizer1, lambda e: 1.0)

        # Restore optimizer state if we're mid-phase-1
        if resume_epoch_start > 1 and ckpt.get("phase") == 1:
            optimizer1.load_state_dict(ckpt["optimizer_state"])

        for ep in range(resume_epoch_start, phase1_epochs + 1):
            global_epoch += 1
            t0 = time.time()

            t_loss, t_acc, _, _ = run_epoch(
                model, train_loader, criterion, optimizer1, device,
                training=True,
                desc=f"Ph1 ep{ep:02d} train",
            )
            v_loss, v_acc, v_preds, v_labels = run_epoch(
                model, val_loader, criterion, None, device,
                training=False,
                desc=f"Ph1 ep{ep:02d} val  ",
            )
            v_f1 = f1_score(v_labels, v_preds, average="macro", zero_division=0)
            dur  = time.time() - t0
            lr   = optimizer1.param_groups[0]["lr"]

            is_best = v_f1 > best_val_f1
            if is_best:
                best_val_f1 = v_f1
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")

            print_epoch_summary(global_epoch, total_epochs, 1,
                                t_loss, v_loss, t_acc, v_acc, v_f1, lr, dur, is_best)
            log_epoch(writer, csv_file, {
                "epoch": global_epoch, "phase": 1,
                "train_loss": round(t_loss, 4), "val_loss": round(v_loss, 4),
                "train_acc": round(t_acc, 4), "val_acc": round(v_acc, 4),
                "val_f1": round(v_f1, 4), "lr": lr,
                "duration_s": round(dur, 1),
            })
            save_resume(model, optimizer1, dummy_sched, 1, ep, global_epoch,
                        best_val_f1, patience_counter)

        resume_epoch_start = 1   # reset for phase 2

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase 2 – Fine-tune layer3+layer4+fc  ({phase2_epochs} epoch(s))")
    unfreeze_for_phase2(model)
    print(f"  Trainable params: {count_trainable_params(model):,}\n")

    optimizer2 = get_optimizer_phase2(model)
    scheduler  = CosineAnnealingLR(optimizer2, T_max=phase2_epochs)

    if resume_phase == 2:
        optimizer2.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        resume_epoch_start = ckpt["epoch"] + 1
        patience_counter   = ckpt["patience_counter"]

    for ep in range(resume_epoch_start, phase2_epochs + 1):
        global_epoch += 1
        t0 = time.time()

        t_loss, t_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer2, device,
            training=True,
            desc=f"Ph2 ep{ep:02d} train",
        )
        v_loss, v_acc, v_preds, v_labels = run_epoch(
            model, val_loader, criterion, None, device,
            training=False,
            desc=f"Ph2 ep{ep:02d} val  ",
        )
        v_f1 = f1_score(v_labels, v_preds, average="macro", zero_division=0)
        scheduler.step()
        dur  = time.time() - t0
        lr   = optimizer2.param_groups[-1]["lr"]

        is_best = v_f1 > best_val_f1
        if is_best:
            best_val_f1      = v_f1
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
        else:
            patience_counter += 1

        print_epoch_summary(global_epoch, total_epochs, 2,
                            t_loss, v_loss, t_acc, v_acc, v_f1, lr, dur, is_best)
        log_epoch(writer, csv_file, {
            "epoch": global_epoch, "phase": 2,
            "train_loss": round(t_loss, 4), "val_loss": round(v_loss, 4),
            "train_acc": round(t_acc, 4), "val_acc": round(v_acc, 4),
            "val_f1": round(v_f1, 4), "lr": lr,
            "duration_s": round(dur, 1),
        })
        save_resume(model, optimizer2, scheduler, 2, ep, global_epoch,
                    best_val_f1, patience_counter)

        if patience_counter >= PATIENCE and not smoke_test:
            print(f"\n  Early stop: no F1 improvement for {PATIENCE} epochs.")
            break

    # ── Finalise ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), CHECKPOINT_DIR / "last_model.pth")
    RESUME_PATH.unlink(missing_ok=True)   # clean up – training finished cleanly
    csv_file.close()

    print(f"\n{'═'*60}")
    print(f"  Training complete.  Best Val F1: {best_val_f1:.4f}")
    print(f"  Checkpoints → {CHECKPOINT_DIR}")
    print(f"  Log         → {LOG_PATH}")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 epoch per phase (quick sanity check)")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore resume.pth and start from scratch")
    args = parser.parse_args()
    train(smoke_test=args.smoke_test, restart=args.restart)
