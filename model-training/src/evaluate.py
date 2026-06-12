"""Evaluate best_model.pth on the test set and generate all plots."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
from torch.utils.data import DataLoader

from config import DEVICE, CHECKPOINT_DIR, PLOT_DIR, CLASS_NAMES, BATCH_SIZE, NUM_WORKERS, LOG_PATH
from src.dataset import get_dataloaders, get_test_dataset
from src.model import build_model


def load_best_model(device: str) -> torch.nn.Module:
    model = build_model(device)
    ckpt = CHECKPOINT_DIR / "best_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[list[int], list[int], list[float]]:
    all_labels, all_preds, all_confs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())
    return all_labels, all_preds, all_confs


def print_metrics(labels: list[int], preds: list[int], confs: list[float]) -> None:
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    errors = sum(1 for l, p in zip(labels, preds) if l != p)

    print("\nTest-Ergebnisse")
    print("─" * 50)
    print(f"Accuracy:        {acc*100:.1f}%")
    print(f"Macro F1:        {f1:.3f}")
    print(f"Fehlklassifikationen: {errors} / {len(labels)}")
    print("─" * 50)
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_training_curves(log_path: Path, plot_dir: Path) -> None:
    import csv
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    epochs    = [r["epoch"] for r in rows]
    t_loss    = [r["train_loss"] for r in rows]
    v_loss    = [r["val_loss"] for r in rows]
    t_acc     = [r["train_acc"] * 100 for r in rows]
    v_acc     = [r["val_acc"] * 100 for r in rows]
    v_f1      = [r["val_f1"] for r in rows]
    lr        = [r["lr"] for r in rows]
    phases    = [int(r["phase"]) for r in rows]

    phase2_start = next((e for e, p in zip(epochs, phases) if p == 2), None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Curves – 4-class Defect Classifier", fontsize=14)

    def vline(ax):
        if phase2_start:
            ax.axvline(phase2_start - 0.5, color="gray", linestyle="--",
                       linewidth=1, label="Phase 1→2")

    ax = axes[0, 0]
    ax.plot(epochs, t_loss, label="Train Loss")
    ax.plot(epochs, v_loss, label="Val Loss")
    vline(ax)
    ax.set_title("Loss")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, t_acc, label="Train Acc %")
    ax.plot(epochs, v_acc, label="Val Acc %")
    vline(ax)
    ax.set_title("Accuracy (%)")
    ax.legend()

    ax = axes[1, 0]
    best_f1 = max(v_f1)
    best_ep = epochs[v_f1.index(best_f1)]
    ax.plot(epochs, v_f1, label="Val Macro F1")
    ax.scatter([best_ep], [best_f1], color="red", zorder=5, label=f"Best F1={best_f1:.3f}")
    vline(ax)
    ax.set_title("Val Macro F1")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, lr)
    vline(ax)
    ax.set_title("Learning Rate")
    ax.set_yscale("log")

    plt.tight_layout()
    out = plot_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_confusion_matrix(labels: list[int], preds: list[int], plot_dir: Path) -> None:
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2%",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion Matrix (normalised) — 4-class")
    ax.set_ylabel("True Class")
    ax.set_xlabel("Predicted Class")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = plot_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_sample_predictions(
    model: torch.nn.Module,
    test_subset,
    device: str,
    plot_dir: Path,
    n: int = 16,
) -> None:
    from torchvision import transforms as T
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229,       1/0.224,       1/0.225],
    )

    indices = list(range(min(n, len(test_subset))))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Sample Predictions (4×4 Grid)", fontsize=12)

    for ax, idx in zip(axes.flat, indices):
        img_tensor, true_label = test_subset[idx]
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        pred      = pred.item()
        conf      = conf.item()
        correct   = pred == true_label
        color     = "green" if correct else "red"

        img = inv_norm(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        ax.set_title(
            f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred]}  {conf*100:.0f}%",
            fontsize=7,
        )

    plt.tight_layout()
    out = plot_dir / "sample_predictions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_misclassifications(
    model: torch.nn.Module,
    test_subset,
    device: str,
    plot_dir: Path,
) -> None:
    from torchvision import transforms as T
    inv_norm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229,       1/0.224,       1/0.225],
    )

    errors = []
    for idx in range(len(test_subset)):
        img_tensor, true_label = test_subset[idx]
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        pred = pred.item()
        conf = conf.item()
        if pred != true_label:
            img = inv_norm(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
            errors.append((img, true_label, pred, conf))

    if not errors:
        print("No misclassifications — perfect test set!")
        return

    cols = min(len(errors), 6)
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.0 * rows))
    axes_flat = np.array(axes).flatten() if rows > 1 or cols > 1 else [axes]
    fig.suptitle(f"Misclassifications (n={len(errors)})", fontsize=11)

    for ax, (img, true_l, pred_l, conf) in zip(axes_flat, errors):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"True: {CLASS_NAMES[true_l]}\nPred: {CLASS_NAMES[pred_l]}  {conf*100:.0f}%",
            fontsize=7, color="red",
        )
    for ax in axes_flat[len(errors):]:
        ax.axis("off")

    plt.tight_layout()
    out = plot_dir / "misclassifications.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_confidence_distribution(
    labels: list[int],
    preds: list[int],
    confs: list[float],
    plot_dir: Path,
) -> None:
    correct = [c for l, p, c in zip(labels, preds, confs) if l == p]
    wrong   = [c for l, p, c in zip(labels, preds, confs) if l != p]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct, bins=20, alpha=0.6, label=f"Korrekt (n={len(correct)})", color="green")
    ax.hist(wrong,   bins=20, alpha=0.6, label=f"Falsch (n={len(wrong)})",   color="red")
    ax.set_xlabel("Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution – Korrekt vs. Falsch")
    ax.legend()
    plt.tight_layout()
    out = plot_dir / "confidence_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------

def evaluate() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders()
    model = load_best_model(device)

    labels, preds, confs = predict_loader(model, test_loader, device)
    print_metrics(labels, preds, confs)

    if LOG_PATH.exists():
        plot_training_curves(LOG_PATH, PLOT_DIR)
    plot_confusion_matrix(labels, preds, PLOT_DIR)

    test_subset = get_test_dataset()
    plot_sample_predictions(model, test_subset, device, PLOT_DIR)
    plot_misclassifications(model, test_subset, device, PLOT_DIR)
    plot_confidence_distribution(labels, preds, confs, PLOT_DIR)

    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    evaluate()
