from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import random as _random

import torch
from torchvision import transforms
from PIL import Image

from config import DEVICE, CHECKPOINT_DIR, IMG_SIZE, CLASS_NAMES
from src.dataset import get_test_dataset
from src.model import build_model


_inv_norm = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229,       1/0.224,       1/0.225],
)


def _tensor_to_b64(tensor: torch.Tensor) -> str:
    img = _inv_norm(tensor).permute(1, 2, 0).clamp(0, 1)
    img_np = (img.numpy() * 255).astype("uint8")
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class InferencePipeline:
    """Loads best_model.pth once. Iterates through the test-set on each next() call."""

    def __init__(self) -> None:
        self.device = DEVICE
        self.model  = build_model(self.device)
        ckpt = CHECKPOINT_DIR / "best_model.pth"
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        self.dataset  = get_test_dataset()
        # Shuffle indices so consecutive samples come from different classes
        self._order: list[int] = list(range(len(self.dataset)))
        _random.shuffle(self._order)
        self._pos = 0

        self._total_inspected = 0
        self._correct         = 0
        self._errors          = 0
        self._conf_sum        = 0.0
        self._class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}

        print(f"InferencePipeline loaded. Test-set size: {len(self.dataset)}")

    @torch.no_grad()
    def next(self) -> dict[str, Any]:
        # Re-shuffle when we've exhausted the shuffled order
        if self._pos >= len(self._order):
            _random.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1

        img_tensor, true_label_int = self.dataset[idx]
        logits = self.model(img_tensor.unsqueeze(0).to(self.device))
        probs  = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

        pred_int  = pred_idx.item()
        conf_val  = conf.item()
        correct   = pred_int == true_label_int

        self._total_inspected += 1
        self._conf_sum        += conf_val
        if correct:
            self._correct += 1
        else:
            self._errors += 1
        self._class_counts[CLASS_NAMES[pred_int]] += 1

        return {
            "index":       self._pos,
            "total":       len(self.dataset),
            "image_b64":   _tensor_to_b64(img_tensor),
            "true_label":  CLASS_NAMES[true_label_int],
            "prediction":  CLASS_NAMES[pred_int],
            "confidence":  round(conf_val, 4),
            "correct":     bool(correct),
            "class_probs": {
                CLASS_NAMES[i]: round(probs[0, i].item(), 4)
                for i in range(len(CLASS_NAMES))
            },
        }

    def reset(self) -> None:
        _random.shuffle(self._order)
        self._pos             = 0
        self._total_inspected = 0
        self._correct         = 0
        self._errors          = 0
        self._conf_sum        = 0.0
        self._class_counts    = {c: 0 for c in CLASS_NAMES}

    def get_stats(self) -> dict[str, Any]:
        n = self._total_inspected or 1
        return {
            "total_inspected":  self._total_inspected,
            "accuracy":         round(self._correct / n, 4),
            "errors":           self._errors,
            "avg_confidence":   round(self._conf_sum / n, 4),
            "class_counts":     dict(self._class_counts),
        }
