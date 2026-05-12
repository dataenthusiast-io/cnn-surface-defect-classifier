from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision import transforms
from PIL import Image

from config import DEVICE, CHECKPOINT_DIR, IMG_SIZE, CLASS_NAMES
from src.dataset import get_test_dataset
from src.model import build_model


_to_pil = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229,       1/0.224,       1/0.225],
    ),
])


def _tensor_to_b64(tensor: torch.Tensor) -> str:
    """Convert a normalised CHW tensor to a base64-encoded PNG string."""
    img = _to_pil(tensor).permute(1, 2, 0).clamp(0, 1)
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

        self.dataset = get_test_dataset()
        self._index  = 0

        self._total_inspected = 0
        self._correct         = 0
        self._defects         = 0
        self._false_negatives = 0
        self._false_positives = 0
        self._conf_sum        = 0.0

    # ------------------------------------------------------------------

    @torch.no_grad()
    def next(self) -> dict[str, Any]:
        idx = self._index % len(self.dataset)
        self._index += 1

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
        if pred_int == 1:
            self._defects += 1
        if true_label_int == 1 and pred_int == 0:
            self._false_negatives += 1
        if true_label_int == 0 and pred_int == 1:
            self._false_positives += 1

        return {
            "index":      self._index,
            "total":      len(self.dataset),
            "image_b64":  _tensor_to_b64(img_tensor),
            "true_label": CLASS_NAMES[true_label_int],
            "prediction": CLASS_NAMES[pred_int],
            "confidence": round(conf_val, 4),
            "correct":    bool(correct),
            "class_probs": {
                CLASS_NAMES[0]: round(probs[0, 0].item(), 4),
                CLASS_NAMES[1]: round(probs[0, 1].item(), 4),
            },
        }

    def reset(self) -> None:
        self._index           = 0
        self._total_inspected = 0
        self._correct         = 0
        self._defects         = 0
        self._false_negatives = 0
        self._false_positives = 0
        self._conf_sum        = 0.0

    def get_stats(self) -> dict[str, Any]:
        n = self._total_inspected or 1
        return {
            "total_inspected": self._total_inspected,
            "defect_rate":     round(self._defects / n, 4),
            "accuracy":        round(self._correct / n, 4),
            "avg_confidence":  round(self._conf_sum / n, 4),
            "false_negatives": self._false_negatives,
            "false_positives": self._false_positives,
        }
