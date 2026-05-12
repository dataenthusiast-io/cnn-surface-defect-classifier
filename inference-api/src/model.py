"""Inference-only model builder (no optimizer / training utilities)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(device: str) -> nn.Module:
    import torch
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 2),
    )
    return model.to(device)
