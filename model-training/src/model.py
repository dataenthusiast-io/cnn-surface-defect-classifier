from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

from config import (
    PHASE1_LR_HEAD,
    PHASE2_LR_HEAD,
    PHASE2_LR_BACKBONE,
)


def build_model(device: str) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 2),
    )
    return model.to(device)


def freeze_for_phase1(model: nn.Module) -> None:
    """Freeze entire backbone; only the head trains."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_for_phase2(model: nn.Module) -> None:
    """Unfreeze layer3, layer4, and fc for fine-tuning."""
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ["layer3", "layer4", "fc"]):
            param.requires_grad = True


def get_optimizer_phase1(model: nn.Module) -> optim.Optimizer:
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return optim.Adam(trainable, lr=PHASE1_LR_HEAD)


def get_optimizer_phase2(model: nn.Module) -> optim.Optimizer:
    return optim.Adam([
        {"params": model.layer3.parameters(), "lr": PHASE2_LR_BACKBONE},
        {"params": model.layer4.parameters(), "lr": PHASE2_LR_BACKBONE},
        {"params": model.fc.parameters(),     "lr": PHASE2_LR_HEAD},
    ])


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
