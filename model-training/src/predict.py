"""Single-image inference CLI.

Usage:
    python src/predict.py path/to/image.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision import transforms
from PIL import Image

from config import DEVICE, CHECKPOINT_DIR, IMG_SIZE, CLASS_NAMES
from src.model import build_model


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path: str | Path, device: str = DEVICE) -> dict:
    model = build_model(device)
    ckpt  = CHECKPOINT_DIR / "best_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    img    = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    pred_idx = pred_idx.item()
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(conf.item(), 4),
        "class_probs": {
            CLASS_NAMES[0]: round(probs[0, 0].item(), 4),
            CLASS_NAMES[1]: round(probs[0, 1].item(), 4),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print(f"Prediction:  {result['prediction']}")
    print(f"Confidence:  {result['confidence']*100:.1f}%")
    print(f"Class probs: {result['class_probs']}")
