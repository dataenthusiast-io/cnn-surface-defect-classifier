"""Download the NEU Surface Defect dataset and stage it for training.

Usage:
    .venv/bin/python3 scripts/download_data.py

The Kaggle package ships with train/ and validation/ splits. We merge both
into data/raw/<class>/ so our own stratified split owns the partitioning.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import kagglehub
from config import DATA_DIR

# ── 1. Download ──────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print("Path to dataset files:", path)

# ── 2. Merge train + validation into data/raw/<class>/ ───────────────────────
kaggle_root  = Path(path)
image_dirs   = list(kaggle_root.rglob("images"))

if not image_dirs:
    print("ERROR: no 'images' folder found inside downloaded dataset.")
    sys.exit(1)

DATA_DIR.mkdir(parents=True, exist_ok=True)
total = 0

for img_dir in sorted(image_dirs):
    split = img_dir.parent.name          # "train" or "validation"
    for cls_dir in sorted(img_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        dest = DATA_DIR / cls_dir.name
        dest.mkdir(exist_ok=True)
        for f in cls_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}:
                target = dest / f"{split}_{f.name}"
                if not target.exists():
                    shutil.copy2(f, target)
                    total += 1

# ── 3. Summary ───────────────────────────────────────────────────────────────
print(f"\nDataset staged at: {DATA_DIR}  ({total} new files)")
for cls_dir in sorted(DATA_DIR.iterdir()):
    if cls_dir.is_dir():
        n = sum(1 for f in cls_dir.iterdir() if f.is_file())
        print(f"  {cls_dir.name:<20} {n} images")
