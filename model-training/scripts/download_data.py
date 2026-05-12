"""Download the NEU Surface Defect dataset via kagglehub and symlink/copy to data/raw."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import kagglehub

from config import DATA_DIR

DATASET_SLUG = "kaustubhdikshit/neu-surface-defect-database"


def main() -> None:
    print(f"Downloading dataset: {DATASET_SLUG} ...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    kaggle_root = Path(path)
    print(f"Downloaded to: {kaggle_root}")

    # NEU-DET has train/images/<class> and validation/images/<class>.
    # Merge both splits into data/raw/<class> so we own the split logic.
    image_roots = list(kaggle_root.rglob("images"))
    if not image_roots:
        print("ERROR: Could not locate 'images' folders in downloaded dataset.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    copied_total = 0
    for img_root in sorted(image_roots):
        split_name = img_root.parent.name  # "train" or "validation"
        for cls_dir in sorted(img_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            dest = DATA_DIR / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in cls_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".bmp", ".png"}:
                    target = dest / f"{split_name}_{f.name}"
                    if not target.exists():
                        shutil.copy2(f, target)
                        n += 1
            copied_total += n
            print(f"  [{split_name}] {cls_dir.name}: +{n} images → {dest}")

    print(f"\nDataset ready at: {DATA_DIR}")
    print("Classes:")
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if cls_dir.is_dir():
            n = len([f for f in cls_dir.iterdir() if f.is_file()])
            print(f"  {cls_dir.name}: {n} images")


if __name__ == "__main__":
    main()
