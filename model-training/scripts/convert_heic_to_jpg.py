"""Convert HEIC images from data/data-new/ to JPEG in data/raw/.

Usage:
    python scripts/convert_heic_to_jpg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("ERROR: pillow-heif not installed.  Run:  pip install pillow-heif")
    sys.exit(1)

from PIL import Image

ROOT        = Path(__file__).parent.parent.parent   # cnn-surface-defect-classifier/
SRC_DIR     = ROOT / "data" / "data-new"
DST_DIR     = ROOT / "data" / "raw"
JPEG_QUALITY = 95


def convert_folder(src_folder: Path, dst_folder: Path) -> int:
    dst_folder.mkdir(parents=True, exist_ok=True)
    converted = 0
    heic_files = [f for f in sorted(src_folder.iterdir())
                  if f.suffix.upper() == ".HEIC"]

    for heic in heic_files:
        dst = dst_folder / (heic.stem + ".jpg")
        if dst.exists():
            converted += 1
            continue
        try:
            img = Image.open(heic).convert("RGB")
            img.save(dst, "JPEG", quality=JPEG_QUALITY)
            converted += 1
        except Exception as exc:
            print(f"  WARN: could not convert {heic.name}: {exc}")

    return converted


def main() -> None:
    if not SRC_DIR.exists():
        print(f"ERROR: source folder not found: {SRC_DIR}")
        sys.exit(1)

    class_folders = [f for f in sorted(SRC_DIR.iterdir()) if f.is_dir()]
    if not class_folders:
        print(f"ERROR: no sub-folders found in {SRC_DIR}")
        sys.exit(1)

    print(f"\nSource : {SRC_DIR}")
    print(f"Target : {DST_DIR}")
    print(f"Classes: {[f.name for f in class_folders]}\n")

    total = 0
    for folder in class_folders:
        n = convert_folder(folder, DST_DIR / folder.name)
        total += n
        print(f"  {folder.name:<20}  {n:>4} images")

    print(f"\nDone — {total} images in {DST_DIR}")


if __name__ == "__main__":
    main()
