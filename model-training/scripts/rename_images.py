"""Rename JPEGs to <Fehlermerkmal>_N.jpg (sequential).

Handles partially-renamed folders (e.g. Abdruck1_10.jpg from a prior incomplete run).
Uses a temp-name intermediate pass to avoid filename collisions on Windows.

Result:
  data/raw/<class>/Abdruck1_1.jpg  ...  Abdruck1_163.jpg
  data/raw/<class>/Abdruck2_1.jpg  ...  Abdruck2_963.jpg
  data/raw/<class>/iOTeil_1.jpg    ...  iOTeil_501.jpg
  data/raw/<class>/Stanzfehler_1.jpg ... Stanzfehler_156.jpg

Usage:
    python scripts/rename_images.py
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

ROOT    = Path(__file__).parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"

PREFIX_MAP: dict[str, str] = {
    "Abdruck 1":   "Abdruck1",
    "Abdruck 2":   "Abdruck2",
    "i.O.-Teile":  "iOTeil",
    "Stanzfehler": "Stanzfehler",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
TMP_TAG  = f"_TMPRENAME_{uuid.uuid4().hex[:8]}_"


def natural_key(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else 0


def process_class(class_folder: Path, prefix: str) -> int:
    jpg_files = [f for f in class_folder.iterdir() if f.suffix.lower() in IMG_EXTS]
    if not jpg_files:
        print(f"  WARN: no images in {class_folder.name}")
        return 0

    # Sort by last number in filename (handles both Teil (N) and Prefix_N)
    jpg_files.sort(key=natural_key)

    # Pass 1: rename everything to unique temp names to avoid collisions
    tmp_paths: list[Path] = []
    for src in jpg_files:
        tmp = class_folder / f"{TMP_TAG}{src.name}"
        src.rename(tmp)
        tmp_paths.append(tmp)

    # Pass 2: rename temp -> final sequential names
    for n, tmp in enumerate(tmp_paths, start=1):
        tmp.rename(class_folder / f"{prefix}_{n}.jpg")

    return len(tmp_paths)


def main() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run convert_heic_to_jpg.py first.")
        sys.exit(1)

    print(f"Images : {RAW_DIR}")
    print()

    total = 0
    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir():
            continue
        prefix = PREFIX_MAP.get(folder.name)
        if prefix is None:
            print(f"  SKIP (no prefix): {folder.name}")
            continue
        n = process_class(folder, prefix)
        total += n
        print(f"  {folder.name:<20}  {n:>4} files  ->  {prefix}_1.jpg .. {prefix}_{n}.jpg")

    print(f"\nDone -- {total} images renamed.")


if __name__ == "__main__":
    main()
