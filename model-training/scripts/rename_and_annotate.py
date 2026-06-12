"""Rename JPEGs to <Fehlermerkmal>_N.jpg (sequential) and create Pascal VOC XML annotations.

Handles partially-renamed folders (e.g. Abdruck1_10.jpg from a prior incomplete run).
Uses a temp-name intermediate pass to avoid filename collisions on Windows.

Result:
  data/raw/<class>/Abdruck1_1.jpg  ...  Abdruck1_163.jpg
  data/raw/<class>/Abdruck2_1.jpg  ...  Abdruck2_963.jpg
  data/raw/<class>/iOTeil_1.jpg    ...  iOTeil_501.jpg
  data/raw/<class>/Stanzfehler_1.jpg ... Stanzfehler_156.jpg
  data/annotations/<prefix>_N.xml  (flat folder, one per image)

Usage:
    python scripts/rename_and_annotate.py
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed.")
    sys.exit(1)

ROOT      = Path(__file__).parent.parent.parent
RAW_DIR   = ROOT / "data" / "raw"
ANNOT_DIR = ROOT / "data" / "annotations"
DB_NAME   = "Produktionsdaten"

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


def make_xml(folder_name: str, filename: str, w: int, h: int) -> str:
    return (
        "<annotation>\n"
        f"  <folder>{folder_name}</folder>\n"
        f"  <filename>{filename}</filename>\n"
        "  <source>\n"
        f"    <database>{DB_NAME}</database>\n"
        "  </source>\n"
        "  <size>\n"
        f"    <width>{w}</width>\n"
        f"    <height>{h}</height>\n"
        "    <depth>3</depth>\n"
        "  </size>\n"
        "  <segmented>0</segmented>\n"
        "  <object>\n"
        f"    <name>{folder_name}</name>\n"
        "    <pose>Unspecified</pose>\n"
        "    <truncated>0</truncated>\n"
        "    <difficult>0</difficult>\n"
        "    <bndbox>\n"
        "      <xmin>0</xmin>\n"
        "      <ymin>0</ymin>\n"
        f"      <xmax>{w}</xmax>\n"
        f"      <ymax>{h}</ymax>\n"
        "    </bndbox>\n"
        "  </object>\n"
        "</annotation>\n"
    )


def process_class(class_folder: Path, prefix: str) -> int:
    jpg_files = [f for f in class_folder.iterdir() if f.suffix.lower() in IMG_EXTS]
    if not jpg_files:
        print(f"  WARN: no images in {class_folder.name}")
        return 0

    # Sort by last number in filename (handles both Teil (N) and Prefix_N)
    jpg_files.sort(key=natural_key)

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    # Pass 1: rename everything to unique temp names to avoid collisions
    tmp_paths: list[Path] = []
    for src in jpg_files:
        tmp = class_folder / f"{TMP_TAG}{src.name}"
        src.rename(tmp)
        tmp_paths.append(tmp)

    # Pass 2: rename temp -> final sequential names + write XML
    for n, tmp in enumerate(tmp_paths, start=1):
        final_name = f"{prefix}_{n}.jpg"
        final_path = class_folder / final_name
        tmp.rename(final_path)

        with Image.open(final_path) as img:
            w, h = img.size

        xml_path = ANNOT_DIR / f"{prefix}_{n}.xml"
        xml_path.write_text(
            make_xml(class_folder.name, final_name, w, h),
            encoding="utf-8",
        )

    return len(tmp_paths)


def main() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run convert_heic_to_jpg.py first.")
        sys.exit(1)

    print(f"Images      : {RAW_DIR}")
    print(f"Annotations : {ANNOT_DIR}")
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

    print(f"\nDone -- {total} images, {total} XML annotations.")


if __name__ == "__main__":
    main()
