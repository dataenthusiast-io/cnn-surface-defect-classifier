# Data Requirements

This document describes the exact data format expected by the pipeline.

---

## Source Format (what you bring in)

Raw images are captured with an iPhone (HEIC format) and organised into one folder per class before processing.

```
<source-root>/
├── Abdruck 1/
│   ├── IMG_0001.HEIC
│   └── …
├── Abdruck 2/
│   └── …
├── Stanzfehler/
│   └── …
└── i.O.-Teile/
    └── …
```

### Classes and sample counts

| Class | Description | Samples |
|---|---|---|
| `Abdruck 1` | Werkzeugabdruck Typ 1 (leicht) | ~163 |
| `Abdruck 2` | Werkzeugabdruck Typ 2 (schwer) | ~963 |
| `Stanzfehler` | Stanzprozessfehler | ~156 |
| `i.O.-Teile` | Gut-Teile (kein Defekt) | ~501 |

---

## Data Preparation Scripts

Run the two preparation scripts in order:

### Step 1 — HEIC → JPEG conversion

```bash
python scripts/convert_heic_to_jpg.py
```

Converts all `.HEIC` files in the source folders to `.jpg` and places them in `data/raw/<class>/`.
Requires `pillow-heif` (already in `requirements.txt`).

### Step 2 — Sequential rename + Pascal VOC annotation

```bash
python scripts/rename_and_annotate.py
```

Renames every image to `<Prefix>_N.jpg` (e.g. `Abdruck1_1.jpg`) and writes a matching Pascal VOC XML file to `data/annotations/`.

---

## Staged Format (what the pipeline reads)

After both scripts have run:

```
data/
├── raw/
│   ├── Abdruck 1/
│   │   ├── Abdruck1_1.jpg
│   │   └── …
│   ├── Abdruck 2/
│   │   └── …
│   ├── Stanzfehler/
│   │   └── …
│   └── i.O.-Teile/
│       └── …
└── annotations/
    ├── Abdruck1_1.xml
    ├── Abdruck2_1.xml
    └── …
```

`DefectDataset` (in `src/dataset.py`) reads every image under `data/raw/<class>/` and assigns the label from the `CLASS_NAMES` index in `config.py`. The annotations folder is not read by the classifier — it is kept for potential future object-detection extensions.

---

## Annotation XML Format (Pascal VOC)

Each `.xml` file covers one image. The `<bndbox>` spans the full image since no region-level labelling was done.

```xml
<annotation>
  <folder>Abdruck 1</folder>
  <filename>Abdruck1_1.jpg</filename>
  <source>
    <database>Produktionsdaten</database>
  </source>
  <size>
    <width>4032</width>
    <height>3024</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>Abdruck 1</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>0</xmin>
      <ymin>0</ymin>
      <xmax>4032</xmax>
      <ymax>3024</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## Class Imbalance

The dataset is significantly imbalanced (`Abdruck 2` ~963 vs. `Stanzfehler` ~156 samples). The training script compensates with a weighted `CrossEntropyLoss` computed from the actual training split at runtime — no manual weight tuning needed.

---

## Adapting for a New Domain

1. **Populate `data/raw/`** — one sub-folder per class, JPEG images inside.
2. **Update `CLASS_NAMES` in `model-training/config.py` and `inference-api/config.py`** — must match folder names exactly.
3. **Update the cockpit business logic** — `inference-cockpit/types/inspection.ts` contains priority levels, root-cause strings, and recommended actions specific to the current classes.
4. **Retrain from scratch** — `python src/train.py --restart`.
