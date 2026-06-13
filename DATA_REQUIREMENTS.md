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

### Step 2 — Sequential rename

```bash
python scripts/rename_images.py
```

Renames every image to `<Prefix>_N.jpg` (e.g. `Abdruck1_1.jpg`).

---

## Staged Format (what the pipeline reads)

After both scripts have run:

```
data/
└── raw/
    ├── Abdruck 1/
    │   ├── Abdruck1_1.jpg
    │   └── …
    ├── Abdruck 2/
    │   └── …
    ├── Stanzfehler/
    │   └── …
    └── i.O.-Teile/
        └── …
```

`DefectDataset` (in `src/dataset.py`) reads every image under `data/raw/<class>/` and assigns the label from the `CLASS_NAMES` index in `config.py` — the folder name is the label, no annotation files are needed for classification.

---

## Class Imbalance

The dataset is significantly imbalanced (`Abdruck 2` ~963 vs. `Stanzfehler` ~156 samples). The training script compensates with a weighted `CrossEntropyLoss` computed from the actual training split at runtime — no manual weight tuning needed.

---

## Adapting for a New Domain

1. **Populate `data/raw/`** — one sub-folder per class, JPEG images inside.
2. **Update `CLASS_NAMES` in `model-training/config.py` and `inference-api/config.py`** — must match folder names exactly.
3. **Update the cockpit business logic** — `inference-cockpit/types/inspection.ts` contains priority levels, root-cause strings, and recommended actions specific to the current classes.
4. **Retrain from scratch** — `python src/train.py --restart`.
