# Data Requirements

This document describes the exact data format expected by the pipeline so the repo can be re-used as a boilerplate for any image classification domain, not only the original NEU steel surface dataset.

---

## Source Format (what you bring in)

The pipeline was designed against the **NEU-DET** dataset structure (Pascal VOC-style). New datasets should match this layout.

```
<dataset-root>/
├── train/
│   ├── images/
│   │   ├── <class_a>/
│   │   │   ├── <class_a>_1.jpg
│   │   │   ├── <class_a>_2.jpg
│   │   │   └── …
│   │   ├── <class_b>/
│   │   └── …
│   └── annotations/
│       ├── <class_a>_1.xml
│       ├── <class_a>_2.xml
│       └── …                   ← flat folder, one XML per image
└── validation/
    ├── images/
    │   ├── <class_a>/
    │   └── …
    └── annotations/
        └── …
```

### Key constraints

| Property | Value |
|---|---|
| Image format | JPEG (`.jpg` / `.jpeg`); PNG and BMP also accepted by the loader |
| Image dimensions | Any fixed square or rectangular resolution; NEU-DET uses **200 × 200 px** |
| Image colour depth | Grayscale content is fine — store as 3-channel RGB (even if R = G = B) |
| Class folders | One sub-folder per class inside `images/`; folder name **is** the class label |
| Annotation files | One XML per image, named `<image_stem>.xml`, placed flat in `annotations/` |
| Splits | A `train/` and `validation/` split at minimum; ratio is not critical (the pipeline re-splits internally) |
| Class balance | Aim for equal samples per class; NEU-DET has **300 per class** (240 train + 60 val) |

---

## Annotation XML Format (Pascal VOC)

Each `.xml` file describes one image and may contain **one or more defect regions**.

```xml
<annotation>
  <folder>cr</folder>                   <!-- short label or folder abbreviation; informational only -->
  <filename>crazing_1.jpg</filename>    <!-- must match the paired .jpg filename -->

  <source>
    <database>YOUR-DATASET-NAME</database>
  </source>

  <size>
    <width>200</width>                  <!-- image width in pixels -->
    <height>200</height>                <!-- image height in pixels -->
    <depth>1</depth>                    <!-- 1 = grayscale intent (even if stored as RGB) -->
  </size>

  <segmented>0</segmented>              <!-- always 0 — no polygon segmentation used -->

  <!-- repeat <object> for every defect region in the image -->
  <object>
    <name>crazing</name>                <!-- must match the class folder name exactly -->
    <pose>Unspecified</pose>
    <truncated>0</truncated>            <!-- 1 if the defect is cut off at image edge -->
    <difficult>0</difficult>            <!-- 1 if hard to classify; loader can filter these out -->
    <bndbox>
      <xmin>2</xmin>                    <!-- pixel coordinates, 0-indexed, top-left origin -->
      <ymin>2</ymin>
      <xmax>193</xmax>
      <ymax>194</ymax>
    </bndbox>
  </object>

</annotation>
```

### Notes on the `<object>` block

- An image can contain **1–N defect instances** — NEU-DET has up to 9 per image.
- All `<object>` entries in a single file belong to the **same class** (one class per image). Mixed-class images are technically supported by the XML schema but are not used here.
- `<difficult>` marks ambiguous or borderline examples. The current classifier ignores this flag — all objects are treated equally. If you want to exclude difficult examples during training, filter on this field in `dataset.py`.
- `<bndbox>` coordinates are used for **object detection extensions only**. The current classification pipeline does not read bounding boxes — the class label comes from the folder name alone.

---

## Staged Format (what the pipeline actually reads)

The `download_data.py` script merges `train/images/<class>/` and `validation/images/<class>/` into a single flat pool:

```
data/raw/
├── <class_a>/
│   ├── train_<class_a>_1.jpg
│   ├── train_<class_a>_2.jpg
│   ├── validation_<class_a>_241.jpg
│   └── …
├── <class_b>/
└── …
```

- Files are prefixed with `train_` or `validation_` to avoid name collisions.
- The annotations folder is **not copied** — the classifier only needs the images.
- `dataset.py` (via `NEUDefectDataset`) reads every image file under `data/raw/<class>/` and assigns a label by the **sorted alphabetical position** of the folder name.

This is the only format the training and inference pipelines read. If you supply your own data, you can skip the Kaggle download entirely and populate `data/raw/` directly in this structure.

---

## Adapting for a New Domain

### Minimum checklist

1. **Populate `data/raw/`** — one sub-folder per class, images inside.

2. **Update `CLASS_NAMES` in `model-training/config.py`** — must match folder names exactly, in alphabetical order (the sort order determines the integer label):

   ```python
   CLASS_NAMES = [
       "class_a",   # label 0
       "class_b",   # label 1
       "class_c",   # label 2
   ]
   NUM_CLASSES = len(CLASS_NAMES)
   ```

3. **Update `NUM_CLASSES`** — ResNet-18 head is built with this value; must equal `len(CLASS_NAMES)`.

4. **Update the cockpit business logic** — `inference-cockpit/lib/` contains the priority levels, root-cause strings, and recommended actions that are currently specific to steel surface defects. Search for `KRITISCH`, `Ursache`, `Maßnahme` to find all display strings that need replacing.

5. **Retrain from scratch** — use `src/train.py --restart` to discard any existing checkpoint and start fresh with the new class set.

### Optional: bring your own XML annotations

The XML annotations are not required by the classifier. They are included in the source format description above because:

- They are present in NEU-DET and document the defect locations precisely.
- They enable a future upgrade to **object detection** (e.g. YOLO, Faster R-CNN) without re-labelling.
- If you plan to stay classification-only, plain images in class folders are sufficient — no XML needed.

### Sample counts

The model reaches ≥ 99 % accuracy on NEU-DET with 300 images per class. As a rule of thumb:

| Samples per class | Expected outcome |
|---|---|
| < 50 | Likely to underfit; consider heavy augmentation or a smaller backbone |
| 50 – 150 | Workable with strong augmentation and a frozen backbone (Phase 1 only) |
| 150 – 500 | Good fit for the current two-phase ResNet-18 transfer learning setup |
| 500+ | Consider unfreezing more backbone layers or switching to a larger model |

---

## File Naming Convention (informational)

NEU-DET uses `<class>_<n>.jpg` / `<class>_<n>.xml` where `n` is a sequential integer starting at 1. The pipeline does not enforce this — any unique filename is valid as long as:

- The image file extension is `.jpg`, `.jpeg`, `.png`, or `.bmp`.
- The XML `<filename>` tag matches the paired image file name (only relevant if you add detection code later).
- File names within a class folder are unique.
