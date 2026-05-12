# CNN Surface Defect Detector

Binary defect classification on steel surfaces using ResNet-18 transfer learning, a FastAPI inference service, and a Next.js real-time cockpit.

---

## Repository Layout

```
cnn-project/
├── model-training/      Training pipeline (PyTorch · ResNet-18)
├── inference-api/       REST inference service (FastAPI · uvicorn)
└── inference-cockpit/   Real-time inspection cockpit (Next.js 16)
```

Each component is self-contained. `inference-api` reads the trained checkpoint from `model-training/outputs/checkpoints/best_model.pth` and shares the same Python environment.

---

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.14 (managed by `model-training/.venv`) |
| Node.js | 18+ |
| Kaggle credentials | `~/.kaggle/kaggle.json` (for initial data download) |

---

## Quick Start

### 1 · Set up the Python environment

```bash
cd model-training
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2 · Download the dataset

```bash
.venv/bin/python3 scripts/download_data.py
# Downloads kaustubhdikshit/neu-surface-defect-database → data/raw/
# 1 800 images across 6 surface-defect classes
```

### 3 · Train the model

```bash
# Quick sanity check (2 epochs total, ~40 s on M2 MPS)
.venv/bin/python3 src/train.py --smoke-test

# Full training (Phase 1: 5 ep head-only · Phase 2: 20 ep fine-tune)
.venv/bin/python3 src/train.py

# Resume after interruption (picks up from outputs/checkpoints/resume.pth)
.venv/bin/python3 src/train.py

# Force restart from scratch
.venv/bin/python3 src/train.py --restart
```

Checkpoints and a CSV log are written to `model-training/outputs/`.

### 4 · Evaluate

```bash
.venv/bin/python3 src/evaluate.py
# Prints accuracy, F1, confusion matrix
# Saves plots to model-training/outputs/plots/
```

### 5 · Start the inference API

```bash
cd ../inference-api
../model-training/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
# Health check: http://127.0.0.1:8000/health
```

### 6 · Start the cockpit

```bash
cd ../inference-cockpit
npm install          # first time only
npm run dev
# Open: http://localhost:3000
```

---

## Model Results

Trained on 1 800 NEU surface-defect images; binary label: **i.O.** (patches class) vs **n.i.O.** (all 5 defect classes).

| Metric | Value | PRD Target |
|---|---|---|
| Test Accuracy | 99.6 % | > 88 % |
| Macro F1 | 0.993 | > 0.80 |
| False Negatives | 1 / 225 | < 15 % FNR |
| Training Time | ~80 s (M2 MPS) | < 15 min |

---

## Architecture

```
NEU Dataset  ──►  model-training/
                    ResNet-18 (ImageNet pre-trained)
                    Phase 1 – head only (5 epochs)
                    Phase 2 – fine-tune layer3+4+fc (≤20 epochs, early stop)
                    outputs/checkpoints/best_model.pth
                           │
                           ▼
                  inference-api/  (FastAPI :8000)
                    GET /health
                    GET /predict/next    ← iterate test set
                    GET /predict/reset
                    GET /stats
                           │  HTTP polling
                           ▼
                  inference-cockpit/  (Next.js :3000)
                    Live conveyor view  (current image + verdict)
                    Running stats       (accuracy, F1, FP/FN counters)
                    Confidence chart    (Recharts line chart)
                    Inspection log      (scrollable history)
```

---

## Component Details

### `model-training/`

```
model-training/
├── src/
│   ├── dataset.py      NEU loader, binary label mapping, stratified split
│   ├── model.py        ResNet-18 builder, freeze/unfreeze helpers
│   ├── train.py        Two-phase training loop, tqdm progress, checkpointing
│   ├── evaluate.py     Test-set metrics + plot generation
│   └── predict.py      Single-image CLI inference
├── scripts/
│   └── download_data.py   kagglehub download + merge train+val splits
├── data/raw/           1 800 .bmp images (git-ignored)
├── outputs/
│   ├── checkpoints/    best_model.pth · last_model.pth · resume.pth
│   ├── logs/           metrics.csv (epoch-by-epoch)
│   └── plots/          training_curves · confusion_matrix · sample_predictions …
├── config.py           All hyper-parameters and path roots
└── requirements.txt
```

### `inference-api/`

```
inference-api/
├── api/
│   ├── main.py         FastAPI app + CORS + endpoints
│   └── pipeline.py     InferencePipeline: loads model, iterates test set
├── src/
│   ├── model.py        Inference-only build_model()
│   └── dataset.py      get_test_dataset() helper
├── config.py           Paths (→ ../model-training/outputs), device, class names
└── requirements.txt
```

### `inference-cockpit/`

Standard Next.js 16 App Router project with Tailwind CSS, shadcn-style components, and Recharts. No build-time configuration needed beyond `npm install`.

---

## Development Notes

- **Python env**: a single `.venv` inside `model-training/` is used by both training and the inference API.
- **MPS / CUDA / CPU**: device is selected automatically in `config.py`; `NUM_WORKERS=0` avoids fork issues on macOS.
- **Checkpoint recovery**: training can be interrupted and resumed at any time; the resume checkpoint saves model + optimizer + scheduler state.
- **CORS**: the API allows all origins in dev; restrict `allow_origins` in `inference-api/api/main.py` for production.
