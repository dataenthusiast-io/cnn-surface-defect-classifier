# Critical Project Analysis — CNN Surface Defect Classifier

This analysis covers the code, training logs, output plots, and the raw data itself — including an empirical data-leakage experiment.

**Overall verdict up front:** The engineering is above average and clean, but the headline metrics (96.3 % test accuracy, macro F1 0.93) are very likely heavily inflated, because the dataset consists of near-duplicate images that were distributed across the train and test splits. This was demonstrated empirically (Section 3).

---

## 1. The ML Setup

**Model** (`model-training/src/model.py`): ResNet-18 with ImageNet weights, the head replaced by `Dropout(0.3) + Linear(512→4)`. Two-phase transfer learning:

- **Phase 1** (5 epochs): backbone frozen, only the head trains with LR 1e-3
- **Phase 2** (max. 20 epochs): layer3 + layer4 + head trainable, discriminative LRs (backbone 1e-4, head 1e-3), cosine annealing, early stopping (patience 5)

In addition: weighted `CrossEntropyLoss` to compensate for class imbalance (963 "Abdruck 2" vs. 156 "Stanzfehler"), stratified 70/15/15 split, model selection via validation macro F1.

**This is a clean, textbook setup** — the architecture choice (ResNet-18 for ~1,800 images) is appropriately sized.

## 2. Training Behaviour

From `outputs/logs/metrics.csv` (23 epochs, early stop):

| | Phase 1 (Ep. 1–5) | Phase 2 (Ep. 6–23) |
|---|---|---|
| Val accuracy | 66 % → 85 % | 85 % → **97.0 %** (Ep. 18) |
| Val macro F1 | 0.52 → 0.73 | 0.80 → **0.945** (Ep. 18) |

The training curves are unremarkable in a good way: loss decreases consistently, there is no classic overfitting pattern (validation loss does not diverge), and the phase transition at epoch 5 produces the expected jump.

On the test set: **96.3 % accuracy, 10 errors out of 270**. The confusion matrix shows that all errors are confusions *between defect classes* (mainly Abdruck 2 → Abdruck 1); not a single defect was waved through as "i.O." (OK). On paper, this is ideal from a business perspective.

## 3. The Central Weakness: Data Leakage via Near-Duplicates

This is where it gets critical. `sample_predictions.png` and `misclassifications.png` show **practically identical images**: the same round part, same viewing angle, same metal background, same lighting.

This was quantified as follows: the project's exact split was replicated (seed 42, stratified 70/15/15), and for every test image the most similar training neighbour was found via perceptual hashing (dHash, 64 bit):

- **Abdruck 1: 25 of 25 test images** have a near-duplicate in the training set (Hamming distance ≤ 4 of 64 bits; several with distance **0** — practically identical)
- **Stanzfehler: 23 of 24 test images** likewise

Concretely, this means: the test does not measure whether the model can classify *new parts* — it measures whether the model recognises *photos it has already seen in almost identical form*. The root cause lies in the data collection (many iPhone burst-style photos of the same physical parts in the same capture session) combined with a **split at image level instead of part level** (`dataset.py:91`). Since no part IDs were recorded, a correct group-based split is not even possible retroactively.

**The 96 % figure is therefore not a reliable estimate of generalisation.** An honest test would require a separate capture session with different parts, a different background, and different lighting.

## 4. Further Critical Points

### Methodology

- **The split is not persisted**: `get_test_dataset()` reconstructs the split at API runtime from the folder contents plus the seed. If a single image is added, the split silently shifts and the API "tests" on training images. Saving a split manifest (JSON of file names) at training time should be mandatory.
- **Reproducibility is incomplete**: `RANDOM_SEED` only governs the split. Torch initialisation, DataLoader shuffling, and augmentation are unseeded — two training runs produce different models.
- **Small per-class test sets**: the "100 % recall" for Abdruck 1 is based on n=25 — the 95 % confidence interval extends down to ~86 %. Such numbers without uncertainty estimates look more precise than they are. No cross-validation; a single split.
- **`Resize((224, 224))`** squashes the 4:3 iPhone images anisotropically — the round parts become ellipses. It works here because it is applied consistently, but it discards information.
- **Uncalibrated confidence**: the confidence distribution is glued to 1.0 (typical softmax overconfidence). Since the cockpit displays these values to operators and business logic depends on them, temperature scaling would be appropriate.

### Data & Surrounding Architecture

- The **1,783 Pascal VOC XML files carry no information** — every bounding box spans the full image (as documented in `DATA_REQUIREMENTS.md`). Effort without benefit; real defect boxes could have been used to validate the Grad-CAM regions.
- **`CLASS_NAMES` is maintained twice** (model-training and inference-api) and must be kept in sync manually — a classic source of silent failures. The comment in `config.py:21` ("Python-alphabetical") is also misleading; the mapping is explicit.
- The **inference API is a demo, not an inference service**: it iterates over the local test set instead of offering an upload endpoint. Legitimate for a presentation, but it should be named as such.
- The Grad-CAM implementation itself is correct (hooks on layer4, ReLU-weighted activations), but the bounding box derived from it (a single box over the 0.5-threshold mask) is crude — disconnected activation regions are merged into one oversized box.

## 5. What Is Done Well

To be fair: resume checkpointing with optimizer/scheduler state, CSV logging, early stopping, weighted loss, stratified splitting, a hand-written Grad-CAM, and a clean repo structure with honest documentation — this is clearly above the level of typical semester projects. The fact that the test set is only touched once at the end (selection runs on validation data) is also methodologically sound. The problem is not in the code, but **one level deeper: in the data collection**.

## 6. Key Recommendations (Prioritised)

1. **Create an honest test set**: a separate capture session (different parts, different day, ideally a different background) and evaluate on it — the number will drop significantly, but it will then be trustworthy, and that drop is exactly the interesting finding for the report.
2. For future data collection, **record part IDs** and split by part (group split).
3. **Persist a split manifest** and load it in the API.
4. Calibrate confidence (temperature scaling), report confidence intervals, and seed everything for full reproducibility.
