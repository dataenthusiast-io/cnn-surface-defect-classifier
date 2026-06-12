from pathlib import Path
import torch

BASE_DIR        = Path(__file__).parent
ROOT_DIR        = BASE_DIR.parent

DATA_DIR        = ROOT_DIR / "data" / "raw"
OUTPUT_DIR      = BASE_DIR / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
PLOT_DIR        = OUTPUT_DIR / "plots"
LOG_PATH        = OUTPUT_DIR / "logs" / "metrics.csv"

IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_WORKERS     = 0
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# 4 production classes — labels assigned by Python-alphabetical folder name.
CLASS_NAMES = [
    "Abdruck 1",    # 0
    "Abdruck 2",    # 1
    "Stanzfehler",  # 2
    "i.O.-Teile",   # 3
]
NUM_CLASSES = len(CLASS_NAMES)

PHASE1_EPOCHS       = 5
PHASE1_LR_HEAD      = 1e-3

PHASE2_EPOCHS       = 20
PHASE2_LR_HEAD      = 1e-3
PHASE2_LR_BACKBONE  = 1e-4

PATIENCE            = 5

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
