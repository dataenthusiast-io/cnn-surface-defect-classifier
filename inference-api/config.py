from pathlib import Path
import torch

# Inference-API is a sibling of model-training/; the trained artefacts live there.
_HERE           = Path(__file__).parent
_MT             = _HERE.parent / "model-training"

DATA_DIR        = _MT / "data" / "raw"
CHECKPOINT_DIR  = _MT / "outputs" / "checkpoints"

IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_WORKERS     = 0
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

OK_CLASSES      = ["patches"]
DEFECT_CLASSES  = ["crazing", "inclusion", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_NAMES     = ["i.O.", "n.i.O."]

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

API_HOST        = "127.0.0.1"
API_PORT        = 8000
