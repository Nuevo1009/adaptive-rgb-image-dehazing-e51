"""
EDIT THIS FILE ONCE.
Set DATASET_ROOT to the full path of the folder named "CNN Dataset",
the folder containing train/ and test/.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# DATASET_ROOT = PROJECT_ROOT / "dataset" / "O-HAZY"

# HAZY_DIR = DATASET_ROOT / "ohaze_hazy"
# GT_DIR = DATASET_ROOT / "ohaze_GT"

# # Compatibility with the current gamma_ablation.py
# dataset_dir = DATASET_ROOT

# OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ohaze_ablation"

GAMMA_VALUES = [1.5, 2.0]

GAMMA = 2.0
OMEGA = 0.9375
DOWNSAMPLE_FACTOR = 3
PATCH_SIZE = 5
T_MIN = 0.05