# Easy-run still-image dehazing project

## One-time configuration

1. Open `settings.py`.
2. Set `DATASET_ROOT` to the full path of the `CNN Dataset` folder, the folder containing `train` and `test`.
   Example:
   `DATASET_ROOT = Path(r"C:\Users\YourUsername\OneDrive\Desktop\Dataset\CNN Dataset")`
3. Save the file.

Expected dataset:
```text
CNN Dataset/
├── train/
│   ├── hazy/
│   └── clear/
└── test/
    ├── hazy/
    └── clear/
```

## Install once
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Then run from this project directory

```powershell
python run_single.py
python run_batch.py
python evaluate.py
python gamma_ablation.py
```

- `run_single.py` asks you to choose an image from `test/hazy`.
- `run_batch.py` processes all images from `test/hazy`.
- `evaluate.py` asks whether to evaluate `train` or `test` and compares against the matching `clear` folder.
- `gamma_ablation.py` compares gamma values on `train` pairs.
- `object_detection_eval.py` is optional and needs a local YOLO weights file plus the optional dependency.

## Equation / implementation notes
This is a still-image adaptation of the paper, not video processing. It implements dark-channel Eq. (2), reconstruction Eq. (6), atmospheric-light estimation related to Eqs. (7)-(10), modified transmission Eq. (12), and correction factor Eq. (13). Since still images have no previous frame, the current image mean gray is used for near/far grouping. Temporal smoothing is omitted. `t_min=0.05` is a numerical safeguard and is not a parameter specified by the paper. This is a floating-point Python reference, not bit-exact FPGA RTL.

Pairing is by matching filename stem, case-insensitive; clear and hazy images must have matching dimensions for PSNR/SSIM.
