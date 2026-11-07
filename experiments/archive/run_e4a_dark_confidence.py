from pathlib import Path
import csv
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dehazing.core import DehazeConfig, dehaze_image, read_rgb, save_rgb
from dataset_config import match_pairs

DATA = ROOT / "dataset" / "CNN_Dataset" / "validation"
OUT = ROOT / "outputs" / "E4A_dark_confidence"
HAZY = DATA / "hazy"
CLEAR = DATA / "clear"

# Keep these E3-E settings aligned with the experiment used for the reported
# E3-E validation PSNR. Do not change these while comparing E4-A strengths.
BASE = dict(
    downsample_factor=3,
    patch_size=5,
    omega=0.9375,
    gamma=2.0,
    t_min=0.05,
    adaptive_airlight=True,
    adaptive_strength=0.75,
    large_region_weight=0.25,
    refine_transmission=True,
    guided_filter_radius=20,
    guided_filter_eps=1e-3,
    adaptive_omega=False,
    color_preserve_strength=0.40,
)

VARIANTS = [
    ("E3E_reference", False, 0.00),
    ("E4A_alpha_0.25", True, 0.25),
    ("E4A_alpha_0.35", True, 0.35),
    ("E4A_alpha_0.40", True, 0.40),
]

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)

def ssim_global(a, b):
    # Global SSIM approximation for a quick independent sanity check.
    # Use evaluate_extended.py for the project's official SSIM implementation.
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = 0.01**2, 0.03**2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a-ma)*(b-mb))
    return ((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))

def main():
    
    # Build pairs directly from the validation folders.
    # This avoids the nested-tuple format returned by match_pairs().
    hazy_files = sorted(
        p for p in HAZY.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    clear_lookup = {
        p.name: p
        for p in CLEAR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }

    pairs = [
        (hazy_path, clear_lookup[hazy_path.name])
        for hazy_path in hazy_files
        if hazy_path.name in clear_lookup
    ]  

    if not pairs:
        raise RuntimeError(
            f"No matching image pairs found. Check these folders:\n{HAZY}\n{CLEAR}"
        )

    print(f"Validation hazy folder: {HAZY}")
    print(f"Validation clear folder: {CLEAR}")
    print(f"Matched validation pairs: {len(pairs)}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, pair in enumerate(pairs, 1):
        # match_pairs is used by the existing project; support tuple/list pairs.
        hazy_path, clear_path = pair
        hazy = read_rgb(hazy_path).astype(np.float32) / 255.0
        clear = read_rgb(clear_path).astype(np.float32) / 255.0

        for name, enabled, alpha in VARIANTS:
            cfg = DehazeConfig(
                **BASE,
                confidence_reconstruction=enabled,
                confidence_strength=alpha,
            )
            result = dehaze_image(hazy, cfg)
            pred = result["dehazed"]
            save_rgb(OUT / name / Path(hazy_path).name, pred)

            rows.append({
                "image": Path(hazy_path).name,
                "variant": name,
                "PSNR": psnr(clear, pred),
                "SSIM_global_check": ssim_global(clear, pred),
                "confidence_mean": (
                    float(np.mean(result["confidence_map"]))
                    if result.get("confidence_map") is not None else ""
                ),
            })

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(pairs)} validation pairs")

    csv_path = OUT / "e4a_quick_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved outputs: {OUT}")
    print(f"Saved quick comparison CSV: {csv_path}")
    print("Use evaluate_extended.py for the official six-metric evaluation.")
    print("Do not select a variant from SSIM_global_check; it is only a sanity check.")

if __name__ == "__main__":
    main()
