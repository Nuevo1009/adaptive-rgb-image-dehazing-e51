"""
Diagnose whether the bright-region penalty changes estimated airlight.
Run from the project root where `dehazing/core.py` is importable.

Example:
python airlight_estimate_diagnostic.py --input-dir "path/to/hazy_images" --csv "targeted_region_ablation_per_image.csv" --output-dir "outputs/airlight_diagnostic"

The script uses the adaptive estimator in dehazing.core and writes:
- airlight_estimates.csv: baseline, penalty, and no-penalty A_far/A_near plus differences
- comparison images for the 10 images with largest detected regions
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from dehazing.core import (
    DehazeConfig, read_rgb, validate_rgb, save_rgb, dark_channel,
    estimate_airlight, estimate_airlight_adaptive, _large_uniform_near_white_mask
)

def downsample(rgb, factor):
    h, w = rgb.shape[:2]
    if factor <= 1:
        return rgb.copy()
    return cv2.resize(rgb, (max(1, round(w/factor)), max(1, round(h/factor))),
                      interpolation=cv2.INTER_AREA)

def estimate_for(rgb, cfg, weight):
    small = downsample(rgb, cfg.downsample_factor)
    dark = dark_channel(small, cfg.patch_size)
    bfar, bnear, _ = estimate_airlight(small, dark)
    c = DehazeConfig(**vars(cfg))
    c.large_region_weight = weight
    afar, anear, _, diag = estimate_airlight_adaptive(
        small, dark, c, baseline_values=(bfar, bnear)
    )
    return bfar, bnear, afar, anear, diag, small

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing hazy input images")
    ap.add_argument("--csv", required=True, help="targeted_region_ablation_per_image.csv")
    ap.add_argument("--output-dir", default="outputs/airlight_diagnostic")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    input_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if str(input_dir) == "YOUR_HAZY_IMAGES_FOLDER":
        ap.error('Replace "YOUR_HAZY_IMAGES_FOLDER" with the real folder containing your hazy input images.')
    if not input_dir.exists() or not input_dir.is_dir():
        ap.error(f"--input-dir does not exist or is not a folder: {input_dir}")
    if not Path(args.csv).is_file():
        ap.error(f"--csv file does not exist: {args.csv}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv).sort_values("region_fraction", ascending=False)
    cfg = DehazeConfig(adaptive_airlight=True, adaptive_strength=0.5,
                       large_region_weight=0.25)
    rows = []

    for idx, row in df.iterrows():
        name = str(row["hazy_image"])
        path = input_dir / name
        if not path.exists():
            # Permit matching by basename in immediate child folders.
            matches = list(input_dir.rglob(name))
            if not matches:
                rows.append({"hazy_image": name, "error": "input image not found"})
                continue
            path = matches[0]
        try:
            rgb = validate_rgb(read_rgb(path))
            bfar, bnear, pfar, pnear, pdiag, small = estimate_for(rgb, cfg, 0.25)
            _, _, nfar, nnear, ndiag, _ = estimate_for(rgb, cfg, 1.0)
            row_out = {
                "hazy_image": name,
                "region_fraction_csv": float(row["region_fraction"]),
                "detected_region_fraction_core": pdiag["large_uniform_near_white_fraction"],
            }
            for label, value in [
                ("baseline_A_far", bfar), ("baseline_A_near", bnear),
                ("penalty_A_far", pfar), ("penalty_A_near", pnear),
                ("no_penalty_A_far", nfar), ("no_penalty_A_near", nnear),
            ]:
                for i, ch in enumerate(["R", "G", "B"]):
                    row_out[f"{label}_{ch}"] = float(value[i])
            row_out["penalty_vs_no_penalty_A_far_L2"] = float(np.linalg.norm(pfar-nfar))
            row_out["penalty_vs_no_penalty_A_near_L2"] = float(np.linalg.norm(pnear-nnear))
            row_out["penalty_vs_baseline_A_far_L2"] = float(np.linalg.norm(pfar-bfar))
            row_out["penalty_vs_baseline_A_near_L2"] = float(np.linalg.norm(pnear-bnear))
            row_out["far_candidate_count"] = pdiag["far_candidate_count"]
            row_out["near_candidate_count"] = pdiag["near_candidate_count"]
            rows.append(row_out)

            if int((df["hazy_image"] == name).argmax()) < args.top_n:
                # Save visual comparison for the top-region images.
                c_pen = DehazeConfig(**vars(cfg)); c_pen.large_region_weight = 0.25
                c_none = DehazeConfig(**vars(cfg)); c_none.large_region_weight = 1.0
                from dehazing.core import dehaze_image
                out_pen = dehaze_image(rgb, c_pen)["dehazed"]
                out_none = dehaze_image(rgb, c_none)["dehazed"]
                stem = Path(name).stem
                save_rgb(out_dir / f"{stem}_input.png", rgb)
                save_rgb(out_dir / f"{stem}_penalty.png", out_pen)
                save_rgb(out_dir / f"{stem}_no_penalty.png", out_none)
                diff = np.abs(out_pen - out_none)
                save_rgb(out_dir / f"{stem}_absolute_difference.png", np.clip(diff * 5, 0, 1))
        except Exception as e:
            rows.append({"hazy_image": name, "error": str(e)})

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "airlight_estimates.csv", index=False)
    # Build the validity mask from the output DataFrame itself so its index aligns.
    if "penalty_vs_no_penalty_A_far_L2" in out.columns:
        valid_mask = out["penalty_vs_no_penalty_A_far_L2"].notna()
        good = out.loc[valid_mask].copy()
    else:
        good = out.iloc[0:0].copy()
    print(f"Processed {len(good)} images; errors/missing: {len(out)-len(good)}")
    if len(good):
        print("Mean airlight L2 difference (penalty vs no penalty):")
        print("  A_far:", good["penalty_vs_no_penalty_A_far_L2"].mean())
        print("  A_near:", good["penalty_vs_no_penalty_A_near_L2"].mean())
        print("CSV:", out_dir / "airlight_estimates.csv")
        print("Visual comparisons:", out_dir)

if __name__ == "__main__":
    main()
