"""E3 follow-up: evaluate alpha 0.30/0.35/0.40 and save visual panels.
Run from the project root. Uses validation split only; does not modify project files.
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw
from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb
from color_preservation_ablation import uint8, metrics


def main():
    root = PROJECT_ROOT / "dataset" / "CNN_Dataset"
    hazy_dir, clear_dir = root / "validation" / "hazy", root / "validation" / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders not found: {hazy_dir} and {clear_dir}")
    pairs, uh, uc = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No paired validation images found")
    print(f"Matched validation pairs: {len(pairs)}")
    common = dict(downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE,
        omega=OMEGA, gamma=GAMMA, t_min=T_MIN, adaptive_airlight=True,
        adaptive_strength=0.75, large_region_weight=0.25,
        refine_transmission=True, guided_filter_radius=20, guided_filter_eps=1e-3)
    alphas = [0.0, 0.30, 0.35, 0.40]
    labels = {0.0: "E1 alpha=0.00", 0.30: "E3-C alpha=0.30",
              0.35: "E3-D alpha=0.35", 0.40: "E3-E alpha=0.40"}
    configs = {a: DehazeConfig(**common, color_preserve_strength=a) for a in alphas}
    out = PROJECT_ROOT / "outputs" / "e3_alpha_followup" / "CNN_Dataset_validation"
    vis = out / "visual_comparisons"
    vis.mkdir(parents=True, exist_ok=True)
    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    rows = []
    for i, (hp, gp) in enumerate(pairs, 1):
        try:
            hazy, gt = read_rgb(hp), uint8(read_rgb(gp))
            row = {"filename": hp.name, "ground_truth": gp.name}
            outputs = {}
            for a, cfg in configs.items():
                pred = uint8(dehaze_image(hazy, cfg)["dehazed"])
                outputs[a] = pred
                for mn, val in zip(metric_names, metrics(gt, pred)):
                    row[f"alpha_{a:.2f}_{mn}"] = val
            rows.append(row)
            if i <= 12:
                tiles = [("Hazy", uint8(hazy)), ("E1", outputs[0.0]),
                         ("E3-C 0.30", outputs[0.30]), ("E3-D 0.35", outputs[0.35]),
                         ("E3-E 0.40", outputs[0.40]), ("Ground truth", gt)]
                # Uniform thumbnail grid, with labels.
                thumb_w, thumb_h, label_h, gap = 320, 220, 30, 10
                canvas = Image.new("RGB", (2*thumb_w + 3*gap, 3*(thumb_h+label_h) + 4*gap), "white")
                draw = ImageDraw.Draw(canvas)
                for j, (name, arr) in enumerate(tiles):
                    im = Image.fromarray(arr).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    x = gap + (j % 2) * (thumb_w + gap)
                    y = gap + (j // 2) * (thumb_h + label_h + gap)
                    canvas.paste(im, (x, y + label_h))
                    draw.text((x + 4, y + 7), name, fill="black")
                canvas.save(vis / f"{i:03d}_{hp.stem}_e3_alpha_comparison.jpg", quality=92)
        except Exception as exc:
            print(f"[SKIP] {hp.name}: {exc}")
        if i % 10 == 0 or i == len(pairs):
            print(f"Progress: {i}/{len(pairs)}", end="\r", flush=True)
    if not rows:
        raise SystemExit("No validation images evaluated")
    per_path = out / "e3_alpha_per_image.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary_path = out / "e3_alpha_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["configuration", "metric", "mean", "valid_images"])
        for a in alphas:
            for mn in metric_names:
                vals = np.asarray([r[f"alpha_{a:.2f}_{mn}"] for r in rows], dtype=float)
                vals = vals[np.isfinite(vals)]
                w.writerow([labels[a], mn, float(vals.mean()) if len(vals) else float("nan"), len(vals)])
    print(f"\nEvaluated: {len(rows)}")
    print(f"Summary: {summary_path}\nPer-image: {per_path}\nVisuals: {vis}")

if __name__ == "__main__":
    main()
