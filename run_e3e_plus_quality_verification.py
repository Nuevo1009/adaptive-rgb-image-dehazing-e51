"""E3-E+ full-resolution quality verification on validation split only.

Run from project root:
  .\\.venv\\Scripts\\python.exe run_e3e_plus_quality_verification.py

Tests frozen E3-E, GF r30/e0.012, GF r30/e0.015, GF r30/e0.020.
The original atmospheric-light, transmission and reconstruction equations are
not modified. This script adds diagnostics only; it does not use test data.
"""
from pathlib import Path
import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb
from evaluate_extended import calc_fsim, calc_delta_e, calc_epi, calc_edge_retention, to_uint8

VARIANTS = {
    "E3E_reference_r20_e0.001": (20, 0.001),
    "GF_r30_e0.012": (30, 0.012),
    "GF_r30_e0.015": (30, 0.015),
    "GF_r30_e0.020": (30, 0.020),
}
METRICS = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]


def gray01(rgb_u8):
    return rgb2gray(rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def local_std(gray, radius=7):
    k = 2 * radius + 1
    mean = cv2.boxFilter(gray, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def quality_metrics(gt, pred):
    g = gray01(gt)
    p = gray01(pred)
    base = [
        float(peak_signal_noise_ratio(gt, pred, data_range=255)),
        float(structural_similarity(gt, pred, channel_axis=2, data_range=255)),
        float(calc_fsim(gt, pred)),
        float(calc_delta_e(gt, pred)),
        float(calc_epi(g, p)),
        float(calc_edge_retention(g, p)),
    ]

    # Ground-truth-defined darkest 20% avoids defining darkness from the output.
    cutoff = float(np.quantile(g, 0.20))
    dark = g <= cutoff
    err = (p - g)[dark]
    dark_mae = float(np.mean(np.abs(err))) if err.size else float("nan")
    dark_mse = float(np.mean(err * err)) if err.size else float("nan")
    dark_psnr = float(10.0 * np.log10(1.0 / dark_mse)) if dark_mse > 0 else float("inf")

    # Local contrast ratio in GT-defined dark regions; >1 can mean added texture/noise.
    gt_std = local_std(g)
    pred_std = local_std(p)
    dark_gt_contrast = float(np.mean(gt_std[dark])) if np.any(dark) else float("nan")
    dark_pred_contrast = float(np.mean(pred_std[dark])) if np.any(dark) else float("nan")
    dark_contrast_ratio = dark_pred_contrast / dark_gt_contrast if dark_gt_contrast > 1e-8 else float("nan")

    # Strong GT boundaries: compare output/GT gradient magnitude. This is a
    # diagnostic proxy for edge amplification/halo risk, not a validated halo detector.
    gx_g = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy_g = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gx_p = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy_p = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    mag_g = np.hypot(gx_g, gy_g)
    mag_p = np.hypot(gx_p, gy_p)
    edge_cut = float(np.quantile(mag_g, 0.90))
    strong = mag_g >= edge_cut
    ratios = mag_p[strong] / np.maximum(mag_g[strong], 1e-6)
    edge_ratio_mean = float(np.mean(ratios)) if ratios.size else float("nan")
    edge_ratio_p95 = float(np.quantile(ratios, 0.95)) if ratios.size else float("nan")
    edge_amp_fraction = float(np.mean(ratios > 1.5)) if ratios.size else float("nan")

    return base + [dark_mae, dark_psnr, dark_gt_contrast, dark_pred_contrast,
                   dark_contrast_ratio, edge_ratio_mean, edge_ratio_p95, edge_amp_fraction]


def save_fullres_samples(out_dir, index, stem, hazy, gt, preds):
    # Store original-resolution PNGs for manual zoom inspection, not resized panels.
    sample_dir = out_dir / "full_resolution_samples" / f"{index:03d}_{stem}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hazy).save(sample_dir / "hazy.png")
    Image.fromarray(gt).save(sample_dir / "ground_truth.png")
    for name, arr in preds.items():
        Image.fromarray(arr).save(sample_dir / f"{name}.png")

    # Contact sheet is explicitly labelled as a resized overview only.
    tiles = [("Hazy", hazy), ("Ground truth", gt)] + list(preds.items())
    tw, th, lh, gap, cols = 320, 220, 28, 8, 3
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * (th + lh) + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for j, (label, arr) in enumerate(tiles):
        x = gap + (j % cols) * (tw + gap)
        y = gap + (j // cols) * (th + lh + gap)
        draw.text((x + 3, y + 6), label, fill="black")
        tile = Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y + lh))
    canvas.save(out_dir / "visual_overviews" / f"{index:03d}_{stem}_quality.jpg", quality=94)


def main():
    data_dir = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "validation"
    hazy_dir, clear_dir = data_dir / "hazy", data_dir / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders missing: {hazy_dir} | {clear_dir}")
    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No matched validation pairs found.")

    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE,
        omega=OMEGA, gamma=GAMMA, t_min=T_MIN,
        adaptive_airlight=True, adaptive_strength=0.75,
        large_region_weight=0.25, refine_transmission=True,
        color_preserve_strength=0.40,
    )
    out_dir = PROJECT_ROOT / "outputs" / "e3e_plus_quality_verification" / "CNN_Dataset_validation"
    (out_dir / "visual_overviews").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    metric_names = METRICS + [
        "DarkRegion_MAE", "DarkRegion_PSNR_dB", "DarkRegion_GT_LocalStd",
        "DarkRegion_Pred_LocalStd", "DarkRegion_LocalContrast_Ratio",
        "StrongEdge_GradientRatio_Mean", "StrongEdge_GradientRatio_P95",
        "StrongEdge_AmplificationFraction_gt1.5",
    ]

    print(f"Matched validation pairs: {len(pairs)}; unmatched hazy={len(unmatched_hazy)}, clear={len(unmatched_clear)}")
    for idx, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = to_uint8(read_rgb(hazy_path))
            gt = to_uint8(read_rgb(gt_path))
            row = {"filename": hazy_path.name, "ground_truth": gt_path.name}
            preds = {}
            for name, (radius, eps) in VARIANTS.items():
                cfg = DehazeConfig(**common, guided_filter_radius=radius, guided_filter_eps=eps)
                result = dehaze_image(hazy, cfg)
                pred = to_uint8(result["dehazed"])
                if pred.shape != gt.shape:
                    raise ValueError(f"Shape mismatch pred={pred.shape}, GT={gt.shape}")
                preds[name] = pred
                vals = quality_metrics(gt, pred)
                for metric, value in zip(metric_names, vals):
                    row[f"{name}_{metric}"] = value
            rows.append(row)
            # Preserve original resolution for first 12 images for zoom inspection.
            if idx <= 12:
                save_fullres_samples(out_dir, idx, hazy_path.stem, hazy, gt, preds)
        except Exception as exc:
            skipped.append((hazy_path.name, str(exc)))
            print(f"[SKIP] {hazy_path.name}: {exc}")
        if idx % 10 == 0 or idx == len(pairs):
            print(f"Progress: {idx}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No validation images were evaluated successfully.")
    per_path = out_dir / "quality_verification_per_image.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary_path = out_dir / "quality_verification_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["configuration", "metric", "mean", "valid_images"])
        for name in VARIANTS:
            for metric in metric_names:
                vals = np.asarray([r.get(f"{name}_{metric}", np.nan) for r in rows], dtype=float)
                vals = vals[np.isfinite(vals)]
                w.writerow([name, metric, float(vals.mean()) if vals.size else float("nan"), int(vals.size)])

    skip_path = out_dir / "quality_verification_skipped.csv"
    with skip_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filename", "error"]); w.writerows(skipped)

    print(f"\nCompleted: {len(rows)} evaluated; skipped={len(skipped)}")
    print(f"Summary: {summary_path}\nPer-image: {per_path}\nSkipped: {skip_path}")
    print(f"Full-resolution PNG samples: {out_dir / 'full_resolution_samples'}")
    print("Only CNN_Dataset/validation was accessed; test data was not read.")
    print("Dark-region mask = darkest 20% of ground-truth luminance; local contrast = mean 15x15 local standard deviation within that mask.")
    print("Strong-edge ratios are diagnostic proxies, not a definitive halo detector. Inspect saved full-resolution PNGs manually.")
    print("Edge_Retention from evaluate_extended is mean predicted Sobel magnitude / mean GT Sobel magnitude; it is not bounded to [0,1].")
    print(f"Unmatched files: hazy={len(unmatched_hazy)}, clear={len(unmatched_clear)}")

if __name__ == "__main__":
    main()
