"""E3-E+ ablation runner. Validation split only; never reads the test split.
Run from project root: python run_e3e_plus_validation.py
"""
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb
from evaluate_extended import calc_fsim, calc_delta_e, calc_epi, calc_edge_retention, to_uint8


def metric_values(gt, pred):
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    fsim = calc_fsim(gt, pred)
    delta_e = calc_delta_e(gt, pred)
    g = rgb2gray(gt.astype(np.float32) / 255.0)
    p = rgb2gray(pred.astype(np.float32) / 255.0)
    return [psnr, ssim, fsim, delta_e, calc_epi(g, p), calc_edge_retention(g, p)]


def main():
    root = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "validation"
    hazy_dir, clear_dir = root / "hazy", root / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders not found: {hazy_dir} and {clear_dir}")
    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No safely matched validation image pairs found.")
    print(f"Validation pairs: {len(pairs)} | unmatched hazy: {len(unmatched_hazy)} | unmatched clear: {len(unmatched_clear)}")

    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE,
        omega=OMEGA, gamma=GAMMA, t_min=T_MIN,
        adaptive_airlight=True, adaptive_strength=0.75,
        large_region_weight=0.25, refine_transmission=True,
        guided_filter_radius=20, guided_filter_eps=1e-3,
        color_preserve_strength=0.40,
    )
    # Frozen E3-E reference, followed by one-factor-at-a-time experiments.
    variants = {
        "E3E_reference": {},
        "GF_r10_eps1e-3": {"guided_filter_radius": 10, "guided_filter_eps": 1e-3},
        "GF_r20_eps5e-4": {"guided_filter_radius": 20, "guided_filter_eps": 5e-4},
        "GF_r30_eps2e-3": {"guided_filter_radius": 30, "guided_filter_eps": 2e-3},
        "TFloor_strength0.01": {"adaptive_t_floor": True, "t_floor_strength": 0.01},
        "TFloor_strength0.02": {"adaptive_t_floor": True, "t_floor_strength": 0.02},
        "AlphaAdaptive_gain0.10": {"adaptive_color_alpha": True, "color_alpha_darkness_gain": 0.10},
        "AlphaAdaptive_gain0.20": {"adaptive_color_alpha": True, "color_alpha_darkness_gain": 0.20},
    }
    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    out_dir = PROJECT_ROOT / "outputs" / "e3e_plus_validation" / "CNN_Dataset_validation"
    visual_dir = out_dir / "visual_comparisons"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, (hp, gp) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hp)
            gt = to_uint8(read_rgb(gp))
            row = {"filename": hp.name, "ground_truth": gp.name}
            predictions = {}
            for name, overrides in variants.items():
                cfg_values = dict(common)
                cfg_values.update(overrides)
                cfg = DehazeConfig(**cfg_values)
                pred = to_uint8(dehaze_image(hazy, cfg)["dehazed"])
                if pred.shape != gt.shape:
                    raise ValueError(f"shape mismatch: pred={pred.shape}, GT={gt.shape}")
                predictions[name] = pred
                for metric, value in zip(metric_names, metric_values(gt, pred)):
                    row[f"{name}_{metric}"] = value
            rows.append(row)

            if idx <= 12:
                tiles = [("Hazy", to_uint8(hazy)), ("Ground truth", gt)]
                tiles += [(name, predictions[name]) for name in variants]
                tw, th, lh, gap, cols = 260, 180, 28, 8, 3
                nrows = int(np.ceil(len(tiles) / cols))
                canvas = Image.new("RGB", (cols * tw + (cols + 1) * gap, nrows * (th + lh) + (nrows + 1) * gap), "white")
                draw = ImageDraw.Draw(canvas)
                for j, (label, arr) in enumerate(tiles):
                    x = gap + (j % cols) * (tw + gap)
                    y = gap + (j // cols) * (th + lh + gap)
                    im = Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS)
                    canvas.paste(im, (x, y + lh))
                    draw.text((x + 3, y + 6), label, fill="black")
                canvas.save(visual_dir / f"{idx:03d}_{hp.stem}_e3e_plus.jpg", quality=90)
        except Exception as exc:
            print(f"[SKIP] {hp.name}: {exc}")
        if idx % 10 == 0 or idx == len(pairs):
            print(f"Progress: {idx}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No validation pairs were successfully evaluated.")
    per_file = out_dir / "e3e_plus_per_image.csv"
    with per_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    summary = out_dir / "e3e_plus_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "metric", "mean", "valid_images"])
        for name in variants:
            for metric in metric_names:
                vals = np.asarray([r.get(f"{name}_{metric}", np.nan) for r in rows], dtype=float)
                vals = vals[np.isfinite(vals)]
                writer.writerow([name, metric, float(vals.mean()) if len(vals) else float("nan"), int(len(vals))])
    print(f"\nEvaluated validation images: {len(rows)}")
    print(f"Summary CSV: {summary}\nPer-image CSV: {per_file}\nVisual comparisons: {visual_dir}")
    print("Note: CIE76_DeltaE is ordinary CIE76 Lab DeltaE, not spatial S-CIELAB.")

if __name__ == "__main__":
    main()
