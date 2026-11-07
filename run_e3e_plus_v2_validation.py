"""E3-E+ V2 guided-filter epsilon sweep. Validation split only.
Run from the project root: .\\.venv\\Scripts\\python.exe run_e3e_plus_v2_validation.py
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


def metrics(gt, pred):
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    fsim = float(calc_fsim(gt, pred))
    delta_e = float(calc_delta_e(gt, pred))
    g = rgb2gray(gt.astype(np.float32) / 255.0)
    p = rgb2gray(pred.astype(np.float32) / 255.0)
    return [psnr, ssim, fsim, delta_e, float(calc_epi(g, p)), float(calc_edge_retention(g, p))]


def main():
    data = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "validation"
    hazy_dir, clear_dir = data / "hazy", data / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders missing: {hazy_dir} | {clear_dir}")
    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No matched validation image pairs found.")
    print(f"Matched validation pairs: {len(pairs)}; unmatched hazy={len(unmatched_hazy)}, clear={len(unmatched_clear)}")

    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE,
        omega=OMEGA, gamma=GAMMA, t_min=T_MIN,
        adaptive_airlight=True, adaptive_strength=0.75,
        large_region_weight=0.25, refine_transmission=True,
        color_preserve_strength=0.40,
    )
    # Baseline is the frozen E3-E reference: radius 20, epsilon 0.001.
    variants = {
        "E3E_reference_r20_e1e-3": (20, 1e-3),
        "GF_r30_e1e-3": (30, 1e-3),
        "GF_r30_e2e-3": (30, 2e-3),
        "GF_r30_e3e-3": (30, 3e-3),
    }
    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    out = PROJECT_ROOT / "outputs" / "e3e_plus_v2_validation" / "CNN_Dataset_validation"
    vis = out / "visual_comparisons"
    vis.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hazy_path)
            gt = to_uint8(read_rgb(gt_path))
            row = {"filename": hazy_path.name, "ground_truth": gt_path.name}
            predictions = {}
            for name, (radius, eps) in variants.items():
                cfg_args = dict(common, guided_filter_radius=radius, guided_filter_eps=eps)
                cfg = DehazeConfig(**cfg_args)
                pred = to_uint8(dehaze_image(hazy, cfg)["dehazed"])
                if pred.shape != gt.shape:
                    raise ValueError(f"Image dimensions differ: prediction {pred.shape}, GT {gt.shape}")
                predictions[name] = pred
                for metric_name, value in zip(metric_names, metrics(gt, pred)):
                    row[f"{name}_{metric_name}"] = value
            rows.append(row)

            if index <= 12:
                tiles = [("Hazy", to_uint8(hazy)), ("Ground truth", gt)]
                tiles += [(name, predictions[name]) for name in variants]
                tw, th, label_h, gap, cols = 260, 180, 30, 8, 3
                nrows = int(np.ceil(len(tiles) / cols))
                canvas = Image.new("RGB", (cols * tw + (cols + 1) * gap,
                    nrows * (th + label_h) + (nrows + 1) * gap), "white")
                draw = ImageDraw.Draw(canvas)
                for j, (label, arr) in enumerate(tiles):
                    x = gap + (j % cols) * (tw + gap)
                    y = gap + (j // cols) * (th + label_h + gap)
                    canvas.paste(Image.fromarray(arr).resize((tw, th), Image.Resampling.LANCZOS), (x, y + label_h))
                    draw.text((x + 3, y + 7), label, fill="black")
                canvas.save(vis / f"{index:03d}_{hazy_path.stem}_e3e_plus_v2.jpg", quality=92)
        except Exception as exc:
            print(f"[SKIP] {hazy_path.name}: {exc}")
        if index % 10 == 0 or index == len(pairs):
            print(f"Progress: {index}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No validation images were successfully evaluated.")
    per_image = out / "e3e_plus_v2_per_image.csv"
    with per_image.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = out / "e3e_plus_v2_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "metric", "mean", "valid_images"])
        for name in variants:
            for metric_name in metric_names:
                values = np.asarray([r.get(f"{name}_{metric_name}", np.nan) for r in rows], dtype=float)
                values = values[np.isfinite(values)]
                writer.writerow([name, metric_name, float(values.mean()) if values.size else float("nan"), int(values.size)])
    print(f"\nCompleted: {len(rows)} validation pairs")
    print(f"Summary: {summary}\nPer-image: {per_image}\nVisuals: {vis}")
    print("No test images were read. DeltaE is CIE76 Lab DeltaE, not spatial S-CIELAB.")

if __name__ == "__main__":
    main()
