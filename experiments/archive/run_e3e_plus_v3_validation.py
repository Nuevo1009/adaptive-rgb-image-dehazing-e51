"""E3-E+ V3 guided-filter epsilon refinement; validation split only.
Run from the project root:
  .\\.venv\\Scripts\\python.exe run_e3e_plus_v3_validation.py

The paper's atmospheric-light, transmission and reconstruction equations are
unchanged. Only guided-filter radius/epsilon are varied. No test data is read.
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
from evaluate_extended import (
    calc_fsim, calc_delta_e, calc_epi, calc_edge_retention, to_uint8
)


def calculate_metrics(gt, pred):
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    fsim = float(calc_fsim(gt, pred))
    delta_e = float(calc_delta_e(gt, pred))
    gray_gt = rgb2gray(gt.astype(np.float32) / 255.0)
    gray_pred = rgb2gray(pred.astype(np.float32) / 255.0)
    epi = float(calc_epi(gray_gt, gray_pred))
    edge = float(calc_edge_retention(gray_gt, gray_pred))
    return [psnr, ssim, fsim, delta_e, epi, edge]


def main():
    data_dir = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "validation"
    hazy_dir, clear_dir = data_dir / "hazy", data_dir / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders missing: {hazy_dir} | {clear_dir}")

    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No matched validation image pairs found.")
    print(
        f"Matched validation pairs: {len(pairs)}; "
        f"unmatched hazy={len(unmatched_hazy)}, clear={len(unmatched_clear)}"
    )

    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR,
        patch_size=PATCH_SIZE,
        omega=OMEGA,
        gamma=GAMMA,
        t_min=T_MIN,
        adaptive_airlight=True,
        adaptive_strength=0.75,
        large_region_weight=0.25,
        refine_transmission=True,
        color_preserve_strength=0.40,
    )

    # E3-E reference and best V2 setting are retained as controls.
    variants = {
        "E3E_reference_r20_e1e-3": (20, 0.0010),
        "GF_r30_e2p5e-3": (30, 0.0025),
        "GF_r30_e3e-3_previous_best": (30, 0.0030),
        "GF_r30_e3p5e-3": (30, 0.0035),
        "GF_r30_e4e-3": (30, 0.0040),
    }
    metric_names = [
        "PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"
    ]
    out_dir = (
        PROJECT_ROOT / "outputs" / "e3e_plus_v3_validation"
        / "CNN_Dataset_validation"
    )
    visual_dir = out_dir / "visual_comparisons"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    for index, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = to_uint8(read_rgb(hazy_path))
            gt = to_uint8(read_rgb(gt_path))
            row = {"filename": hazy_path.name, "ground_truth": gt_path.name}
            predictions = {}
            for name, (radius, epsilon) in variants.items():
                config = DehazeConfig(
                    **common,
                    guided_filter_radius=radius,
                    guided_filter_eps=epsilon,
                )
                result = dehaze_image(hazy, config)
                pred = to_uint8(result["dehazed"])
                if pred.shape != gt.shape:
                    raise ValueError(
                        f"Shape mismatch: prediction={pred.shape}, ground_truth={gt.shape}"
                    )
                predictions[name] = pred
                for metric_name, value in zip(metric_names, calculate_metrics(gt, pred)):
                    row[f"{name}_{metric_name}"] = value
            rows.append(row)

            if index <= 12:
                tiles = [("Hazy", hazy), ("Ground truth", gt)]
                tiles.extend((name, predictions[name]) for name in variants)
                tile_w, tile_h, label_h, gap, cols = 260, 180, 30, 8, 3
                row_count = int(np.ceil(len(tiles) / cols))
                canvas = Image.new(
                    "RGB",
                    (cols * tile_w + (cols + 1) * gap,
                     row_count * (tile_h + label_h) + (row_count + 1) * gap),
                    "white",
                )
                draw = ImageDraw.Draw(canvas)
                for tile_index, (label, array) in enumerate(tiles):
                    x = gap + (tile_index % cols) * (tile_w + gap)
                    y = gap + (tile_index // cols) * (tile_h + label_h + gap)
                    tile = Image.fromarray(array).resize(
                        (tile_w, tile_h), Image.Resampling.LANCZOS
                    )
                    canvas.paste(tile, (x, y + label_h))
                    draw.text((x + 3, y + 7), label, fill="black")
                canvas.save(
                    visual_dir / f"{index:03d}_{hazy_path.stem}_e3e_plus_v3.jpg",
                    quality=92,
                )
        except Exception as exc:
            skipped.append((hazy_path.name, str(exc)))
            print(f"[SKIP] {hazy_path.name}: {exc}")

        if index % 10 == 0 or index == len(pairs):
            print(f"Progress: {index}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No validation images were successfully evaluated.")

    per_image_path = out_dir / "e3e_plus_v3_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "e3e_plus_v3_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "metric", "mean", "valid_images"])
        for name in variants:
            for metric_name in metric_names:
                values = np.asarray(
                    [r.get(f"{name}_{metric_name}", np.nan) for r in rows],
                    dtype=float,
                )
                values = values[np.isfinite(values)]
                writer.writerow([
                    name,
                    metric_name,
                    float(values.mean()) if values.size else float("nan"),
                    int(values.size),
                ])

    skipped_path = out_dir / "e3e_plus_v3_skipped.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "error"])
        writer.writerows(skipped)

    print(f"\nCompleted: {len(rows)} validation pairs; skipped: {len(skipped)}")
    print(f"Summary: {summary_path}")
    print(f"Per-image: {per_image_path}")
    print(f"Skipped: {skipped_path}")
    print(f"Visual comparisons: {visual_dir}")
    print("Only CNN_Dataset/validation was accessed; test data was not read.")
    print("DeltaE is CIE76 Lab DeltaE, not spatial S-CIELAB.")


if __name__ == "__main__":
    main()
