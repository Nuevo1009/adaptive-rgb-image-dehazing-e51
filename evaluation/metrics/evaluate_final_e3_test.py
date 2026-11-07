import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Final locked evaluation: original baseline vs E1 vs E3-E (alpha=0.40).
Run from the project root. Uses CNN_Dataset/test only; no parameter tuning.
"""
from pathlib import Path
import csv
import numpy as np

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, save_rgb
from evaluation.metrics.evaluate_extended import to_uint8, calc_fsim, calc_delta_e, calc_epi, calc_edge_retention


def make_configs():
    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR,
        patch_size=PATCH_SIZE,
        omega=OMEGA,
        gamma=GAMMA,
        t_min=T_MIN,
    )
    # Original algorithm: original airlight estimator, no guided filter, no colour correction.
    baseline = DehazeConfig(**common)
    # E1 reference: adaptive airlight + guided transmission; no colour correction.
    e1 = DehazeConfig(
        **common, adaptive_airlight=True, adaptive_strength=0.75,
        large_region_weight=0.25, refine_transmission=True,
        guided_filter_radius=20, guided_filter_eps=1e-3,
        color_preserve_strength=0.0,
    )
    # Frozen E3-E candidate: exactly E1 plus alpha=0.40.
    e3e = DehazeConfig(
        **common, adaptive_airlight=True, adaptive_strength=0.75,
        large_region_weight=0.25, refine_transmission=True,
        guided_filter_radius=20, guided_filter_eps=1e-3,
        color_preserve_strength=0.40,
    )
    return {"Baseline": baseline, "E1": e1, "E3_E_alpha_0.40": e3e}


def metrics(gt, pred):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.color import rgb2gray
    gt8, pred8 = to_uint8(gt), to_uint8(pred)
    if gt8.shape != pred8.shape:
        raise ValueError(f"Image size mismatch: prediction {pred8.shape}, GT {gt8.shape}")
    psnr = float(peak_signal_noise_ratio(gt8, pred8, data_range=255))
    ssim = float(structural_similarity(gt8, pred8, channel_axis=2, data_range=255))
    ref_gray = rgb2gray(gt8.astype(np.float32) / 255.0)
    pred_gray = rgb2gray(pred8.astype(np.float32) / 255.0)
    return [psnr, ssim, calc_fsim(gt8, pred8), calc_delta_e(gt8, pred8),
            calc_epi(ref_gray, pred_gray), calc_edge_retention(ref_gray, pred_gray)]


def main():
    hazy_dir = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "test" / "hazy"
    clear_dir = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "test" / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Test folders not found. Expected:\n{hazy_dir}\n{clear_dir}")

    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No paired test images found. Check filenames and dataset_config.match_pairs().")

    out_dir = PROJECT_ROOT / "outputs" / "final_e3_test_evaluation" / "CNN_Dataset_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "e3_e_dehazed"
    configs = make_configs()
    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    rows = []
    print(f"Matched test pairs: {len(pairs)}")
    print(f"Unmatched hazy: {len(unmatched_hazy)} | unmatched clear: {len(unmatched_clear)}")
    print("Evaluating locked configurations; no tuning is performed.")

    for index, (hazy_path, gt_path) in enumerate(pairs, 1):
        hazy = read_rgb(hazy_path)
        gt = read_rgb(gt_path)
        row = {"hazy_image": hazy_path.name, "ground_truth": gt_path.name}
        try:
            for label, cfg in configs.items():
                pred = dehaze_image(hazy, cfg)["dehazed"]
                values = metrics(gt, pred)
                for name, value in zip(metric_names, values):
                    row[f"{label}_{name}"] = value
                if label == "E3_E_alpha_0.40":
                    save_rgb(save_dir / f"{hazy_path.stem}_dehazed.png", pred)
            rows.append(row)
        except Exception as exc:
            print(f"\n[SKIP] {hazy_path.name}: {exc}")
        if index % 10 == 0 or index == len(pairs):
            print(f"Progress: {index}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No test pairs were successfully evaluated.")

    per_image_path = out_dir / "final_test_per_image.csv"
    fields = list(rows[0].keys())
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "final_test_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Configuration", "Metric", "Mean", "Valid_images"])
        for label in configs:
            for metric_name in metric_names:
                key = f"{label}_{metric_name}"
                vals = np.asarray([r[key] for r in rows], dtype=float)
                valid = vals[np.isfinite(vals)]
                writer.writerow([label, metric_name,
                                 float(np.mean(valid)) if len(valid) else float("nan"), len(valid)])

    print(f"\nSuccessfully evaluated: {len(rows)} / {len(pairs)}")
    print(f"Per-image results: {per_image_path}")
    print(f"Summary: {summary_path}")
    print(f"E3-E output images: {save_dir}")
    print("Note: CIE76 DeltaE is a mean Lab colour difference, not spatial S-CIELAB.")


if __name__ == "__main__":
    main()
