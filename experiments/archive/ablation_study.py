"""
Ablation study for baseline and bright-region-aware adaptive atmospheric-light
estimation.

Place this file in the project root, beside dataset_config.py, settings.py,
and the dehazing/ folder. It evaluates the same paired images for each
configuration and saves per-image results plus a summary CSV.

Experiments:
A. baseline
B. full adaptive
C. adaptive_strength=0.0 (should reproduce baseline estimator)
D. no bright-region penalty (large_region_weight=1.0)
E. weaker adaptive blending (strength=0.25)
F. stronger adaptive blending (strength=0.75)

Important:
- Use a validation split for parameter selection.
- Reserve the final test split for confirmation.
- This script does not modify the core algorithm.
"""

from pathlib import Path
import csv
import math
import numpy as np

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablation_study"


def to_uint8(image):
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and np.nanmax(arr) <= 1.0:
        arr *= 255.0
    return np.clip(arr + 0.5, 0, 255).astype(np.uint8)


def calc_fsim(ref, pred):
    try:
        import torch
        import piq
    except ImportError:
        return float("nan")
    a = torch.from_numpy(ref.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(pred.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(piq.fsim(b, a, data_range=1.0).item())


def calc_delta_e(ref, pred):
    # Mean CIE76 Delta E in Lab; this is not true spatial S-CIELAB.
    try:
        from skimage.color import rgb2lab, deltaE_cie76
    except ImportError:
        return float("nan")
    lab_ref = rgb2lab(ref.astype(np.float32) / 255.0)
    lab_pred = rgb2lab(pred.astype(np.float32) / 255.0)
    return float(np.mean(deltaE_cie76(lab_ref, lab_pred)))


def calc_epi(ref_gray, pred_gray):
    try:
        from scipy.ndimage import sobel
    except ImportError:
        return float("nan")
    a = np.hypot(sobel(ref_gray, axis=1), sobel(ref_gray, axis=0)).ravel()
    b = np.hypot(sobel(pred_gray, axis=1), sobel(pred_gray, axis=0)).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def calc_edge_retention(ref_gray, pred_gray):
    try:
        from scipy.ndimage import sobel
    except ImportError:
        return float("nan")
    a = np.hypot(sobel(ref_gray, axis=1), sobel(ref_gray, axis=0))
    b = np.hypot(sobel(pred_gray, axis=1), sobel(pred_gray, axis=0))
    mean_ref = float(np.mean(a))
    return float(np.mean(b) / mean_ref) if mean_ref > 1e-8 else float("nan")


def get_metrics(pred, gt):
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from skimage.color import rgb2gray
    except ImportError as exc:
        raise SystemExit(
            "Missing scikit-image. Install with: pip install scikit-image"
        ) from exc

    pred8, gt8 = to_uint8(pred), to_uint8(gt)
    if pred8.shape != gt8.shape:
        raise ValueError(f"Image size mismatch: prediction {pred8.shape}, GT {gt8.shape}")

    psnr = float(peak_signal_noise_ratio(gt8, pred8, data_range=255))
    ssim = float(structural_similarity(gt8, pred8, channel_axis=2, data_range=255))
    fsim = calc_fsim(gt8, pred8)
    delta_e = calc_delta_e(gt8, pred8)
    gt_gray = rgb2gray(gt8.astype(np.float32) / 255.0)
    pred_gray = rgb2gray(pred8.astype(np.float32) / 255.0)
    epi = calc_epi(gt_gray, pred_gray)
    edge = calc_edge_retention(gt_gray, pred_gray)

    return {
        "PSNR_dB": psnr,
        "SSIM": ssim,
        "FSIM": fsim,
        "CIE76_DeltaE": delta_e,
        "EPI": epi,
        "Edge_Retention": edge,
    }


EXPERIMENTS = [
    {
        "id": "A_baseline",
        "description": "Original baseline estimator",
        "adaptive_airlight": False,
        "adaptive_strength": 0.5,
        "large_region_weight": 0.25,
    },
    {
        "id": "B_full_adaptive",
        "description": "Full proposed adaptive estimator",
        "adaptive_airlight": True,
        "adaptive_strength": 0.5,
        "large_region_weight": 0.25,
    },
    {
        "id": "C_strength_zero",
        "description": "Adaptive blend strength zero; should match baseline",
        "adaptive_airlight": True,
        "adaptive_strength": 0.0,
        "large_region_weight": 0.25,
    },
    {
        "id": "D_no_region_penalty",
        "description": "Candidate averaging without bright-region penalty",
        "adaptive_airlight": True,
        "adaptive_strength": 0.5,
        "large_region_weight": 1.0,
    },
    {
        "id": "E_strength_025",
        "description": "Adaptive blend strength 0.25",
        "adaptive_airlight": True,
        "adaptive_strength": 0.25,
        "large_region_weight": 0.25,
    },
    {
        "id": "F_strength_075",
        "description": "Adaptive blend strength 0.75",
        "adaptive_airlight": True,
        "adaptive_strength": 0.75,
        "large_region_weight": 0.25,
    },
]

METRIC_NAMES = [
    "PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"
]


def select_dataset():
    root = PROJECT_ROOT / "dataset"
    print("\nSelect dataset for ablation study:")
    print("1. CNN_Dataset (choose train/validation/test split)")
    print("2. I-HAZE")
    print("3. O-HAZE")
    print("4. Enter paired hazy and clear folder paths manually")
    choice = input("Choice [1-4]: ").strip()

    if choice == "1":
        dataset_root = root / "CNN_Dataset"
        if not dataset_root.is_dir():
            dataset_root = root / "CNN Dataset"
        split = input("Split [train/validation/test] (default validation): ").strip().lower() or "validation"
        if split not in {"train", "validation", "val", "test"}:
            raise SystemExit("Split must be train, validation/val, or test.")
        split_dir = dataset_root / ("validation" if split == "val" else split)
        hazy_dir = split_dir / "hazy"
        clear_dir = split_dir / "clear"
        label = f"CNN_Dataset_{split}"
    elif choice == "2":
        dataset_root = root / "I-HAZE"
        hazy_dir, clear_dir = dataset_root / "ihaze_hazy", dataset_root / "ihaze_gt"
        label = "I-HAZE"
    elif choice == "3":
        dataset_root = root / "O-HAZE"
        if not dataset_root.is_dir():
            dataset_root = root / "O-HAZY"
        hazy_dir, clear_dir = dataset_root / "ohaze_hazy", dataset_root / "ohaze_GT"
        label = "O-HAZE"
    elif choice == "4":
        hazy_dir = Path(input("Full path to hazy images folder: ").strip().strip('"')).expanduser()
        clear_dir = Path(input("Full path to clear/ground-truth folder: ").strip().strip('"')).expanduser()
        label = "custom_dataset"
    else:
        raise SystemExit("Invalid choice.")

    if not hazy_dir.is_dir():
        raise SystemExit(f"Hazy folder not found: {hazy_dir}")
    if not clear_dir.is_dir():
        raise SystemExit(f"Clear/ground-truth folder not found: {clear_dir}")

    print(f"\nHazy images : {hazy_dir}")
    print(f"Clear images: {clear_dir}")
    return hazy_dir, clear_dir, label


def finite_mean(values):
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def run_experiment(pairs, experiment, dataset_label, cfg_base):
    experiment_id = experiment["id"]
    out_dir = OUTPUT_ROOT / dataset_label / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_image_metrics.csv"

    cfg = DehazeConfig(
        downsample_factor=cfg_base["downsample_factor"],
        patch_size=cfg_base["patch_size"],
        omega=cfg_base["omega"],
        gamma=cfg_base["gamma"],
        t_min=cfg_base["t_min"],
        adaptive_airlight=experiment["adaptive_airlight"],
        adaptive_strength=experiment["adaptive_strength"],
        large_region_weight=experiment["large_region_weight"],
    )

    rows = []
    print(f"\n========== {experiment_id}: {experiment['description']} ==========")
    for i, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            result = dehaze_image(read_rgb(hazy_path), cfg)
            metrics = get_metrics(result["dehazed"], read_rgb(gt_path))
            rows.append([
                hazy_path.name, gt_path.name,
                *[metrics[name] for name in METRIC_NAMES]
            ])
        except Exception as exc:
            print(f"\n[SKIP] {hazy_path.name}: {exc}")
        if i % 10 == 0 or i == len(pairs):
            print(f"Progress: {i}/{len(pairs)}", end="\r", flush=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hazy_image", "ground_truth_image", *METRIC_NAMES])
        writer.writerows(rows)
        if rows:
            writer.writerow([])
            means = [finite_mean([r[j + 2] for r in rows]) for j in range(len(METRIC_NAMES))]
            writer.writerow(["AVERAGE", "", *means])

    means = {
        metric: finite_mean([row[i + 2] for row in rows])
        for i, metric in enumerate(METRIC_NAMES)
    }
    print(f"\nSuccessfully evaluated: {len(rows)}/{len(pairs)}")
    for metric in METRIC_NAMES:
        value = means[metric]
        print(f"Average {metric}: {value:.6f}" if np.isfinite(value) else f"Average {metric}: N/A")
    print(f"Saved: {csv_path}")
    return rows, means


def write_summary(dataset_label, summary_rows):
    out_dir = OUTPUT_ROOT / dataset_label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "description", "images_evaluated",
            *METRIC_NAMES
        ])
        writer.writerows(summary_rows)

    # Also create paired per-image deltas relative to baseline.
    baseline_csv = out_dir / "A_baseline" / "per_image_metrics.csv"
    if baseline_csv.exists():
        with baseline_csv.open("r", newline="", encoding="utf-8") as f:
            baseline_rows = list(csv.DictReader(f))
        baseline_map = {
            row["hazy_image"]: row for row in baseline_rows
            if row.get("hazy_image") and row["hazy_image"] != "AVERAGE"
        }
        delta_path = out_dir / "paired_deltas_vs_baseline.csv"
        with delta_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "hazy_image", *[f"delta_{m}" for m in METRIC_NAMES]])
            for exp in EXPERIMENTS[1:]:
                exp_path = out_dir / exp["id"] / "per_image_metrics.csv"
                if not exp_path.exists():
                    continue
                with exp_path.open("r", newline="", encoding="utf-8") as ef:
                    exp_rows = list(csv.DictReader(ef))
                for row in exp_rows:
                    base = baseline_map.get(row.get("hazy_image"))
                    if not base:
                        continue
                    deltas = []
                    for metric in METRIC_NAMES:
                        try:
                            deltas.append(float(row[metric]) - float(base[metric]))
                        except (ValueError, KeyError, TypeError):
                            deltas.append(float("nan"))
                    writer.writerow([exp["id"], row["hazy_image"], *deltas])

    print(f"\nAblation summary saved: {summary_path}")
    if baseline_csv.exists():
        print(f"Paired per-image deltas saved: {out_dir / 'paired_deltas_vs_baseline.csv'}")


def main():
    try:
        import skimage  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing scikit-image. Install with: pip install scikit-image") from exc

    hazy_dir, clear_dir, dataset_label = select_dataset()
    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    print(
        f"\nMatched pairs: {len(pairs)} | unmatched hazy: {len(unmatched_hazy)}"
        f" | unmatched clear: {len(unmatched_clear)}"
    )
    if not pairs:
        raise SystemExit("No safely matched image pairs found.")

    cfg_base = {
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "patch_size": PATCH_SIZE,
        "omega": OMEGA,
        "gamma": GAMMA,
        "t_min": T_MIN,
    }

    summary_rows = []
    for experiment in EXPERIMENTS:
        rows, means = run_experiment(pairs, experiment, dataset_label, cfg_base)
        summary_rows.append([
            experiment["id"], experiment["description"], len(rows),
            *[means[name] for name in METRIC_NAMES]
        ])

    write_summary(dataset_label, summary_rows)
    print("\nAblation study complete.")
    print("Remember: use validation data to select parameters; evaluate the final selected setting on held-out test data.")


if __name__ == "__main__":
    main()
