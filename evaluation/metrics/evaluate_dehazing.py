import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Evaluate baseline and adaptive-airlight dehazing on the same paired images.

Metrics: PSNR, SSIM, FSIM, CIE76 Delta E, EPI, Edge Retention.
Note: Delta E is CIE76 Lab Delta E, not true spatial S-CIELAB.
"""
from pathlib import Path
import csv
import numpy as np

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CNN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnn_guided_dehazing" / "dehazed"


def choose_ground_truth():
    root = PROJECT_ROOT / "dataset"
    print("\nChoose the clear ground-truth dataset matching the processed images:")
    print("1. CNN_Dataset (then choose train/test)")
    print("2. I-HAZE")
    print("3. O-HAZY")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        split = input("Evaluate which split? Enter train or test: ").strip().lower()
        if split not in {"train", "test"}:
            raise SystemExit("Invalid split. Enter train or test.")
        clear_dir = root / "CNN_Dataset" / split / "clear"
        label = f"CNN_Dataset_{split}"
    elif choice == "2":
        clear_dir, label = root / "I-HAZE" / "ihaze_gt", "I-HAZE"
    elif choice == "3":
        clear_dir, label = root / "O-HAZY" / "ohaze_GT", "O-HAZY"
    else:
        raise SystemExit("Invalid choice.")

    if not clear_dir.is_dir():
        raise SystemExit(f"Clear ground-truth folder not found: {clear_dir}")
    return clear_dir, label


def to_uint8(image):
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and arr.max() <= 1.0:
        arr *= 255.0
    return np.clip(arr + 0.5, 0, 255).astype(np.uint8)


def calc_fsim(ref, pred):
    try:
        import torch
        import piq
    except ImportError:
        return float("nan")
    a = torch.from_numpy(ref.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(pred.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(piq.fsim(b, a, data_range=1.0).item())


def calc_delta_e(ref, pred):
    """Mean CIE76 DeltaE in Lab; not a spatial S-CIELAB implementation."""
    try:
        from skimage.color import rgb2lab, deltaE_cie76
    except ImportError:
        return float("nan")
    a = rgb2lab(ref.astype(np.float32) / 255)
    b = rgb2lab(pred.astype(np.float32) / 255)
    return float(np.mean(deltaE_cie76(a, b)))


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
    mean_a = float(np.mean(a))
    return float(np.mean(b) / mean_a) if mean_a > 1e-8 else float("nan")


def _metric_means(rows):
    means = []
    for col in range(2, 8):
        vals = [float(r[col]) for r in rows if np.isfinite(float(r[col]))]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    return means


def evaluate_pairs(pairs, cfg, output_dir, apply_dehazing=True):
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from skimage.color import rgb2gray
    except ImportError as exc:
        raise SystemExit("Missing scikit-image. Install with: pip install scikit-image") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    rows = []

    for i, (input_path, gt_path) in enumerate(pairs, 1):
        try:
            source = read_rgb(input_path)
            pred = dehaze_image(source, cfg)["dehazed"] if apply_dehazing else source
            pred8, gt8 = to_uint8(pred), to_uint8(read_rgb(gt_path))

            if pred8.shape != gt8.shape:
                print(f"\n[SKIP size mismatch] {input_path.name}: prediction {pred8.shape}, GT {gt8.shape}")
                continue

            psnr = float(peak_signal_noise_ratio(gt8, pred8, data_range=255))
            ssim = float(structural_similarity(gt8, pred8, channel_axis=2, data_range=255))
            fsim = calc_fsim(gt8, pred8)
            delta_e = calc_delta_e(gt8, pred8)
            ref_gray = rgb2gray(gt8.astype(np.float32) / 255)
            pred_gray = rgb2gray(pred8.astype(np.float32) / 255)
            epi = calc_epi(ref_gray, pred_gray)
            edge = calc_edge_retention(ref_gray, pred_gray)
            rows.append([input_path.name, gt_path.name, psnr, ssim, fsim, delta_e, epi, edge])
        except Exception as exc:
            print(f"\n[SKIP] {input_path.name}: {exc}")

        if i % 10 == 0 or i == len(pairs):
            print(f"Progress: {i}/{len(pairs)}", end="\r", flush=True)

    out_file = output_dir / "evaluation_results.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction_image", "ground_truth_image", *names])
        writer.writerows(rows)
        if rows:
            writer.writerow([])
            writer.writerow(["AVERAGE", "", *_metric_means(rows)])

    print(f"\nSuccessfully evaluated: {len(rows)}/{len(pairs)}")
    if rows:
        for name, value in zip(names, _metric_means(rows)):
            print(f"Average {name}: {value:.6f}" if np.isfinite(value) else f"Average {name}: N/A")
    print(f"Saved: {out_file}")
    return rows


def compare_experiments(baseline_rows, adaptive_rows, output_dir):
    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]

    def means_by_image(rows):
        # Use common image names so both averages use the same evaluated cases.
        by_name = {row[0]: row for row in rows}
        return by_name

    base_map, adapt_map = means_by_image(baseline_rows), means_by_image(adaptive_rows)
    common = sorted(set(base_map) & set(adapt_map))
    if not common:
        raise SystemExit("No common successfully evaluated images to compare.")

    def mean_col(image_map, col):
        vals = [float(image_map[name][col]) for name in common if np.isfinite(float(image_map[name][col]))]
        return float(np.mean(vals)) if vals else float("nan")

    higher_is_better = {"PSNR_dB", "SSIM", "FSIM", "EPI", "Edge_Retention"}
    out_file = output_dir / "baseline_vs_adaptive.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== BASELINE vs ADAPTIVE (common images) ==========")
    print(f"Common images evaluated: {len(common)}")
    print(f"{'Metric':<20}{'Baseline':>14}{'Adaptive':>14}{'Adaptive - Base':>18}")

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "Adaptive", "Adaptive_minus_baseline", "Interpretation", "Common_images"])
        for idx, name in enumerate(metric_names, start=2):
            b, a = mean_col(base_map, idx), mean_col(adapt_map, idx)
            delta = a - b
            if not np.isfinite(b) or not np.isfinite(a):
                verdict = "Unavailable"
            elif name in higher_is_better:
                verdict = "Adaptive improved" if delta > 0 else "Baseline better" if delta < 0 else "Equal"
            else:
                verdict = "Adaptive improved" if delta < 0 else "Baseline better" if delta > 0 else "Equal"
            writer.writerow([name, b, a, delta, verdict, len(common)])
            print(f"{name:<20}{b:>14.6f}{a:>14.6f}{delta:>18.6f}")

    print(f"\nComparison saved: {out_file}")


def _get_direct_pairs(choice):
    root = PROJECT_ROOT / "dataset"
    if choice == "1":
        split = input("Choose split (train/test): ").strip().lower()
        if split not in {"train", "test"}:
            raise SystemExit("Invalid split. Enter train or test.")
        hazy_dir = root / "CNN_Dataset" / split / "hazy"
        clear_dir = root / "CNN_Dataset" / split / "clear"
        label = f"CNN_Dataset_{split}"
    elif choice == "2":
        hazy_dir, clear_dir, label = root / "I-HAZE" / "ihaze_hazy", root / "I-HAZE" / "ihaze_gt", "I-HAZE"
    elif choice == "3":
        hazy_dir, clear_dir, label = root / "O-HAZY" / "ohaze_hazy", root / "O-HAZY" / "ohaze_GT", "O-HAZY"
    else:
        raise SystemExit("Invalid dataset option.")

    if not hazy_dir.is_dir():
        raise SystemExit(f"Hazy folder not found: {hazy_dir}")
    if not clear_dir.is_dir():
        raise SystemExit(f"Clear folder not found: {clear_dir}")

    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    print(f"\nMatched pairs: {len(pairs)} | unmatched hazy: {len(unmatched_hazy)} | unmatched clear: {len(unmatched_clear)}")
    if not pairs:
        raise SystemExit("No image pairs matched safely. Check filenames.")
    return pairs, label


def main():
    print("\nImage Dehazing Evaluation")
    print("-------------------------")
    print("1. CNN_Dataset (run baseline and adaptive dehazing now)")
    print("2. I-HAZE (run baseline and adaptive dehazing now)")
    print("3. O-HAZY (run baseline and adaptive dehazing now)")
    print("4. Evaluate existing CNN-guided outputs (single-method evaluation)")
    choice = input("Select option (1-4): ").strip()

    if choice in {"1", "2", "3"}:
        pairs, label = _get_direct_pairs(choice)
        baseline_cfg = DehazeConfig(
            DOWNSAMPLE_FACTOR, PATCH_SIZE, OMEGA, GAMMA, T_MIN,
            adaptive_airlight=False,
        )
        adaptive_cfg = DehazeConfig(
            DOWNSAMPLE_FACTOR, PATCH_SIZE, OMEGA, GAMMA, T_MIN,
            adaptive_airlight=True,
        )

        experiment_root = PROJECT_ROOT / "outputs" / "evaluation_comparison" / label
        print("\n========== EXPERIMENT A: BASELINE ==========")
        baseline_rows = evaluate_pairs(pairs, baseline_cfg, experiment_root / "baseline")

        print("\n========== EXPERIMENT B: ADAPTIVE AIRLIGHT ==========")
        adaptive_rows = evaluate_pairs(pairs, adaptive_cfg, experiment_root / "adaptive")

        compare_experiments(baseline_rows, adaptive_rows, experiment_root)
        return

    if choice == "4":
        if not CNN_OUTPUT_DIR.is_dir():
            raise SystemExit(f"CNN-guided output folder not found: {CNN_OUTPUT_DIR}\nRun run_classified.py first.")
        clear_dir, label = choose_ground_truth()
        predictions = sorted(
            p for p in CNN_OUTPUT_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        gt_files = [p for p in clear_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        gt_map = {p.stem.lower(): p for p in gt_files}
        pairs, unmatched = [], []
        for pred_path in predictions:
            stem = pred_path.stem
            if stem.lower().endswith("_dehazed"):
                stem = stem[:-len("_dehazed")]
            gt_path = gt_map.get(stem.lower())
            if gt_path is None:
                unmatched.append(pred_path)
            else:
                pairs.append((pred_path, gt_path))

        print(f"\nOutputs found: {len(predictions)} | matched pairs: {len(pairs)} | unmatched outputs: {len(unmatched)}")
        if not pairs:
            raise SystemExit("No prediction/ground-truth pairs matched. Check dataset selection and filenames.")

        cfg = DehazeConfig(DOWNSAMPLE_FACTOR, PATCH_SIZE, OMEGA, GAMMA, T_MIN)
        out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "cnn_guided_dehazing" / label
        evaluate_pairs(pairs, cfg, out_dir, apply_dehazing=False)
        return

    raise SystemExit("Invalid option. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
