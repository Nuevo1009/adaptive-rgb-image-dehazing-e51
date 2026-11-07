import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""Evaluate dehazing outputs against paired clear ground-truth images.
Metrics: PSNR, SSIM, FSIM, S-CIELAB DeltaE, EPI, Edge Retention.
Note: DeltaE is implemented as CIE76 Lab DeltaE, not true spatial S-CIELAB.
"""
from pathlib import Path
import csv
import numpy as np
from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CNN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cnn_guided_dehazing" / "dehazed"


def choose_ground_truth():
    root = PROJECT_ROOT / "dataset"
    print("\nChoose the clear ground-truth dataset matching the processed images:")
    print("1. CNN_Dataset (then choose train/test)")
    print("2. I-HAZE")
    print("3. O-HAZY")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == "1":
        split = input("Evaluate which split? Enter train/validation/test: ").strip().lower()
        if split not in {"train", "validation", "test"}:
            raise SystemExit("Invalid split. Enter train, validation, or test.")
        clear_dir, label = root / "CNN_Dataset" / split / "clear", f"CNN_Dataset_{split}"
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


def evaluate_pairs(pairs, apply_dehazing, cfg, output_dir):
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from skimage.color import rgb2gray
    except ImportError as exc:
        raise SystemExit("Missing scikit-image. Install with: pip install scikit-image") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["PSNR_dB", "SSIM", "FSIM", "S_CIELAB_DeltaE", "EPI", "Edge_Retention"]
    rows = []
    for i, (input_path, gt_path) in enumerate(pairs, 1):
        try:
            pred = dehaze_image(read_rgb(input_path), cfg)["dehazed"] if apply_dehazing else read_rgb(input_path)
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
            means = []
            for col in range(2, 8):
                vals = [float(r[col]) for r in rows if np.isfinite(float(r[col]))]
                means.append(float(np.mean(vals)) if vals else float("nan"))
            writer.writerow([])
            writer.writerow(["AVERAGE", "", *means])

    if rows:
        print(f"\nEvaluated pairs: {len(rows)}")
        for col, name in enumerate(names, start=2):
            vals = [float(r[col]) for r in rows if np.isfinite(float(r[col]))]
            print(f"Average {name}: {np.mean(vals):.6f}" if vals else f"Average {name}: N/A")
    else:
        print("\nNo pairs were successfully evaluated.")
    print(f"Saved: {out_file}")


def main():
    cfg = DehazeConfig(
        DOWNSAMPLE_FACTOR, PATCH_SIZE, OMEGA, GAMMA, T_MIN,
        adaptive_airlight=True,
        adaptive_strength=0.75,
        large_region_weight=0.25,
        refine_transmission=True,
        guided_filter_radius=20,
        guided_filter_eps=1e-3,
    )
    print("\nImage Dehazing Evaluation")
    print("-------------------------")
    print("1. CNN_Dataset (run dehazing now)")
    print("2. I-HAZE (run dehazing now)")
    print("3. O-HAZY (run dehazing now)")
    print("4. Evaluate existing CNN-guided outputs")
    choice = input("Select option (1-4): ").strip()

    if choice == "4":
        if not CNN_OUTPUT_DIR.is_dir():
            raise SystemExit(f"CNN-guided output folder not found: {CNN_OUTPUT_DIR}\nRun run_classified.py first.")
        clear_dir, label = choose_ground_truth()
        predictions = sorted(p for p in CNN_OUTPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        gt_map = {p.stem.lower(): p for p in clear_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
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
        out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "cnn_guided_dehazing" / label
        evaluate_pairs(pairs, False, cfg, out_dir)
        return

    root = PROJECT_ROOT / "dataset"
    if choice == "1":
        split = input("Choose split (train/validation/test): ").strip().lower()
        if split not in {"train", "validation", "test"}:
            raise SystemExit("Invalid split. Enter train, validation, or test.")
        hazy_dir, clear_dir, label = root / "CNN_Dataset" / split / "hazy", root / "CNN_Dataset" / split / "clear", f"CNN_Dataset_{split}"
    elif choice == "2":
        hazy_dir, clear_dir, label = root / "I-HAZE" / "ihaze_hazy", root / "I-HAZE" / "ihaze_gt", "I-HAZE"
    elif choice == "3":
        hazy_dir, clear_dir, label = root / "O-HAZY" / "ohaze_hazy", root / "O-HAZY" / "ohaze_GT", "O-HAZY"
    else:
        raise SystemExit("Invalid option. Enter 1, 2, 3, or 4.")

    if not hazy_dir.is_dir():
        raise SystemExit(f"Hazy folder not found: {hazy_dir}")
    if not clear_dir.is_dir():
        raise SystemExit(f"Clear folder not found: {clear_dir}")
    pairs, uh, uc = match_pairs(hazy_dir, clear_dir)
    print(f"\nMatched pairs: {len(pairs)} | unmatched hazy: {len(uh)} | unmatched clear: {len(uc)}")
    if not pairs:
        raise SystemExit("No image pairs matched safely. Check filenames.")
    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / label
    evaluate_pairs(pairs, True, cfg, out_dir)


if __name__ == "__main__":
    main()
