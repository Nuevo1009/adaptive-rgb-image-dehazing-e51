"""
Evaluate the integrated CNN-guided E3-E selected pipeline against paired clear GT.
Metrics: PSNR, SSIM, FSIM (requires piq), CIE76 DeltaE, EPI, Edge Retention.
This evaluates the actual saved pipeline outputs, including CLEAR images that were bypassed.
"""

from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image

from dataset_config import PROJECT_ROOT, match_pairs

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cnn_guided_dehazing_e3e_selected"
LOG_PATH = OUTPUT_ROOT / "classification_results.csv"


def to_uint8(image):
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and np.nanmax(arr) <= 1.0:
        arr *= 255.0
    return np.clip(arr + 0.5, 0, 255).astype(np.uint8)


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def metric_fsim(gt, pred):
    try:
        import torch
        import piq
    except ImportError:
        return float("nan")
    a = torch.from_numpy(gt.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(pred.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(piq.fsim(b, a, data_range=1.0).item())


def metric_delta_e(gt, pred):
    try:
        from skimage.color import rgb2lab, deltaE_cie76
    except ImportError:
        return float("nan")
    lab_gt = rgb2lab(gt.astype(np.float32) / 255.0)
    lab_pred = rgb2lab(pred.astype(np.float32) / 255.0)
    return float(np.mean(deltaE_cie76(lab_gt, lab_pred)))


def metric_edge(gt_gray, pred_gray):
    from scipy.ndimage import sobel
    gx_gt, gy_gt = sobel(gt_gray, axis=1), sobel(gt_gray, axis=0)
    gx_pr, gy_pr = sobel(pred_gray, axis=1), sobel(pred_gray, axis=0)
    mag_gt = np.hypot(gx_gt, gy_gt)
    mag_pr = np.hypot(gx_pr, gy_pr)
    epi = float("nan")
    if np.std(mag_gt) > 0 and np.std(mag_pr) > 0:
        epi = float(np.corrcoef(mag_gt.ravel(), mag_pr.ravel())[0, 1])
    mean_gt = float(np.mean(mag_gt))
    retention = float(np.mean(mag_pr) / mean_gt) if mean_gt > 1e-8 else float("nan")
    return epi, retention


def main():
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from skimage.color import rgb2gray
    except ImportError as exc:
        raise SystemExit("Install scikit-image: pip install scikit-image") from exc
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install scipy: pip install scipy") from exc

    print("\nEvaluate saved CNN-guided E3-E selected outputs")
    print("1. validation")
    print("2. test")
    split = input("Choose split (validation/test): ").strip().lower()
    if split not in {"validation", "test"}:
        raise SystemExit("Please enter validation or test.")

    hazy_dir = PROJECT_ROOT / "dataset" / "CNN_Dataset" / split / "hazy"
    clear_dir = PROJECT_ROOT / "dataset" / "CNN_Dataset" / split / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Dataset folders not found:\n{hazy_dir}\n{clear_dir}")
    if not LOG_PATH.is_file():
        raise SystemExit(f"Classification log not found: {LOG_PATH}")

    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    print(f"Ground-truth pairs: {len(pairs)} | unmatched hazy: {len(unmatched_hazy)} | unmatched clear: {len(unmatched_clear)}")
    if not pairs:
        raise SystemExit("No hazy/clear pairs found.")

    # The log stores the exact saved output path for every processed image.
    log = {}
    with LOG_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = str(row.get("filename", "")).replace("\\", "/").lower()
            log[key] = row

    results = []
    missing_outputs = []
    skipped = []

    for index, (hazy_path, gt_path) in enumerate(pairs, 1):
        rel = hazy_path.relative_to(hazy_dir).as_posix()
        row = log.get(rel.lower())
        if row is None:
            # Fallback for logs that contain only the basename.
            row = log.get(hazy_path.name.lower())
        if row is None:
            missing_outputs.append(rel + " (no classification log row)")
            continue

        out_path = Path(row.get("output_path", ""))
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        if not out_path.is_file():
            missing_outputs.append(rel + f" (missing output: {out_path})")
            continue

        try:
            pred = to_uint8(read_rgb(out_path))
            gt = to_uint8(read_rgb(gt_path))
            if pred.shape != gt.shape:
                skipped.append((rel, f"shape mismatch pred={pred.shape}, gt={gt.shape}"))
                continue

            psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
            ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
            fsim = metric_fsim(gt, pred)
            delta_e = metric_delta_e(gt, pred)
            gt_gray = rgb2gray(gt.astype(np.float32) / 255.0)
            pred_gray = rgb2gray(pred.astype(np.float32) / 255.0)
            epi, edge_retention = metric_edge(gt_gray, pred_gray)

            results.append({
                "filename": rel,
                "classification": row.get("classification", ""),
                "action": row.get("action", ""),
                "PSNR_dB": psnr,
                "SSIM": ssim,
                "FSIM": fsim,
                "CIE76_DeltaE": delta_e,
                "EPI": epi,
                "Edge_Retention": edge_retention,
                "prediction_path": str(out_path),
                "ground_truth_path": str(gt_path),
            })
        except Exception as exc:
            skipped.append((rel, str(exc)))

        if index % 25 == 0 or index == len(pairs):
            print(f"Progress: {index}/{len(pairs)}", end="\r", flush=True)

    output_dir = PROJECT_ROOT / "outputs" / "evaluation" / "cnn_guided_e3e_selected" / f"CNN_Dataset_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image = output_dir / "metrics_per_image.csv"
    summary = output_dir / "metrics_summary.csv"
    skipped_file = output_dir / "metrics_skipped.csv"

    fields = ["filename", "classification", "action", "PSNR_dB", "SSIM", "FSIM",
              "CIE76_DeltaE", "EPI", "Edge_Retention", "prediction_path", "ground_truth_path"]
    with per_image.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "median", "std", "valid_count"])
        for name in metric_names:
            vals = np.asarray([r[name] for r in results if np.isfinite(r[name])], dtype=float)
            writer.writerow([name,
                             float(np.mean(vals)) if len(vals) else "N/A",
                             float(np.median(vals)) if len(vals) else "N/A",
                             float(np.std(vals)) if len(vals) else "N/A",
                             len(vals)])

    with skipped_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reason"])
        writer.writerows(skipped)
        writer.writerows([[x, "missing output or classification log row"] for x in missing_outputs])

    print(f"\n\nSuccessfully evaluated: {len(results)}")
    print(f"Missing outputs/log rows: {len(missing_outputs)} | skipped: {len(skipped)}")
    for name in metric_names:
        vals = [r[name] for r in results if np.isfinite(r[name])]
        print(f"{name}: {np.mean(vals):.6f}" if vals else f"{name}: N/A")
    if any(not np.isfinite(r["FSIM"]) for r in results):
        print("FSIM is N/A because PIQ/PyTorch may not be installed. Install with: pip install piq torch")
    print(f"\nSaved per-image metrics: {per_image}")
    print(f"Saved summary metrics:   {summary}")
    print(f"Saved skipped list:      {skipped_file}")


if __name__ == "__main__":
    main()
