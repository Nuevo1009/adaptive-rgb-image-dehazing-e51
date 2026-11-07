"""Controlled E1 vs E2 validation ablation for spatially adaptive omega."""
from pathlib import Path
import csv
import numpy as np

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.color import rgb2gray, rgb2lab, deltaE_cie76
except ImportError as exc:
    raise SystemExit("Install scikit-image: pip install scikit-image") from exc

try:
    import torch
    import piq
except ImportError:
    torch = None
    piq = None

try:
    from scipy.ndimage import sobel
except ImportError:
    sobel = None


def to_uint8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    if x.size and float(np.max(x)) <= 1.0:
        x = x * 255.0
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def calculate_metrics(gt, pred):
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    delta = float(np.mean(deltaE_cie76(
        rgb2lab(gt.astype(np.float32) / 255.0),
        rgb2lab(pred.astype(np.float32) / 255.0)
    )))
    ref_gray = rgb2gray(gt.astype(np.float32) / 255.0)
    pred_gray = rgb2gray(pred.astype(np.float32) / 255.0)
    if sobel is None:
        epi = edge = float("nan")
    else:
        eg = np.hypot(sobel(ref_gray, axis=1), sobel(ref_gray, axis=0)).ravel()
        ep = np.hypot(sobel(pred_gray, axis=1), sobel(pred_gray, axis=0)).ravel()
        epi = float(np.corrcoef(eg, ep)[0, 1]) if np.std(eg) and np.std(ep) else float("nan")
        edge = float(np.mean(ep) / np.mean(eg)) if np.mean(eg) > 1e-8 else float("nan")
    if torch is None or piq is None:
        fsim = float("nan")
    else:
        a = torch.from_numpy(gt.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(pred.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            fsim = float(piq.fsim(b, a, data_range=1.0).item())
    return [psnr, ssim, fsim, delta, epi, edge]


def main():
    dataset_root = PROJECT_ROOT / "dataset" / "CNN_Dataset"
    hazy_dir = dataset_root / "validation" / "hazy"
    clear_dir = dataset_root / "validation" / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders not found:\n{hazy_dir}\n{clear_dir}")

    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    print(f"Matched validation pairs: {len(pairs)} | unmatched hazy: {len(unmatched_hazy)} | unmatched clear: {len(unmatched_clear)}")
    if not pairs:
        raise SystemExit("No paired validation images found.")

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
        guided_filter_radius=20,
        guided_filter_eps=1e-3,
    )
    configs = {
        "E1_fixed_omega_guided": DehazeConfig(**common, adaptive_omega=False),
        "E2_adaptive_omega_guided": DehazeConfig(
            **common,
            adaptive_omega=True,
            omega_min=0.80,
            omega_max=0.98,
            omega_haze_gain=0.08,
            omega_contrast_gain=0.08,
            omega_contrast_radius=7,
        ),
    }
    metric_names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    out_dir = PROJECT_ROOT / "outputs" / "adaptive_omega_ablation" / "CNN_Dataset_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hazy_path)
            gt = to_uint8(read_rgb(gt_path))
            row = {"filename": hazy_path.name, "ground_truth": gt_path.name}
            for label, cfg in configs.items():
                result = dehaze_image(hazy, cfg)
                pred = to_uint8(result["dehazed"])
                if pred.shape != gt.shape:
                    raise ValueError(f"shape mismatch: prediction {pred.shape}, GT {gt.shape}")
                for name, value in zip(metric_names, calculate_metrics(gt, pred)):
                    row[f"{label}_{name}"] = value
                diag = result.get("omega_diagnostics", {})
                for key, value in diag.items():
                    row[f"{label}_{key}"] = value
            rows.append(row)
        except Exception as exc:
            print(f"\n[SKIP] {hazy_path.name}: {exc}")
        if idx % 10 == 0 or idx == len(pairs):
            print(f"Progress: {idx}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No images were successfully evaluated.")

    per_image = out_dir / "adaptive_omega_per_image.csv"
    with per_image.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = out_dir / "adaptive_omega_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "metric", "mean", "valid_images", "delta_vs_E1"])
        means = {}
        for label in configs:
            means[label] = {}
            for name in metric_names:
                vals = np.asarray([r[f"{label}_{name}"] for r in rows], dtype=float)
                vals = vals[np.isfinite(vals)]
                means[label][name] = float(np.mean(vals)) if len(vals) else float("nan")
        for label in configs:
            for name in metric_names:
                delta = means[label][name] - means["E1_fixed_omega_guided"][name]
                writer.writerow([label, name, means[label][name], len(rows), delta])

    print(f"\nEvaluated validation images: {len(rows)}")
    print(f"Per-image CSV: {per_image}")
    print(f"Summary CSV: {summary}")
    print("\nMean metric comparison (E2 minus E1):")
    for name in metric_names:
        print(f"{name}: {means['E1_fixed_omega_guided'][name]:.6f} -> {means['E2_adaptive_omega_guided'][name]:.6f} | delta {means['E2_adaptive_omega_guided'][name] - means['E1_fixed_omega_guided'][name]:+.6f}")
    if torch is None or piq is None:
        print("NOTE: FSIM is NaN because torch/piq is not installed.")


if __name__ == "__main__":
    main()
