"""Paired validation ablation: adaptive airlight with/without guided transmission refinement."""
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def uint8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    if x.size and x.max() <= 1:
        x = x * 255.0
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def metrics(gt, pred):
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    delta = float(np.mean(deltaE_cie76(
        rgb2lab(gt.astype(np.float32) / 255.0),
        rgb2lab(pred.astype(np.float32) / 255.0)
    )))
    g = rgb2gray(gt.astype(np.float32) / 255.0)
    p = rgb2gray(pred.astype(np.float32) / 255.0)
    if sobel is None:
        epi = edge = float("nan")
    else:
        eg = np.hypot(sobel(g, axis=1), sobel(g, axis=0)).ravel()
        ep = np.hypot(sobel(p, axis=1), sobel(p, axis=0)).ravel()
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
    root = PROJECT_ROOT / "dataset" / "CNN_Dataset"
    hazy_dir = root / "validation" / "hazy"
    clear_dir = root / "validation" / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders not found:\n{hazy_dir}\n{clear_dir}")

    pairs, uh, uc = match_pairs(hazy_dir, clear_dir)
    print(f"Matched validation pairs: {len(pairs)} | unmatched hazy: {len(uh)} | unmatched clear: {len(uc)}")
    if not pairs:
        raise SystemExit("No paired validation images found.")

    base_cfg = dict(
        downsample_factor=DOWNSAMPLE_FACTOR,
        patch_size=PATCH_SIZE,
        omega=OMEGA,
        gamma=GAMMA,
        t_min=T_MIN,
        adaptive_airlight=True,
        adaptive_strength=0.75,
        large_region_weight=0.25,
    )
    configs = {
        "E0_adaptive_no_refinement": DehazeConfig(**base_cfg, refine_transmission=False),
        "E1_adaptive_guided_refinement": DehazeConfig(
            **base_cfg, refine_transmission=True,
            guided_filter_radius=20, guided_filter_eps=1e-3
        ),
    }
    names = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    out_dir = PROJECT_ROOT / "outputs" / "guided_transmission_ablation" / "CNN_Dataset_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hazy_path)
            gt = uint8(read_rgb(gt_path))
            row = {"filename": hazy_path.name, "ground_truth": gt_path.name}
            for label, cfg in configs.items():
                pred = uint8(dehaze_image(hazy, cfg)["dehazed"])
                if pred.shape != gt.shape:
                    raise ValueError(f"shape mismatch: prediction {pred.shape}, GT {gt.shape}")
                vals = metrics(gt, pred)
                for metric_name, value in zip(names, vals):
                    row[f"{label}_{metric_name}"] = value
            rows.append(row)
        except Exception as exc:
            print(f"\n[SKIP] {hazy_path.name}: {exc}")
        if idx % 10 == 0 or idx == len(pairs):
            print(f"Progress: {idx}/{len(pairs)}", end="\r", flush=True)

    if not rows:
        raise SystemExit("No images were successfully evaluated.")

    per_image = out_dir / "guided_transmission_per_image.csv"
    columns = list(rows[0].keys())
    with per_image.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = out_dir / "guided_transmission_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "metric", "mean", "valid_images"])
        for label in configs:
            for metric_name in names:
                key = f"{label}_{metric_name}"
                vals = np.asarray([r[key] for r in rows], dtype=float)
                vals = vals[np.isfinite(vals)]
                writer.writerow([label, metric_name, float(np.mean(vals)) if len(vals) else float("nan"), len(vals)])

    print(f"\nEvaluated images: {len(rows)}")
    print(f"Per-image results: {per_image}")
    print(f"Summary: {summary}")
    if torch is None or piq is None:
        print("NOTE: FSIM is NaN because torch/piq is not installed.")


if __name__ == "__main__":
    main()
