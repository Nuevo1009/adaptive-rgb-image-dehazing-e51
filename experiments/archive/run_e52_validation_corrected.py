"""E5.2 Edge-Aware Confidence-Guided Adaptive RGB Chroma Preservation.

Validation-only experiment. Keeps the paper's dehazing equations and E3-E
preset unchanged; E5.1 applies a spatially varying RGB-chroma blend afterward; E5.2 reduces
that blend near strong image edges to protect edge structure.
No ground-truth data is used to calculate the adaptive map.
"""
from pathlib import Path
import csv
import math
import cv2
import numpy as np

from dataset_config import PROJECT_ROOT, match_pairs
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, validate_rgb

SPLIT = "validation"  # Deliberately fixed: never change to test for tuning.
HAZY_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT / "hazy"
CLEAR_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT / "clear"
OUT = PROJECT_ROOT / "outputs" / "E5_2_edge_aware_validation"
CHECKPOINT_EVERY = 20

# E3-E is frozen as the reference configuration currently selected in core.py.
E3E_CONFIG = DehazeConfig.e3e_selected(
    guided_filter_radius=20,
    guided_filter_eps=1e-3,
)

# E5.1 adaptive-map parameters. These are initial hypotheses to evaluate,
# not claimed optimal values. Alpha is bounded to avoid extreme colour changes.
BASE_ALPHA = 0.40
ALPHA_GAIN = 0.20
ALPHA_MIN = 0.20
ALPHA_MAX = 0.60
DARK_LUMA = 0.16
BRIGHT_LUMA = 0.88


def rgb_chroma(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y = np.sum(rgb * weights, axis=2, keepdims=True)
    return rgb - y, y[..., 0]


def adaptive_alpha_map(input_rgb, e3e_rgb):
    """Return per-pixel alpha using only hazy input and E3-E estimate.

    Higher chroma disagreement raises input-chroma preservation in reliable
    mid-tones. Dark and near-saturated highlights suppress the adaptation to
    avoid importing haze tint into shadows or changing bright regions.
    A small box filter smooths the map spatially.
    """
    src = validate_rgb(input_rgb)
    out = validate_rgb(e3e_rgb)
    c_src, y_src = rgb_chroma(src)
    c_out, _ = rgb_chroma(out)

    disagreement = np.mean(np.abs(c_src - c_out), axis=2)
    # Normalize disagreement conservatively; 0.15 is a practical RGB scale.
    disagreement_conf = np.clip(disagreement / 0.15, 0.0, 1.0)

    # Mid-tone reliability is low in deep shadows and near-clipped highlights.
    dark_gate = np.clip((y_src - DARK_LUMA) / 0.18, 0.0, 1.0)
    bright_gate = np.clip((BRIGHT_LUMA - y_src) / 0.12, 0.0, 1.0)
    luminance_gate = dark_gate * bright_gate

    # Chroma-rich regions get a modestly higher confidence, but this is not
    # a semantic vegetation detector and must not be interpreted as one.
    saturation = np.linalg.norm(c_src, axis=2)
    saturation_gate = np.clip(saturation / 0.20, 0.0, 1.0)
    confidence = disagreement_conf * luminance_gate * (0.65 + 0.35 * saturation_gate)

    alpha = BASE_ALPHA + ALPHA_GAIN * (confidence - 0.35)
    alpha = np.clip(alpha, ALPHA_MIN, ALPHA_MAX).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return np.clip(alpha, ALPHA_MIN, ALPHA_MAX)


def apply_adaptive_chroma(input_rgb, e3e_rgb):
    src = validate_rgb(input_rgb)
    out = validate_rgb(e3e_rgb)
    alpha = adaptive_alpha_map(src, out)[..., None]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y_out = np.sum(out * weights, axis=2, keepdims=True)
    c_src, _ = rgb_chroma(src)
    c_out, _ = rgb_chroma(out)
    c_new = (1.0 - alpha) * c_out + alpha * c_src
    return np.clip(y_out + c_new, 0.0, 1.0).astype(np.float32), alpha[..., 0]


def edge_aware_alpha_map(input_rgb, e3e_rgb, alpha51):
    """Reduce chroma blending near strong edges using input-derived edges.

    No ground truth is used. The edge map is based on grayscale input gradients,
    robustly normalized by the 95th percentile and softly smoothed.
    """
    src = validate_rgb(input_rgb)
    gray = cv2.cvtColor(src.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 95))
    if scale <= 1e-8:
        edge = np.zeros_like(magnitude, dtype=np.float32)
    else:
        edge = np.clip(magnitude / scale, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.2, sigmaY=1.2)

    # At strong edges, reduce alpha by at most 25%; retain E5.1 behaviour
    # in flat regions. This is a conservative first E5.2 hypothesis.
    edge_protection = 0.25
    alpha52 = alpha51 * (1.0 - edge_protection * edge)
    return np.clip(alpha52, ALPHA_MIN, ALPHA_MAX), edge


def apply_e52(input_rgb, e3e_rgb, alpha51):
    """Apply edge-reduced chroma preservation ONCE to the E3-E reference.

    Applying another chroma blend to an already blended E5.1 image would
    increase the total input-chroma contribution, not reduce it. Therefore
    E5.2 is computed directly from E3-E using the edge-adjusted alpha map.
    """
    src = validate_rgb(input_rgb)
    base = validate_rgb(e3e_rgb)
    alpha52, edge_map = edge_aware_alpha_map(src, base, alpha51)
    alpha = alpha52[..., None]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y_base = np.sum(base * weights, axis=2, keepdims=True)
    c_src, _ = rgb_chroma(src)
    c_base, _ = rgb_chroma(base)
    c_new = (1.0 - alpha) * c_base + alpha * c_src
    result = np.clip(y_base + c_new, 0.0, 1.0).astype(np.float32)
    return result, alpha52, edge_map


def as_u8(x):
    return np.clip(validate_rgb(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def metrics(gt, pred):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.color import rgb2lab, deltaE_cie76, rgb2gray
    from scipy.ndimage import sobel

    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    lab_gt = rgb2lab(gt.astype(np.float32) / 255.0)
    lab_pred = rgb2lab(pred.astype(np.float32) / 255.0)
    delta_e = float(np.mean(deltaE_cie76(lab_gt, lab_pred)))
    g = rgb2gray(gt.astype(np.float32) / 255.0)
    p = rgb2gray(pred.astype(np.float32) / 255.0)
    gx = np.hypot(sobel(g, axis=1), sobel(g, axis=0)).ravel()
    px = np.hypot(sobel(p, axis=1), sobel(p, axis=0)).ravel()
    epi = float(np.corrcoef(gx, px)[0, 1]) if np.std(gx) > 0 and np.std(px) > 0 else float("nan")
    edge = float(np.mean(np.hypot(sobel(p.reshape(gt.shape[:2]), axis=1), sobel(p.reshape(gt.shape[:2]), axis=0))) /
                  (np.mean(np.hypot(sobel(g.reshape(gt.shape[:2]), axis=1), sobel(g.reshape(gt.shape[:2]), axis=0))) + 1e-12))
    try:
        import torch
        import piq
        a = torch.from_numpy(gt.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        b = torch.from_numpy(pred.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            fsim = float(piq.fsim(b, a, data_range=1.0).item())
    except Exception:
        fsim = float("nan")
    return psnr, ssim, fsim, delta_e, epi, edge


def save_worst_case_panel(out_dir, name, hazy, gt, e3e, e51, e52):
    """Save a five-column visual panel for the four pre-identified cases."""
    images = [as_u8(x) for x in (hazy, gt, e3e, e51, e52)]
    labels = ["Hazy input", "Ground truth", "E3-E", "E5.1", "E5.2"]
    h, w = images[0].shape[:2]
    cell_w = min(360, w)
    cell_h = max(1, int(h * cell_w / w))
    tiles = []
    for label, img in zip(labels, images):
        tile = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (cell_w, cell_h))
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    panel = cv2.hconcat(tiles)
    cv2.imwrite(str(out_dir / f"{Path(name).stem}_worst_case_comparison.jpg"), panel)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    for d in (HAZY_DIR, CLEAR_DIR):
        if not d.is_dir():
            raise SystemExit(f"Dataset folder not found: {d}")
    pairs, uh, uc = match_pairs(HAZY_DIR, CLEAR_DIR)
    if not pairs:
        raise SystemExit("No paired validation images found. Check filenames and dataset paths.")
    OUT.mkdir(parents=True, exist_ok=True)
    e3e_dir = OUT / "E3E_REFERENCE"
    e51_dir = OUT / "E5_1_REFERENCE"
    e52_dir = OUT / "E5_2_EDGE_AWARE_RGB"
    e3e_dir.mkdir(exist_ok=True)
    e51_dir.mkdir(exist_ok=True)
    e52_dir.mkdir(exist_ok=True)
    per_image_path = OUT / "E3E_E5_1_E5_2_per_image.csv"
    summary_path = OUT / "E3E_E5_1_E5_2_summary.csv"
    header = ["image",
              "PSNR_E3E", "PSNR_E5_1", "PSNR_E5_2",
              "SSIM_E3E", "SSIM_E5_1", "SSIM_E5_2",
              "FSIM_E3E", "FSIM_E5_1", "FSIM_E5_2",
              "CIE76_DeltaE_E3E", "CIE76_DeltaE_E5_1", "CIE76_DeltaE_E5_2",
              "EPI_E3E", "EPI_E5_1", "EPI_E5_2",
              "Edge_E3E", "Edge_E5_1", "Edge_E5_2",
              "alpha51_mean", "alpha52_mean", "alpha52_min", "alpha52_max", "edge_mean"]
    rows = []
    skipped = []
    print(f"Matched validation pairs: {len(pairs)}")
    print("Split: validation only; test split is not accessed.")
    print("E3-E preset: DehazeConfig.e3e_selected()")
    print("E5.1 and E5.2 are compared against the same E3-E output.")
    print("Starting E3-E / E5.1 / E5.2 metrics...")
    for i, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hazy_path)
            gt = as_u8(read_rgb(gt_path))
            if hazy.shape != gt.shape:
                skipped.append([hazy_path.name, "dimension mismatch"])
                continue
            # One E3-E inference. E5.1 is an adaptive RGB chroma post-process.
            e3e = dehaze_image(hazy, E3E_CONFIG)["dehazed"]
            e51, alpha51 = apply_adaptive_chroma(hazy, e3e)
            e52, alpha52, edge_map = apply_e52(hazy, e3e, alpha51)
            e3e8, e518, e528 = as_u8(e3e), as_u8(e51), as_u8(e52)
            m0, m1, m2 = metrics(gt, e3e8), metrics(gt, e518), metrics(gt, e528)
            rows.append([hazy_path.name,
                         m0[0], m1[0], m2[0], m0[1], m1[1], m2[1],
                         m0[2], m1[2], m2[2], m0[3], m1[3], m2[3],
                         m0[4], m1[4], m2[4], m0[5], m1[5], m2[5],
                         float(np.mean(alpha51)), float(np.mean(alpha52)),
                         float(np.min(alpha52)), float(np.max(alpha52)), float(np.mean(edge_map))])
            # Save all three outputs for visual review.
            cv2.imwrite(str(e3e_dir / f"{hazy_path.stem}_dehazed.png"), cv2.cvtColor(e3e8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(e51_dir / f"{hazy_path.stem}_dehazed.png"), cv2.cvtColor(e518, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(e52_dir / f"{hazy_path.stem}_dehazed.png"), cv2.cvtColor(e528, cv2.COLOR_RGB2BGR))
            if hazy_path.stem in {"1392", "922", "4410", "1841"}:
                panel_dir = OUT / "WORST_CASE_VISUALS"
                panel_dir.mkdir(exist_ok=True)
                save_worst_case_panel(panel_dir, hazy_path.name, hazy, gt, e3e, e51, e52)
        except Exception as exc:
            skipped.append([hazy_path.name, repr(exc)])
        if i % CHECKPOINT_EVERY == 0 or i == len(pairs):
            write_csv(per_image_path, header, rows)
            write_csv(OUT / "skipped.csv", ["image", "reason"], skipped)
            print(f"Progress: {i}/{len(pairs)} | successful: {len(rows)} | skipped: {len(skipped)}", flush=True)

    metric_names = ["PSNR", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]
    # Per-image columns are grouped by metric: E3-E, E5.1, E5.2.
    metric_starts = [1, 4, 7, 10, 13, 16]
    summary = []
    for name, first in zip(metric_names, metric_starts):
        values = []
        for offset in range(3):
            arr = np.array([float(r[first + offset]) for r in rows], dtype=float)
            values.append(float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"))
        summary.append([name, values[0], values[1], values[2],
                        values[1] - values[0], values[2] - values[0], len(rows)])
    write_csv(summary_path,
              ["metric", "E3E_mean", "E5_1_mean", "E5_2_mean",
               "change_E5_1_vs_E3E", "change_E5_2_vs_E3E", "n"],
              summary)
    print("\\n=== E3-E vs E5.1 vs E5.2 validation summary ===")
    for r in summary:
        print(f"{r[0]}: E3-E={r[1]:.6f}, E5.1={r[2]:.6f}, "
              f"E5.2={r[3]:.6f}, delta_E5.2={r[5]:+.6f}, n={r[6]}")
    print(f"Per-image CSV: {per_image_path}")
    print(f"Summary CSV:   {summary_path}")
    print(f"E3-E images:   {e3e_dir}")
    print(f"E5.1 images:   {e51_dir}")
    print(f"E5.2 images:   {e52_dir}")
    print("Inspect the per-image CSV for 1392.jpg, 922.jpg, 4410.jpg and 1841.jpg.")
    if uh or uc:
        print(f"Unmatched hazy={len(uh)}, unmatched clear={len(uc)}")


if __name__ == "__main__":
    main()
