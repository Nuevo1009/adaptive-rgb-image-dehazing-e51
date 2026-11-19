"""E5.1 Confidence-Guided Adaptive RGB Chroma Preservation.

Validation-only experiment. Keeps the paper's dehazing equations and E3-E
preset unchanged; E5.1 applies a spatially varying RGB-chroma blend afterward.
No ground-truth data is used to calculate the adaptive map.
"""
from pathlib import Path
import csv
import math
import cv2
import numpy as np


from dataset_config import PROJECT_ROOT, match_pairs
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, validate_rgb

#SPLIT = ""  # Deliberately fixed: never change to test for tuning.
HAZY_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "test" / "hazy"
CLEAR_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / "test" / "clear"
OUT = PROJECT_ROOT / "outputs" / "E5_1_confidence_guided_validation"
CHECKPOINT_EVERY = 20


# Extended perceptual metrics (MS-SSIM, BRISQUE, NIQE, PIQE).
_EXTENDED_METRIC_OBJECTS = None
_EXTENDED_METRIC_DEVICE = None


def initialize_extended_metrics(device=None):
    """Initialize pyiqa metrics once. Inputs are RGB tensors in [0, 1]."""
    global _EXTENDED_METRIC_OBJECTS, _EXTENDED_METRIC_DEVICE
    if _EXTENDED_METRIC_OBJECTS is not None:
        return
    try:
        import torch
        import pyiqa
    except ImportError as exc:
        raise RuntimeError(
            "Install the additional metric dependency in the project venv: "
            "python -m pip install pyiqa"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _EXTENDED_METRIC_DEVICE = torch.device(device)
    names = ("ms_ssim", "brisque", "niqe", "piqe")
    _EXTENDED_METRIC_OBJECTS = {
        name: pyiqa.create_metric(name, device=_EXTENDED_METRIC_DEVICE, as_loss=False)
        for name in names
    }


def extended_metrics(gt, pred):
    """Return MS-SSIM, BRISQUE, NIQE and PIQE for a uint8 RGB prediction."""
    import torch
    if _EXTENDED_METRIC_OBJECTS is None:
        initialize_extended_metrics()

    def to_tensor(image):
        return torch.from_numpy(
            np.ascontiguousarray(image.astype(np.float32) / 255.0)
        ).permute(2, 0, 1).unsqueeze(0).to(_EXTENDED_METRIC_DEVICE)

    gt_t, pred_t = to_tensor(gt), to_tensor(pred)
    values = []
    with torch.no_grad():
        for name in ("ms_ssim", "brisque", "niqe", "piqe"):
            try:
                score = (_EXTENDED_METRIC_OBJECTS[name](pred_t, gt_t)
                         if name == "ms_ssim"
                         else _EXTENDED_METRIC_OBJECTS[name](pred_t))
                values.append(float(score.detach().reshape(-1)[0].cpu().item()))
            except Exception as exc:
                print(f"WARNING: {name} failed: {exc}")
                values.append(float("nan"))
    return tuple(values)


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


def as_u8(x):
    return np.clip(validate_rgb(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)



# ---- Zhang-Wandell S-CIELAB implementation (embedded; no extra module import) ----
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2xyz, xyz2lab, deltaE_cie76

# Zhang & Wandell / Poirson & Wandell XYZ(1931 2-degree) -> opponent matrix.
XYZ_TO_OPPONENT = np.array([
    [278.7336, 721.8031, -106.5520],
    [-448.7736, 289.8056, 77.1569],
    [85.9513, -589.9859, 501.1089],
], dtype=np.float64) / 1000.0
OPPONENT_TO_XYZ = np.linalg.inv(XYZ_TO_OPPONENT)

# Published normalized Gaussian-mixture coefficients from separableFilters.m.
# Each row contains (visual-angle halfwidth in degrees, mixture weight).
FILTERS = (
    ((0.0500, 1.003270), (0.2250, 0.114416), (7.0000, -0.117686)),  # luminance
    ((0.0685, 0.616725), (0.8260, 0.383275)),                        # red-green
    ((0.0920, 0.567885), (0.6451, 0.432115)),                        # blue-yellow
)

DEFAULT_SAMPLES_PER_DEGREE = 60.0


def _as_srgb01(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input images must have shape HxWx3 in RGB order.")
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input image contains NaN or infinite values.")
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _filter_opponent_channel(channel, mixture, samples_per_degree):
    """Apply original S-CIELAB weighted Gaussian-mixture spatial filter."""
    result = np.zeros(channel.shape, dtype=np.float64)
    for halfwidth_deg, weight in mixture:
        sigma_px = halfwidth_deg * samples_per_degree
        # Published Gaussian form is exp(-r^2 / sigma^2), i.e. scipy's
        # Gaussian sigma parameter is sigma_px/sqrt(2).
        sigma = sigma_px / np.sqrt(2.0)
        if sigma < 1e-8:
            component = channel
        else:
            component = gaussian_filter(
                channel, sigma=sigma, mode="reflect", truncate=4.0
            )
        result += weight * component
    return result


def scielab_map(reference_rgb, test_rgb,
                samples_per_degree=DEFAULT_SAMPLES_PER_DEGREE):
    """
    Return pixelwise S-CIELAB DeltaE*ab map for matching RGB image pairs.

    samples_per_degree must represent image sampling density in pixels per
    degree of visual angle for the intended display/viewing condition.
    """
    ref = _as_srgb01(reference_rgb)
    tst = _as_srgb01(test_rgb)
    if ref.shape != tst.shape:
        raise ValueError("Reference and test images must have identical dimensions.")
    if not np.isfinite(samples_per_degree) or samples_per_degree <= 0:
        raise ValueError("samples_per_degree must be a positive finite number.")

    # Convert sRGB to linear CIE XYZ (D65, 2-degree observer).
    ref_xyz = rgb2xyz(ref)
    tst_xyz = rgb2xyz(tst)

    # Linear XYZ -> opponent channels, filter each channel independently.
    ref_opp = ref_xyz @ XYZ_TO_OPPONENT.T
    tst_opp = tst_xyz @ XYZ_TO_OPPONENT.T
    ref_filtered = np.empty_like(ref_opp)
    tst_filtered = np.empty_like(tst_opp)

    for channel, mixture in enumerate(FILTERS):
        ref_filtered[..., channel] = _filter_opponent_channel(
            ref_opp[..., channel], mixture, float(samples_per_degree)
        )
        tst_filtered[..., channel] = _filter_opponent_channel(
            tst_opp[..., channel], mixture, float(samples_per_degree)
        )

    # Filtered opponent representation -> XYZ -> CIELAB (D65).
    ref_filtered_xyz = ref_filtered @ OPPONENT_TO_XYZ.T
    tst_filtered_xyz = tst_filtered @ OPPONENT_TO_XYZ.T
    ref_lab = xyz2lab(ref_filtered_xyz, illuminant="D65", observer="2")
    tst_lab = xyz2lab(tst_filtered_xyz, illuminant="D65", observer="2")

    return deltaE_cie76(ref_lab, tst_lab).astype(np.float32)


def scielab_mean_delta_e(reference_rgb, test_rgb,
                         samples_per_degree=DEFAULT_SAMPLES_PER_DEGREE):
    """Mean pixelwise S-CIELAB DeltaE*ab; lower values indicate less difference."""
    difference = scielab_map(reference_rgb, test_rgb, samples_per_degree)
    return float(np.mean(difference, dtype=np.float64))


def metrics(gt, pred):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.color import rgb2gray
    from scipy.ndimage import sobel

    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=255))
    # Zhang-Wandell pattern-colour separable S-CIELAB.
    # samples/degree is configurable in scielab_metric_zhang_wandell.py.
    delta_e = float(scielab_mean_delta_e(gt, pred))
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
    e51_dir = OUT / "E5_1_ADAPTIVE_RGB"
    e3e_dir.mkdir(exist_ok=True)
    e51_dir.mkdir(exist_ok=True)
    per_image_path = OUT / "E3E_vs_E5_1_per_image.csv"
    summary_path = OUT / "E3E_vs_E5_1_summary.csv"
    header = ["image", "PSNR_E3E", "PSNR_E5_1", "delta_PSNR", "SSIM_E3E", "SSIM_E5_1", "delta_SSIM",
              "FSIM_E3E", "FSIM_E5_1", "delta_FSIM", "S_CIELAB_E3E", "S_CIELAB_E5_1", "delta_S_CIELAB",
              "EPI_E3E", "EPI_E5_1", "delta_EPI", "Edge_E3E", "Edge_E5_1", "delta_Edge",
              "MS_SSIM_E3E", "MS_SSIM_E5_1", "delta_MS_SSIM",
              "BRISQUE_E3E", "BRISQUE_E5_1", "delta_BRISQUE",
              "NIQE_E3E", "NIQE_E5_1", "delta_NIQE",
              "PIQE_E3E", "PIQE_E5_1", "delta_PIQE",
              "alpha_mean", "alpha_min", "alpha_max"]
    initialize_extended_metrics()
    rows = []
    skipped = []
    print(f"Matched validation pairs: {len(pairs)}")
    print("Split: validation only; test split is not accessed.")
    print("E3-E preset: DehazeConfig.e3e_selected()")
    print("Starting E3-E and E5.1 metrics...")
    for i, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hazy_path)
            gt = as_u8(read_rgb(gt_path))
            if hazy.shape != gt.shape:
                skipped.append([hazy_path.name, "dimension mismatch"])
                continue
            # One E3-E inference. E5.1 is an adaptive RGB chroma post-process.
            e3e = dehaze_image(hazy, E3E_CONFIG)["dehazed"]
            e51, alpha_map = apply_adaptive_chroma(hazy, e3e)
            e3e8, e518 = as_u8(e3e), as_u8(e51)
            m0, m1 = metrics(gt, e3e8), metrics(gt, e518)
            x0, x1 = extended_metrics(gt, e3e8), extended_metrics(gt, e518)
            rows.append([
                hazy_path.name,
                m0[0], m1[0], m1[0]-m0[0],
                m0[1], m1[1], m1[1]-m0[1],
                m0[2], m1[2], m1[2]-m0[2],
                m0[3], m1[3], m1[3]-m0[3],
                m0[4], m1[4], m1[4]-m0[4],
                m0[5], m1[5], m1[5]-m0[5],
                x0[0], x1[0], x1[0]-x0[0],
                x0[1], x1[1], x1[1]-x0[1],
                x0[2], x1[2], x1[2]-x0[2],
                x0[3], x1[3], x1[3]-x0[3],
                float(np.mean(alpha_map)), float(np.min(alpha_map)), float(np.max(alpha_map))
            ])
            # Save both outputs so they can be inspected and reused.
            cv2.imwrite(str(e3e_dir / f"{hazy_path.stem}_dehazed.png"), cv2.cvtColor(e3e8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(e51_dir / f"{hazy_path.stem}_dehazed.png"), cv2.cvtColor(e518, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            skipped.append([hazy_path.name, repr(exc)])
        if i % CHECKPOINT_EVERY == 0 or i == len(pairs):
            write_csv(per_image_path, header, rows)
            write_csv(OUT / "skipped.csv", ["image", "reason"], skipped)
            print(f"Progress: {i}/{len(pairs)} | successful: {len(rows)} | skipped: {len(skipped)}", flush=True)

    metric_names = ["PSNR", "SSIM", "FSIM", "S_CIELAB", "EPI", "Edge_Retention",
                    "MS_SSIM", "BRISQUE", "NIQE", "PIQE"]
    idx = [(1,2), (4,5), (7,8), (10,11), (13,14), (16,17),
           (19,20), (22,23), (25,26), (28,29)]
    summary = []
    for name, (a, b) in zip(metric_names, idx):
        va = np.array([float(r[a]) for r in rows], dtype=float)
        vb = np.array([float(r[b]) for r in rows], dtype=float)
        ma = float(np.nanmean(va)) if np.isfinite(va).any() else float("nan")
        mb = float(np.nanmean(vb)) if np.isfinite(vb).any() else float("nan")
        summary.append([name, ma, mb, mb-ma, len(rows)])
    write_csv(summary_path, ["metric", "E3E_mean", "E5_1_mean", "change_E5_1_minus_E3E", "n"], summary)
    print("\n=== E3-E vs E5.1 validation summary ===")
    for r in summary:
        print(f"{r[0]}: E3-E={r[1]:.6f}, E5.1={r[2]:.6f}, change={r[3]:+.6f}, n={r[4]}")
    print(f"Per-image CSV: {per_image_path}")
    print(f"Summary CSV:   {summary_path}")
    print(f"E3-E images:   {e3e_dir}")
    print(f"E5.1 images:   {e51_dir}")
    if uh or uc:
        print(f"Unmatched hazy={len(uh)}, unmatched clear={len(uc)}")


if __name__ == "__main__":
    main()
