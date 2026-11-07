"""E5.3 Edge- and Colour-Confidence-Guided Adaptive RGB Chroma Preservation.

Validation-only. Preserves the original E3-E atmospheric-light, transmission,
and reconstruction equations. E5.3 is a bounded RGB-chroma post-process.
It uses no ground-truth values to construct its confidence maps.

Requires the completed E5.2 per-image CSV to build the four-version comparison.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
import pandas as pd

from dataset_config import PROJECT_ROOT, match_pairs
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, validate_rgb

SPLIT = "validation"  # Do not change to test during development.
HAZY_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT / "hazy"
CLEAR_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT / "clear"
OUT = PROJECT_ROOT / "outputs" / "E5_3_edge_colour_confidence_validation"
E52_CSV = PROJECT_ROOT / "outputs" / "E5_2_edge_aware_validation" / "E3E_E5_1_E5_2_per_image.csv"
CHECKPOINT_EVERY = 20

E3E_CONFIG = DehazeConfig.e3e_selected(
    guided_filter_radius=20,
    guided_filter_eps=1e-3,
)

# Frozen E5.1 parameters
BASE_ALPHA = 0.40
ALPHA_GAIN = 0.20
ALPHA_MIN = 0.20
ALPHA_MAX = 0.60
DARK_LUMA = 0.16
BRIGHT_LUMA = 0.88

# E5.3 bounded adjustments. These are hypotheses, not tuned/optimal values.
MAX_ALPHA_ADJUSTMENT = 0.04
EDGE_PROTECTION_GAIN = 0.04
COLOUR_RESTORE_GAIN = 0.04


def rgb_chroma(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y = np.sum(rgb * weights, axis=2, keepdims=True)
    return rgb - y, y[..., 0]


def e51_alpha_map(input_rgb, e3e_rgb):
    src, out = validate_rgb(input_rgb), validate_rgb(e3e_rgb)
    c_src, y_src = rgb_chroma(src)
    c_out, _ = rgb_chroma(out)
    disagreement = np.mean(np.abs(c_src - c_out), axis=2)
    disagreement_conf = np.clip(disagreement / 0.15, 0.0, 1.0)
    dark_gate = np.clip((y_src - DARK_LUMA) / 0.18, 0.0, 1.0)
    bright_gate = np.clip((BRIGHT_LUMA - y_src) / 0.12, 0.0, 1.0)
    luminance_gate = dark_gate * bright_gate
    saturation = np.linalg.norm(c_src, axis=2)
    saturation_gate = np.clip(saturation / 0.20, 0.0, 1.0)
    confidence = disagreement_conf * luminance_gate * (0.65 + 0.35 * saturation_gate)
    alpha = BASE_ALPHA + ALPHA_GAIN * (confidence - 0.35)
    alpha = np.clip(alpha, ALPHA_MIN, ALPHA_MAX).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return np.clip(alpha, ALPHA_MIN, ALPHA_MAX).astype(np.float32)


def _gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def confidence_maps(input_rgb, e3e_rgb):
    """Independent input-derived structural-edge and colour-change confidences."""
    src, out = validate_rgb(input_rgb), validate_rgb(e3e_rgb)
    gray_src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    sx, sy, sm = _gradient(gray_src)
    ox, oy, om = _gradient(gray_out)

    # Structural confidence is high when both estimates contain a local edge
    # and their gradient directions agree. This avoids treating every input
    # texture/noise response as a genuine boundary.
    dot = sx * ox + sy * oy
    denom = sm * om + 1e-8
    direction_agreement = np.clip(dot / denom, 0.0, 1.0)
    scale_s = float(np.percentile(sm, 95)) + 1e-8
    scale_o = float(np.percentile(om, 95)) + 1e-8
    edge_presence = np.clip(np.minimum(sm / scale_s, om / scale_o), 0.0, 1.0)
    edge_conf = edge_presence * direction_agreement
    edge_conf = cv2.GaussianBlur(edge_conf.astype(np.float32), (0, 0), 1.0)

    # Colour confidence uses RGB chroma disagreement, gated away from deep
    # shadows and clipped highlights. It does not claim to know true colour;
    # it estimates where E3-E changed chroma relative to the observed input.
    cs, ys = rgb_chroma(src)
    co, _ = rgb_chroma(out)
    chroma_diff = np.mean(np.abs(cs - co), axis=2)
    chroma_conf = np.clip(chroma_diff / 0.15, 0.0, 1.0)
    dark_gate = np.clip((ys - DARK_LUMA) / 0.18, 0.0, 1.0)
    bright_gate = np.clip((BRIGHT_LUMA - ys) / 0.12, 0.0, 1.0)
    colour_conf = chroma_conf * dark_gate * bright_gate
    colour_conf = cv2.GaussianBlur(colour_conf.astype(np.float32), (0, 0), 1.5)

    return np.clip(edge_conf, 0, 1), np.clip(colour_conf, 0, 1)


def apply_e53(input_rgb, e3e_rgb):
    src, out = validate_rgb(input_rgb), validate_rgb(e3e_rgb)
    alpha51 = e51_alpha_map(src, out)
    edge_conf, colour_conf = confidence_maps(src, out)

    # Colour confidence can restore a little input chroma in reliable,
    # non-edge areas. Structural confidence suppresses that restoration.
    # The net deviation from E5.1 is strictly bounded.
    delta = (COLOUR_RESTORE_GAIN * colour_conf * (1.0 - edge_conf)
             - EDGE_PROTECTION_GAIN * edge_conf)
    delta = np.clip(delta, -MAX_ALPHA_ADJUSTMENT, MAX_ALPHA_ADJUSTMENT)
    alpha53 = np.clip(alpha51 + delta, ALPHA_MIN, ALPHA_MAX).astype(np.float32)

    # Smooth the adjustment, then enforce the bound again relative to E5.1.
    alpha53 = cv2.GaussianBlur(alpha53, (0, 0), 0.8)
    alpha53 = np.clip(alpha53, alpha51 - MAX_ALPHA_ADJUSTMENT,
                      alpha51 + MAX_ALPHA_ADJUSTMENT)
    alpha53 = np.clip(alpha53, ALPHA_MIN, ALPHA_MAX)

    c_src, _ = rgb_chroma(src)
    c_out, y_out = rgb_chroma(out)
    c_new = (1.0 - alpha53[..., None]) * c_out + alpha53[..., None] * c_src
    result = np.clip(y_out[..., None] + c_new, 0.0, 1.0).astype(np.float32)
    return result, alpha51, alpha53, edge_conf, colour_conf


def as_u8(x):
    return np.clip(validate_rgb(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)




def save_panel(path, hazy, gt, e3e, e51, e52, e53):
    ims = [as_u8(v) for v in (hazy, gt, e3e, e51, e52, e53)]
    labels = ["Hazy", "Ground truth", "E3-E", "E5.1", "E5.2", "E5.3"]
    h, w = ims[0].shape[:2]
    cw = min(300, w)
    ch = max(1, int(h * cw / w))
    tiles = []
    for label, im in zip(labels, ims):
        tile = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (cw, ch))
        cv2.putText(tile, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(path), cv2.hconcat(tiles))


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
    eg = np.hypot(sobel(g, axis=1), sobel(g, axis=0))
    ep = np.hypot(sobel(p, axis=1), sobel(p, axis=0))
    edge = float(np.mean(ep) / (np.mean(eg) + 1e-12))
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


def save_panel(path, hazy, gt, e3e, e51, e52, e53):
    ims = [as_u8(v) for v in (hazy, gt, e3e, e51, e52, e53)]
    labels = ["Hazy", "Ground truth", "E3-E", "E5.1", "E5.2", "E5.3"]
    h, w = ims[0].shape[:2]
    cw = min(300, w)
    ch = max(1, int(h * cw / w))
    tiles = []
    for label, im in zip(labels, ims):
        tile = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (cw, ch))
        cv2.putText(tile, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(path), cv2.hconcat(tiles))


def main():
    if not E52_CSV.is_file():
        raise SystemExit(
            f"Missing prior E5.2 per-image CSV: {E52_CSV}\n"
            "Run run_e52_validation.py first; E5.3 uses its E5.2 measurements "
            "to create a complete four-version comparison."
        )
    for d in (HAZY_DIR, CLEAR_DIR):
        if not d.is_dir():
            raise SystemExit(f"Dataset folder not found: {d}")
    prior = pd.read_csv(E52_CSV)
    required = {"image", "PSNR_E3E", "PSNR_E5_1", "PSNR_E5_2",
                "SSIM_E3E", "SSIM_E5_1", "SSIM_E5_2",
                "FSIM_E3E", "FSIM_E5_1", "FSIM_E5_2",
                "CIE76_DeltaE_E3E", "CIE76_DeltaE_E5_1", "CIE76_DeltaE_E5_2",
                "EPI_E3E", "EPI_E5_1", "EPI_E5_2",
                "Edge_E3E", "Edge_E5_1", "Edge_E5_2"}
    missing = required - set(prior.columns)
    if missing:
        raise SystemExit(f"E5.2 CSV missing columns: {sorted(missing)}")

    pairs, uh, uc = match_pairs(HAZY_DIR, CLEAR_DIR)
    prior["image"] = prior["image"].astype(str)
    prior_map = prior.set_index("image")
    OUT.mkdir(parents=True, exist_ok=True)
    e53_dir = OUT / "E5_3_ADAPTIVE_RGB"
    e53_dir.mkdir(parents=True, exist_ok=True)
    e52_dir = PROJECT_ROOT / "outputs" / "E5_2_edge_aware_validation" / "E5_2_EDGE_AWARE_RGB"
    visual_dir = OUT / "WORST_CASE_VISUALS"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    print(f"Matched validation pairs: {len(pairs)}")
    print("Split: validation only. Test split is not accessed.")
    print("E3-E equations and E3-E preset are unchanged.")
    print(f"Using completed E5.2 comparison: {E52_CSV}")

    for i, (hp, gp) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hp)
            gt = as_u8(read_rgb(gp))
            if hazy.shape != gt.shape:
                skipped.append([hp.name, "dimension mismatch"])
                continue
            e3e = dehaze_image(hazy, E3E_CONFIG)["dehazed"]
            e51, alpha51 = None, None
            # Recreate E5.1 exactly using the frozen E5.1 alpha rule.
            alpha51 = e51_alpha_map(hazy, e3e)
            csrc, _ = rgb_chroma(hazy)
            cout, yout = rgb_chroma(e3e)
            c51 = (1.0-alpha51[..., None])*cout + alpha51[..., None]*csrc
            e51 = np.clip(yout[..., None] + c51, 0, 1).astype(np.float32)
            e53, _, alpha53, edge_conf, colour_conf = apply_e53(hazy, e3e)
            e3e8, e518, e538 = as_u8(e3e), as_u8(e51), as_u8(e53)
            m0, m1, m3 = metrics(gt, e3e8), metrics(gt, e518), metrics(gt, e538)
            if hp.name not in prior_map.index:
                skipped.append([hp.name, "not present in E5.2 per-image CSV"])
                continue
            old = prior_map.loc[hp.name]
            row = {"image": hp.name}
            # Metrics from previous completed E5.2 run, preserving the same
            # E3-E/E5.1/E5.2 observations for the four-way comparison.
            for metric, vals in [
                ("PSNR", [m0[0], m1[0], float(old["PSNR_E5_2"]), m3[0]]),
                ("SSIM", [m0[1], m1[1], float(old["SSIM_E5_2"]), m3[1]]),
                ("FSIM", [m0[2], m1[2], float(old["FSIM_E5_2"]), m3[2]]),
                ("CIE76_DeltaE", [m0[3], m1[3], float(old["CIE76_DeltaE_E5_2"]), m3[3]]),
                ("EPI", [m0[4], m1[4], float(old["EPI_E5_2"]), m3[4]]),
                ("Edge", [m0[5], m1[5], float(old["Edge_E5_2"]), m3[5]])]:
                for ver, val in zip(["E3E", "E5_1", "E5_2", "E5_3"], vals):
                    row[f"{metric}_{ver}"] = val
            row.update({"alpha51_mean": float(np.mean(alpha51)),
                        "alpha53_mean": float(np.mean(alpha53)),
                        "alpha53_min": float(np.min(alpha53)),
                        "alpha53_max": float(np.max(alpha53)),
                        "edge_conf_mean": float(np.mean(edge_conf)),
                        "colour_conf_mean": float(np.mean(colour_conf))})
            rows.append(row)
            cv2.imwrite(str(e53_dir / f"{hp.stem}_dehazed.png"),
                        cv2.cvtColor(e538, cv2.COLOR_RGB2BGR))
            if hp.stem in {"1392", "922", "4410", "1841"}:
                e52_path = e52_dir / f"{hp.stem}_dehazed.png"
                if e52_path.is_file():
                    e52_bgr = cv2.imread(str(e52_path), cv2.IMREAD_COLOR)
                    e52_rgb = cv2.cvtColor(e52_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    save_panel(visual_dir / f"{hp.stem}_E3E_E5_1_E5_2_E5_3.jpg",
                               hazy, gt, e3e, e51, e52_rgb, e53)
                else:
                    skipped.append([hp.name, f"Missing E5.2 visual output: {e52_path}"])
        except Exception as exc:
            skipped.append([hp.name, repr(exc)])
        if i % CHECKPOINT_EVERY == 0 or i == len(pairs):
            pd.DataFrame(rows).to_csv(OUT / "E3E_E5_1_E5_2_E5_3_per_image.csv", index=False)
            write_csv(OUT / "skipped.csv", ["image", "reason"], skipped)
            print(f"Progress: {i}/{len(pairs)} | successful: {len(rows)} | skipped: {len(skipped)}", flush=True)

    if not rows:
        raise SystemExit("No rows evaluated. Check filename matching and prior CSV.")
    frame = pd.DataFrame(rows)
    metric_names = ["PSNR", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge"]
    summary = []
    for metric in metric_names:
        for ver in ["E3E", "E5_1", "E5_2", "E5_3"]:
            vals = pd.to_numeric(frame[f"{metric}_{ver}"], errors="coerce")
            summary.append([metric, ver, float(vals.mean()), int(vals.notna().sum())])
    pd.DataFrame(summary, columns=["metric", "version", "mean", "n"]).to_csv(
        OUT / "E3E_E5_1_E5_2_E5_3_summary.csv", index=False)
    print("\n=== Four-version validation summary ===")
    for metric in metric_names:
        vals = [frame[f"{metric}_{v}"].mean() for v in ["E3E", "E5_1", "E5_2", "E5_3"]]
        print(f"{metric}: " + " | ".join(f"{v}={x:.6f}" for v, x in zip(["E3E","E5.1","E5.2","E5.3"], vals)))
    print(f"Output folder: {OUT}")
    print("Note: E5.2 values are imported from the prior completed E5.2 per-image CSV.")


if __name__ == "__main__":
    main()
