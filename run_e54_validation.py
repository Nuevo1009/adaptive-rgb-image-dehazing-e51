"""E5.4 validation-only experiment: selective edge/colour-confidence RGB chroma preservation.

Keeps the E3-E atmospheric-light, transmission and reconstruction pipeline unchanged.
Uses input/dehazed images only to construct alpha maps; ground truth is used only for
metric evaluation. Requires the completed E5.3 per-image CSV for a five-version table.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
import pandas as pd

from dataset_config import PROJECT_ROOT, match_pairs
from dehazing.core import DehazeConfig, dehaze_image, read_rgb
from run_e53_validation import (
    as_u8, rgb_chroma, e51_alpha_map, confidence_maps, metrics
)

SPLIT = "validation"
HAZY_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT / "hazy"
CLEAR_DIR = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT / "clear"
OUT = PROJECT_ROOT / "outputs" / "E5_4_selective_edge_colour_validation"
E53_CSV = PROJECT_ROOT / "outputs" / "E5_3_edge_colour_confidence_validation" / "E3E_E5_1_E5_2_E5_3_per_image.csv"
CHECKPOINT_EVERY = 20

# Explicitly match the frozen E3-E reference used in the previous experiments.
E3E_CONFIG = DehazeConfig.e3e_selected(
    guided_filter_radius=20,
    guided_filter_eps=1e-3,
)

# E5.4 is intentionally more conservative than E5.3.
MAX_ALPHA_ADJUSTMENT = 0.02
EDGE_THRESHOLD_LOW = 0.20
EDGE_THRESHOLD_HIGH = 0.65
COLOUR_THRESHOLD_LOW = 0.10
COLOUR_THRESHOLD_HIGH = 0.55
SMOOTH_SIGMA = 1.2


def smoothstep(x, lo, hi):
    z = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def selective_confidence_maps(input_rgb, e3e_rgb):
    """Return selective edge protection and reliable colour-change confidence."""
    src = np.clip(input_rgb.astype(np.float32), 0.0, 1.0)
    out = np.clip(e3e_rgb.astype(np.float32), 0.0, 1.0)
    edge_raw, colour_raw = confidence_maps(src, out)

    # Only strong, coherent edges are protected. Weak texture/noise responses
    # receive little weight. Confidence is smoothed to avoid pixel-level flicker.
    edge_selective = smoothstep(edge_raw, EDGE_THRESHOLD_LOW, EDGE_THRESHOLD_HIGH)

    # Require a spatially coherent chroma difference; isolated colour/noise
    # responses are suppressed by local averaging before thresholding.
    colour_coherent = cv2.GaussianBlur(
        colour_raw.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0
    )
    colour_selective = smoothstep(
        colour_coherent, COLOUR_THRESHOLD_LOW, COLOUR_THRESHOLD_HIGH
    )

    # Gate adaptation in deep shadows and clipped highlights.
    _, y = rgb_chroma(src)
    shadow_gate = smoothstep(y, 0.12, 0.28)
    highlight_gate = 1.0 - smoothstep(y, 0.88, 0.98)
    exposure_gate = shadow_gate * highlight_gate

    edge_selective = cv2.GaussianBlur(
        edge_selective.astype(np.float32), (0, 0), SMOOTH_SIGMA
    )
    colour_selective = cv2.GaussianBlur(
        (colour_selective * exposure_gate).astype(np.float32),
        (0, 0), SMOOTH_SIGMA
    )
    return np.clip(edge_selective, 0, 1), np.clip(colour_selective, 0, 1)


def apply_e54(input_rgb, e3e_rgb):
    src = np.clip(input_rgb.astype(np.float32), 0.0, 1.0)
    out = np.clip(e3e_rgb.astype(np.float32), 0.0, 1.0)
    alpha51 = e51_alpha_map(src, out)
    edge_conf, colour_conf = selective_confidence_maps(src, out)

    # Colour restoration is permitted only in coherent, non-edge regions.
    # Strong structural edges suppress the alpha change. The adjustment is
    # strictly bounded to +/-0.02 relative to E5.1.
    positive = colour_conf * (1.0 - edge_conf)
    negative = edge_conf * (1.0 - 0.5 * colour_conf)
    delta = MAX_ALPHA_ADJUSTMENT * (positive - negative)
    delta = np.clip(delta, -MAX_ALPHA_ADJUSTMENT, MAX_ALPHA_ADJUSTMENT)

    alpha54 = alpha51 + delta
    alpha54 = cv2.GaussianBlur(
        alpha54.astype(np.float32), (0, 0), SMOOTH_SIGMA
    )
    alpha54 = np.clip(
        alpha54, alpha51 - MAX_ALPHA_ADJUSTMENT,
        alpha51 + MAX_ALPHA_ADJUSTMENT
    )
    alpha54 = np.clip(alpha54, 0.20, 0.60).astype(np.float32)

    c_src, _ = rgb_chroma(src)
    c_out, y_out = rgb_chroma(out)
    c_new = (1.0 - alpha54[..., None]) * c_out + alpha54[..., None] * c_src
    result = np.clip(y_out[..., None] + c_new, 0.0, 1.0).astype(np.float32)
    return result, alpha51, alpha54, edge_conf, colour_conf


def save_panel(path, hazy, gt, e3e, e51, e52, e53, e54):
    ims = [as_u8(v) for v in (hazy, gt, e3e, e51, e52, e53, e54)]
    labels = ["Hazy", "Ground truth", "E3-E", "E5.1", "E5.2", "E5.3", "E5.4"]
    h, w = ims[0].shape[:2]
    tile_w = min(260, w)
    tile_h = max(1, int(h * tile_w / w))
    tiles = []
    for label, im in zip(labels, ims):
        tile = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (tile_w, tile_h))
        cv2.putText(tile, label, (7, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(path), cv2.hconcat(tiles))


def main():
    if not E53_CSV.is_file():
        raise SystemExit(
            f"Missing E5.3 per-image CSV: {E53_CSV}\n"
            "Run run_e53_validation.py first."
        )
    if not HAZY_DIR.is_dir() or not CLEAR_DIR.is_dir():
        raise SystemExit(f"Validation folders not found: {HAZY_DIR} / {CLEAR_DIR}")

    prior = pd.read_csv(E53_CSV)
    required = {"image"}
    for metric in ["PSNR", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge"]:
        required.update({f"{metric}_{v}" for v in ["E3E", "E5_1", "E5_2", "E5_3"]})
    missing = required - set(prior.columns)
    if missing:
        raise SystemExit(f"E5.3 CSV missing columns: {sorted(missing)}")

    prior["image"] = prior["image"].astype(str)
    prior_map = prior.set_index("image")
    pairs, _, _ = match_pairs(HAZY_DIR, CLEAR_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    out_images = OUT / "E5_4_ADAPTIVE_RGB"
    out_images.mkdir(parents=True, exist_ok=True)
    visuals = OUT / "CHALLENGING_CASES"
    visuals.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    print(f"Matched validation pairs: {len(pairs)}", flush=True)
    print("Validation only. The test split is not accessed.", flush=True)
    print("E3-E atmospheric-light/transmission/reconstruction equations unchanged.", flush=True)

    for idx, (hp, gp) in enumerate(pairs, 1):
        try:
            if hp.name not in prior_map.index:
                skipped.append([hp.name, "missing from E5.3 per-image CSV"])
                continue
            hazy = read_rgb(hp)
            gt = as_u8(read_rgb(gp))
            if hazy.shape != gt.shape:
                skipped.append([hp.name, "dimension mismatch"])
                continue

            e3e = dehaze_image(hazy, E3E_CONFIG)["dehazed"]
            e54, alpha51, alpha54, edge_conf, colour_conf = apply_e54(hazy, e3e)
            m54 = metrics(gt, as_u8(e54))
            old = prior_map.loc[hp.name]

            row = {"image": hp.name}
            for metric, new_value in zip(
                ["PSNR", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge"], m54
            ):
                for version in ["E3E", "E5_1", "E5_2", "E5_3"]:
                    row[f"{metric}_{version}"] = float(old[f"{metric}_{version}"])
                row[f"{metric}_E5_4"] = float(new_value)

            row.update({
                "alpha51_mean": float(np.mean(alpha51)),
                "alpha54_mean": float(np.mean(alpha54)),
                "alpha54_min": float(np.min(alpha54)),
                "alpha54_max": float(np.max(alpha54)),
                "edge_conf_mean": float(np.mean(edge_conf)),
                "colour_conf_mean": float(np.mean(colour_conf)),
            })
            rows.append(row)
            e54_u8 = as_u8(e54)
            cv2.imwrite(str(out_images / f"{hp.stem}_dehazed.png"),
                        cv2.cvtColor(e54_u8, cv2.COLOR_RGB2BGR))

            if hp.stem in {"1392", "922", "4410", "1841"}:
                # Use existing E5.2/E5.3 images if present; these are visual
                # references only. The CSV measurements remain from prior runs.
                def load_prior_image(folder, stem):
                    p = folder / f"{stem}_dehazed.png"
                    if not p.is_file():
                        return None
                    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                e52 = load_prior_image(
                    PROJECT_ROOT / "outputs" / "E5_2_edge_aware_validation" / "E5_2_EDGE_AWARE_RGB",
                    hp.stem
                )
                e53 = load_prior_image(
                    PROJECT_ROOT / "outputs" / "E5_3_edge_colour_confidence_validation" / "E5_3_ADAPTIVE_RGB",
                    hp.stem
                )
                if e52 is not None and e53 is not None:
                    save_panel(visuals / f"{hp.stem}_E3E_to_E54.jpg",
                               hazy, gt, e3e, (1-alpha51[...,None])*rgb_chroma(e3e)[0] + alpha51[...,None]*rgb_chroma(hazy)[0] + rgb_chroma(e3e)[1][...,None],
                               e52, e53, e54)

        except Exception as exc:
            skipped.append([hp.name, repr(exc)])

        if idx % CHECKPOINT_EVERY == 0 or idx == len(pairs):
            pd.DataFrame(rows).to_csv(OUT / "E3E_E5_1_E5_2_E5_3_E5_4_per_image.csv", index=False)
            pd.DataFrame(skipped, columns=["image", "reason"]).to_csv(OUT / "skipped.csv", index=False)
            print(f"Progress: {idx}/{len(pairs)} | successful: {len(rows)} | skipped: {len(skipped)}", flush=True)

    if not rows:
        raise SystemExit("No validation rows completed. See skipped.csv.")

    frame = pd.DataFrame(rows)
    versions = ["E3E", "E5_1", "E5_2", "E5_3", "E5_4"]
    metric_names = ["PSNR", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge"]
    summary_rows = []
    for metric in metric_names:
        for version in versions:
            values = pd.to_numeric(frame[f"{metric}_{version}"], errors="coerce")
            summary_rows.append([metric, version, float(values.mean()), int(values.notna().sum())])
    summary = pd.DataFrame(summary_rows, columns=["metric", "version", "mean", "n"])
    summary.to_csv(OUT / "E3E_E5_1_E5_2_E5_3_E5_4_summary.csv", index=False)

    print("\n=== Five-version validation summary ===")
    for metric in metric_names:
        line = []
        for version in versions:
            val = frame[f"{metric}_{version}"].mean()
            line.append(f"{version}={val:.6f}")
        print(f"{metric}: " + " | ".join(line))
    print(f"\nPer-image CSV: {OUT / 'E3E_E5_1_E5_2_E5_3_E5_4_per_image.csv'}")
    print(f"Summary CSV:   {OUT / 'E3E_E5_1_E5_2_E5_3_E5_4_summary.csv'}")
    print(f"Outputs:       {OUT}")
    print("E3-E/E5.1/E5.2/E5.3 metrics are imported from the completed E5.3 run.")
    print("E5.4 metrics are calculated now on the same validation pairs.")


if __name__ == "__main__":
    main()
