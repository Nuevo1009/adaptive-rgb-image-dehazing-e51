"""Shared utilities for the separate E5/E6/E7 validation runs."""
from pathlib import Path
import csv
import numpy as np
import cv2

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import (
    DehazeConfig, dehaze_image, read_rgb, save_rgb, validate_rgb,
    gray, preserve_rgb_chroma
)

SPLIT = "validation"  # Deliberately never uses test data.
DATA_ROOT = PROJECT_ROOT / "dataset" / "CNN_Dataset" / SPLIT
OUT_ROOT = PROJECT_ROOT / "outputs" / "e5_e6_e7_separate" / "CNN_Dataset_validation"
METRIC_NAMES = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]

def to_u8(rgb):
    a = np.asarray(rgb)
    if a.dtype == np.uint8:
        return a
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)

def box_mean(x, radius):
    k = 2 * int(radius) + 1
    return cv2.boxFilter(x.astype(np.float32), -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT)

def adaptive_rgb_chroma(input_rgb, dehazed_rgb, base_alpha=0.40):
    src = validate_rgb(input_rgb)
    out = validate_rgb(dehazed_rgb)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y = np.sum(src * weights, axis=2)
    mu = box_mean(y, 5)
    var = np.maximum(box_mean(y*y, 5) - mu*mu, 0.0)
    texture = np.sqrt(var)
    p10, p90 = np.percentile(texture, [10, 90])
    texture = np.clip((texture-p10)/max(float(p90-p10), 1e-6), 0, 1)
    alpha = np.clip(0.30 + 0.10*texture, 0.30, min(0.40, base_alpha))
    y_src = np.sum(src*weights, axis=2, keepdims=True)
    y_out = np.sum(out*weights, axis=2, keepdims=True)
    result = y_out + (1-alpha[...,None])*(out-y_out) + alpha[...,None]*(src-y_src)
    return np.clip(result, 0, 1).astype(np.float32), alpha

def e7_spatial_floor(rgb, a_map, t_map, t_min=0.05, extra_floor=0.02):
    y = gray(validate_rgb(rgb))
    darkness = np.clip((0.45-y)/0.45, 0, 1)
    floor_map = float(t_min) + float(extra_floor)*darkness
    out = (validate_rgb(rgb)-a_map)/np.maximum(t_map, floor_map)[...,None] + a_map
    return np.clip(out, 0, 1).astype(np.float32), floor_map

def metric_values(ref_u8, pred_u8, calculate_fsim=True):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.color import rgb2lab, deltaE_cie76, rgb2gray
    from scipy.ndimage import sobel

    psnr = float(peak_signal_noise_ratio(ref_u8, pred_u8, data_range=255))
    ssim = float(structural_similarity(ref_u8, pred_u8, channel_axis=2, data_range=255))
    lab_a = rgb2lab(ref_u8.astype(np.float32)/255.0)
    lab_b = rgb2lab(pred_u8.astype(np.float32)/255.0)
    delta = float(np.mean(deltaE_cie76(lab_a, lab_b)))

    fsim = float("nan")
    if calculate_fsim:
        try:
            import torch, piq
            a = torch.from_numpy(ref_u8.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
            b = torch.from_numpy(pred_u8.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
            with torch.no_grad():
                fsim = float(piq.fsim(b, a, data_range=1.0).item())
        except Exception as exc:
            print(f"[FSIM unavailable] {exc}")

    ga = rgb2gray(ref_u8.astype(np.float32)/255.0)
    gb = rgb2gray(pred_u8.astype(np.float32)/255.0)
    ea = np.hypot(sobel(ga, axis=0), sobel(ga, axis=1))
    eb = np.hypot(sobel(gb, axis=0), sobel(gb, axis=1))
    epi = float(np.corrcoef(ea.ravel(), eb.ravel())[0,1]) if np.std(ea)>0 and np.std(eb)>0 else float("nan")
    edge = float(np.mean(eb)/np.mean(ea)) if np.mean(ea)>1e-12 else float("nan")
    return [psnr, ssim, fsim, delta, epi, edge]

def configs():
    base = dict(downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE,
                omega=OMEGA, gamma=GAMMA, t_min=T_MIN)
    return DehazeConfig(**base), DehazeConfig.e3e_selected(**base)

def run_one(label, variant, fsim=True):
    hazy_dir, clear_dir = DATA_ROOT/"hazy", DATA_ROOT/"clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Missing validation folders: {hazy_dir} | {clear_dir}")
    pairs, uh, uc = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No paired validation images found.")
    baseline_cfg, e3_cfg = configs()
    cfg = e3_cfg
    if variant == "PAPER_BASELINE":
        cfg = baseline_cfg
    elif variant == "E6":
        cfg = DehazeConfig.e3e_selected(
            downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE,
            omega=OMEGA, gamma=GAMMA, t_min=T_MIN,
            guided_filter_radius=30, guided_filter_eps=0.012
        )

    out_dir = OUT_ROOT / label
    img_dir = out_dir / "dehazed"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{label.lower()}_per_image.csv"
    fields = ["hazy_image", "ground_truth"] + METRIC_NAMES + ["extra_mean", "extra_min", "extra_max"]
    # Resume safely: already-written rows are skipped if rerun.
    done = set()
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            done = {r["hazy_image"] for r in csv.DictReader(f)}
    mode = "a" if csv_path.exists() else "w"
    processed = len(done)
    with csv_path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
            f.flush()
        for i, (hp, gp) in enumerate(pairs, 1):
            if hp.name in done:
                continue
            try:
                hazy = validate_rgb(read_rgb(hp))
                gt = to_u8(read_rgb(gp))
                pack = dehaze_image(hazy, cfg)
                pred = pack["dehazed"]
                extra = [float("nan")]*3
                if variant == "E5":
                    pred, amap = adaptive_rgb_chroma(hazy, pred, .40)
                    extra = [float(np.mean(amap)), float(np.min(amap)), float(np.max(amap))]
                elif variant == "E7":
                    pred_phys, fmap = e7_spatial_floor(
                        hazy, pack["atmospheric_map"], pack["transmission_map"],
                        t_min=e3_cfg.t_min, extra_floor=.02
                    )
                    pred = preserve_rgb_chroma(hazy, pred_phys, e3_cfg.color_preserve_strength)
                    extra = [float(np.mean(fmap)), float(np.min(fmap)), float(np.max(fmap))]
                pred8 = to_u8(pred)
                if pred8.shape != gt.shape:
                    raise ValueError(f"Image dimensions differ: output={pred8.shape}, GT={gt.shape}")
                vals = metric_values(gt, pred8, calculate_fsim=fsim)
                save_rgb(img_dir/f"{hp.stem}_dehazed.png", pred)
                row = dict(zip(fields, [hp.name, gp.name, *vals, *extra]))
                writer.writerow(row)
                f.flush()  # Critical: persist every completed image.
                processed += 1
            except Exception as exc:
                print(f"\n[SKIP] {hp.name}: {exc}")
            if i % 10 == 0 or i == len(pairs):
                print(f"{label}: scanned {i}/{len(pairs)} | saved rows: {processed}", flush=True)
    print(f"\nFinished {label}. Per-image metrics: {csv_path}")
    print(f"Unmatched hazy={len(uh)}, unmatched clear={len(uc)}")
    print("The run can be safely restarted; saved image rows will be skipped.")
