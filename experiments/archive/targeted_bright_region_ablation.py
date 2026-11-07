"""Targeted bright-region ablation: compare adaptive weighting (B) against no region penalty (D).
Place in project root beside dataset_config.py, settings.py, and dehazing/.
The 1% mask-fraction threshold is an analysis choice, not a parameter from a paper.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, validate_rgb, _large_uniform_near_white_mask

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ablation_study"
HIGH_REGION_THRESHOLD = 0.01
METRICS = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]

def u8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8: return x
    x = x.astype(np.float32)
    if x.size and np.nanmax(x) <= 1: x *= 255
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)

def get_metrics(pred, gt):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.color import rgb2gray, rgb2lab, deltaE_cie76
    from scipy.ndimage import sobel
    p, g = u8(pred), u8(gt)
    if p.shape != g.shape: raise ValueError(f"Size mismatch {p.shape} vs {g.shape}")
    psnr = float(peak_signal_noise_ratio(g, p, data_range=255))
    ssim = float(structural_similarity(g, p, channel_axis=2, data_range=255))
    try:
        import torch, piq
        a = torch.from_numpy(g.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
        b = torch.from_numpy(p.astype(np.float32)/255).permute(2,0,1).unsqueeze(0)
        with torch.no_grad(): fsim = float(piq.fsim(b, a, data_range=1.0).item())
    except ImportError: fsim = float("nan")
    de = float(np.mean(deltaE_cie76(rgb2lab(g.astype(np.float32)/255), rgb2lab(p.astype(np.float32)/255))))
    gg, pg = rgb2gray(g.astype(np.float32)/255), rgb2gray(p.astype(np.float32)/255)
    eg = np.hypot(sobel(gg, axis=1), sobel(gg, axis=0))
    ep = np.hypot(sobel(pg, axis=1), sobel(pg, axis=0))
    epi = float(np.corrcoef(eg.ravel(), ep.ravel())[0,1]) if np.std(eg)>0 and np.std(ep)>0 else float("nan")
    edge = float(np.mean(ep)/np.mean(eg)) if np.mean(eg)>1e-8 else float("nan")
    return dict(zip(METRICS, [psnr, ssim, fsim, de, epi, edge]))

def choose_dataset():
    root = PROJECT_ROOT / "dataset"
    print("\nChoose dataset: 1 CNN_Dataset, 2 I-HAZE, 3 O-HAZE, 4 manual folders")
    choice = input("Choice [1-4]: ").strip()
    if choice == "1":
        ds = root / "CNN_Dataset"
        if not ds.is_dir(): ds = root / "CNN Dataset"
        split = input("Split [train/validation/test] (default validation): ").strip().lower() or "validation"
        if split == "val": split = "validation"
        if split not in {"train","validation","test"}: raise SystemExit("Invalid split.")
        d = ds / split
        return d/"hazy", d/"clear", f"CNN_Dataset_{split}"
    if choice == "2":
        d = root / "I-HAZE"; return d/"ihaze_hazy", d/"ihaze_gt", "I-HAZE"
    if choice == "3":
        d = root / "O-HAZE"
        if not d.is_dir(): d = root / "O-HAZY"
        return d/"ohaze_hazy", d/"ohaze_GT", "O-HAZE"
    if choice == "4":
        h = Path(input("Hazy folder path: ").strip().strip('"')).expanduser()
        g = Path(input("Clear/GT folder path: ").strip().strip('"')).expanduser()
        return h, g, "custom_dataset"
    raise SystemExit("Invalid choice.")

def mean(vals):
    vals = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")

def main():
    try:
        import skimage, scipy
    except ImportError as e:
        raise SystemExit("Install dependencies: pip install scikit-image scipy") from e
    hazy_dir, gt_dir, label = choose_dataset()
    if not hazy_dir.is_dir(): raise SystemExit(f"Hazy folder not found: {hazy_dir}")
    if not gt_dir.is_dir(): raise SystemExit(f"Ground-truth folder not found: {gt_dir}")
    pairs, uh, ug = match_pairs(hazy_dir, gt_dir)
    print(f"Matched pairs: {len(pairs)} | unmatched hazy: {len(uh)} | unmatched clear: {len(ug)}")
    if not pairs: raise SystemExit("No safely matched pairs.")
    common = dict(downsample_factor=DOWNSAMPLE_FACTOR, patch_size=PATCH_SIZE, omega=OMEGA, gamma=GAMMA, t_min=T_MIN, adaptive_airlight=True, adaptive_strength=0.5)
    cfg_b = DehazeConfig(**common, large_region_weight=0.25)
    cfg_d = DehazeConfig(**common, large_region_weight=1.0)
    rows = []
    for i, (hp, gp) in enumerate(pairs, 1):
        try:
            rgb = validate_rgb(read_rgb(hp)); h,w = rgb.shape[:2]; f=max(1,int(DOWNSAMPLE_FACTOR))
            small = rgb if f==1 else cv2.resize(rgb,(max(1,round(w/f)),max(1,round(h/f))),interpolation=cv2.INTER_AREA)
            fraction = float(np.mean(_large_uniform_near_white_mask(small, cfg_b)))
            out_b = dehaze_image(rgb,cfg_b)["dehazed"]
            out_d = dehaze_image(rgb,cfg_d)["dehazed"]
            gt = read_rgb(gp); mb=get_metrics(out_b,gt); md=get_metrics(out_d,gt)
            r={"hazy_image":hp.name,"ground_truth_image":gp.name,"region_fraction":fraction,
               "group":"high_region" if fraction >= HIGH_REGION_THRESHOLD else "low_region"}
            for m in METRICS:
                r[f"B_{m}"]=mb[m]; r[f"D_{m}"]=md[m]
                r[f"B_minus_D_{m}"]=mb[m]-md[m] if np.isfinite(mb[m]) and np.isfinite(md[m]) else float("nan")
            rows.append(r)
        except Exception as e: print(f"\n[SKIP] {hp.name}: {e}")
        if i%10==0 or i==len(pairs): print(f"Progress: {i}/{len(pairs)}",end="\r",flush=True)
    out = OUTPUT_ROOT / label / "targeted_bright_region"; out.mkdir(parents=True,exist_ok=True)
    per = out/"targeted_region_ablation_per_image.csv"
    if rows:
        with per.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = out/"targeted_region_ablation_summary.csv"
    with summ.open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f); wr.writerow(["group","count","mean_region_fraction","method",*METRICS])
        for group in ["all","high_region","low_region"]:
            sub=rows if group=="all" else [r for r in rows if r["group"]==group]
            for method,prefix in [("B_penalty","B_"),("D_no_penalty","D_"),("B_minus_D","B_minus_D_")]:
                wr.writerow([group,len(sub),mean([r["region_fraction"] for r in sub]),method,*[mean([r[f"{prefix}{m}"] for r in sub]) for m in METRICS]])
    print(f"\nEvaluated: {len(rows)}/{len(pairs)}")
    print(f"High-region threshold: fraction >= {HIGH_REGION_THRESHOLD:.3f} (1% of downsampled pixels)")
    print(f"Per-image CSV: {per}\nSummary CSV: {summ}")
    print("For B_minus_D: positive favors B for PSNR/SSIM/FSIM/EPI; negative favors B for Delta E.")
    print("Edge retention needs visual interpretation.")

if __name__ == "__main__": main()
