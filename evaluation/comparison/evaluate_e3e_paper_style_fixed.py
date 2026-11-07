import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""
E3-E selected pipeline: baseline-vs-corrected metric comparison.

Produces:
  paper_style_e3e_comparison.csv
  paper_style_e3e_summary.csv
  paper_style_e3e_skipped.csv

The baseline and corrected folders must each contain images corresponding
to the same hazy input filenames (same stem is used for matching).
"""

from pathlib import Path
import csv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_cie76, rgb2gray
from scipy.ndimage import sobel

try:
    import torch
    import piq
    HAS_FSIM = True
except Exception:
    HAS_FSIM = False

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def fsim_score(ref, img):
    if not HAS_FSIM:
        return float("nan")
    x = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    y = torch.from_numpy(ref.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(piq.fsim(x, y, data_range=1.0).item())


def metrics(ref, img):
    psnr = float(peak_signal_noise_ratio(ref, img, data_range=255))
    ssim = float(structural_similarity(ref, img, channel_axis=2, data_range=255))
    fsim = fsim_score(ref, img)
    lab_ref = rgb2lab(ref.astype(np.float32) / 255.0)
    lab_img = rgb2lab(img.astype(np.float32) / 255.0)
    de = float(np.mean(deltaE_cie76(lab_ref, lab_img)))

    ref_g = rgb2gray(ref.astype(np.float32) / 255.0)
    img_g = rgb2gray(img.astype(np.float32) / 255.0)
    rx, ry = sobel(ref_g, axis=1), sobel(ref_g, axis=0)
    ix, iy = sobel(img_g, axis=1), sobel(img_g, axis=0)
    ref_mag, img_mag = np.hypot(rx, ry), np.hypot(ix, iy)
    epi = float(np.corrcoef(ref_mag.ravel(), img_mag.ravel())[0, 1]) if np.std(ref_mag) > 0 and np.std(img_mag) > 0 else float("nan")
    edge = float(np.mean(img_mag) / np.mean(ref_mag)) if np.mean(ref_mag) > 1e-12 else float("nan")
    return psnr, ssim, fsim, de, epi, edge


def image_key(path):
    """
    Match dehazed outputs such as 0001_0.8_0.2_dehazed.png
    with dataset references such as 0001_0.8_0.2.jpg.
    The extension is already removed by Path.stem.
    """
    key = path.stem.lower()
    for suffix in ("_dehazed", "_dehaze", "_output", "_result"):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
            break
    return key


def index_images(folder):
    out = {}
    duplicates = []
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            key = image_key(p)
            if key in out:
                duplicates.append((key, str(out[key]), str(p)))
            else:
                out[key] = p
    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate normalized filename keys in {folder}")
        for item in duplicates[:5]:
            print("  Duplicate:", item)
    return out


def main():
    print("\n=== E3-E Selected: Baseline vs Corrected Metrics ===")
    split = input("Dataset split (validation/test): ").strip().lower()
    if split not in {"validation", "test"}:
        raise SystemExit("Enter validation or test.")
    project = Path(__file__).resolve().parent
    gt_dir = project / "dataset" / "CNN_Dataset" / split / "clear"
    hazy_dir = project / "dataset" / "CNN_Dataset" / split / "hazy"
    if not gt_dir.is_dir():
        raise SystemExit(f"Ground-truth folder not found: {gt_dir}")

    print("\nEnter image folders. These should contain corresponding dehazed images.")
    baseline_dir = Path(input("Baseline output folder: ").strip().strip('"'))
    corrected_dir = Path(input("Corrected E3-E selected output folder: ").strip().strip('"'))
    if not baseline_dir.is_dir() or not corrected_dir.is_dir():
        raise SystemExit("One or both output folders do not exist.")

    gt_map = index_images(gt_dir)
    base_map = index_images(baseline_dir)
    corr_map = index_images(corrected_dir)
    common = sorted(set(gt_map) & set(base_map) & set(corr_map))
    print(f"Matched image triplets: {len(common)}")
    if not common:
        raise SystemExit("No common filenames found. Ensure output filenames match GT filenames.")

    rows, skipped = [], []
    for n, key in enumerate(common, 1):
        try:
            gt = read_rgb(gt_map[key])
            base = read_rgb(base_map[key])
            corr = read_rgb(corr_map[key])
            if gt.shape != base.shape or gt.shape != corr.shape:
                skipped.append([key, f"shape mismatch: GT{gt.shape}, baseline{base.shape}, corrected{corr.shape}"])
                continue
            bm = metrics(gt, base)
            cm = metrics(gt, corr)
            rows.append({
                "image": gt_map[key].name,
                "reference_image": gt_map[key].name,
                "clear_ground_truth": str(gt_map[key]),
                "baseline_PSNR_dB_vs_clear_GT": bm[0],
                "baseline_SSIM_vs_clear_GT": bm[1],
                "baseline_FSIM_vs_clear_GT": bm[2],
                "baseline_CIE76_DeltaE_vs_clear_GT": bm[3],
                "baseline_EPI_vs_clear_GT": bm[4],
                "baseline_Edge_Retention_vs_clear_GT": bm[5],
                "corrected_PSNR_dB_vs_clear_GT": cm[0],
                "corrected_SSIM_vs_clear_GT": cm[1],
                "corrected_FSIM_vs_clear_GT": cm[2],
                "corrected_CIE76_DeltaE_vs_clear_GT": cm[3],
                "corrected_EPI_vs_clear_GT": cm[4],
                "corrected_Edge_Retention_vs_clear_GT": cm[5],
                "baseline_output": str(base_map[key]),
                "corrected_output": str(corr_map[key]),
            })
        except Exception as e:
            skipped.append([key, str(e)])
        if n % 25 == 0 or n == len(common):
            print(f"Processed {n}/{len(common)}", end="\r", flush=True)

    out_dir = project / "outputs" / "evaluation" / "e3e_paper_style" / f"CNN_Dataset_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "paper_style_e3e_comparison.csv"
    sum_path = out_dir / "paper_style_e3e_summary.csv"
    skip_path = out_dir / "paper_style_e3e_skipped.csv"

    columns = list(rows[0].keys()) if rows else []
    with per_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    metric_pairs = [
        ("PSNR_dB_vs_clear_GT", "baseline_PSNR_dB_vs_clear_GT", "corrected_PSNR_dB_vs_clear_GT"),
        ("SSIM_vs_clear_GT", "baseline_SSIM_vs_clear_GT", "corrected_SSIM_vs_clear_GT"),
        ("FSIM_vs_clear_GT", "baseline_FSIM_vs_clear_GT", "corrected_FSIM_vs_clear_GT"),
        ("CIE76_DeltaE_vs_clear_GT", "baseline_CIE76_DeltaE_vs_clear_GT", "corrected_CIE76_DeltaE_vs_clear_GT"),
        ("EPI_vs_clear_GT", "baseline_EPI_vs_clear_GT", "corrected_EPI_vs_clear_GT"),
        ("Edge_Retention_vs_clear_GT", "baseline_Edge_Retention_vs_clear_GT", "corrected_Edge_Retention_vs_clear_GT"),
    ]
    with sum_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline_mean", "corrected_mean", "corrected_minus_baseline_mean", "valid_pairs"])
        for label, bc, cc in metric_pairs:
            bv = np.asarray([r[bc] for r in rows], dtype=float)
            cv = np.asarray([r[cc] for r in rows], dtype=float)
            bm = float(np.nanmean(bv)) if np.isfinite(bv).any() else float("nan")
            cm = float(np.nanmean(cv)) if np.isfinite(cv).any() else float("nan")
            w.writerow([label, bm, cm, cm - bm, int(np.isfinite(bv).sum() & np.isfinite(cv).sum())])

    with skip_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "reason"])
        w.writerows(skipped)

    print("\n\n=== Summary ===")
    with sum_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            print(f"{row['metric']}: baseline={float(row['baseline_mean']):.6f}, corrected={float(row['corrected_mean']):.6f}, delta={float(row['corrected_minus_baseline_mean']):+.6f}, n={row['valid_pairs']}")
    if not HAS_FSIM:
        print("\\nFSIM is NaN: install FSIM dependencies with: pip install torch piq")
    print(f"\\nPer-image comparison: {per_path}")
    print(f"Summary:              {sum_path}")
    print(f"Skipped:              {skip_path}")


if __name__ == "__main__":
    main()
