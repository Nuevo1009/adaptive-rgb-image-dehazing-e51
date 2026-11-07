"""Create a paired comparison CSV after all five separate validation runs finish."""
from pathlib import Path
import csv
import numpy as np
from dataset_config import PROJECT_ROOT

ROOT = PROJECT_ROOT / "outputs" / "e5_e6_e7_separate" / "CNN_Dataset_validation"
RUNS = {
    "PAPER_BASELINE": "PAPER_BASELINE",
    "E3E_REFERENCE": "E3E_REFERENCE",
    "E5_ADAPTIVE_RGB": "E5_ADAPTIVE_RGB",
    "E6_GUIDED_TUNE": "E6_GUIDED_TUNE",
    "E7_DARK_TFLOOR": "E7_DARK_TFLOOR",
}
METRICS = ["PSNR_dB", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention"]

def load(path):
    if not path.exists():
        raise SystemExit(f"Missing result file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["hazy_image"]: r for r in csv.DictReader(f)}

def main():
    data = {}
    for label, folder in RUNS.items():
        path = ROOT/folder/f"{folder.lower()}_per_image.csv"
        data[label] = load(path)
    common = set.intersection(*(set(v) for v in data.values()))
    if not common:
        raise SystemExit("No common image filenames across runs.")
    missing = {k: len(set(v)-common) for k,v in data.items()}
    rows = []
    summary = []
    for label in RUNS:
        for metric in METRICS:
            vals = []
            deltas = []
            for name in sorted(common):
                try:
                    value = float(data[label][name][metric])
                    base = float(data["PAPER_BASELINE"][name][metric])
                except (ValueError, KeyError):
                    continue
                if np.isfinite(value):
                    vals.append(value)
                if np.isfinite(value) and np.isfinite(base):
                    deltas.append(value-base)
            summary.append({
                "configuration": label, "metric": metric,
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "delta_vs_baseline": float(np.mean(deltas)) if deltas else float("nan"),
                "valid_images": len(vals), "paired_images": len(deltas)
            })
    for name in sorted(common):
        row = {"hazy_image": name}
        for label in RUNS:
            for metric in METRICS:
                row[f"{label}_{metric}"] = data[label][name].get(metric, "")
                if label != "PAPER_BASELINE":
                    try:
                        row[f"{label}_delta_{metric}"] = (
                            float(data[label][name][metric]) -
                            float(data["PAPER_BASELINE"][name][metric])
                        )
                    except (ValueError, KeyError):
                        row[f"{label}_delta_{metric}"] = ""
        rows.append(row)

    ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT/"e5_e6_e7_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    per_path = ROOT/"e5_e6_e7_comparison_per_image.csv"
    with per_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Common paired validation images: {len(common)}")
    print(f"Images missing from common set by run: {missing}")
    print(f"Summary comparison: {summary_path}")
    print(f"Per-image comparison: {per_path}")
    print("\nMean results (higher is better except CIE76_DeltaE):")
    for r in summary:
        print(f'{r["configuration"]:20s} {r["metric"]:18s} mean={r["mean"]:.6f} '
              f'delta_vs_baseline={r["delta_vs_baseline"]:+.6f} n={r["valid_images"]}')
if __name__ == "__main__":
    main()
