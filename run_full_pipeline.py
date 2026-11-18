"""
Full CNN classification + RGB dehazing evaluation pipeline.

Place this file in the project root (beside dataset_config.py, classifier/,
dehazing/, and run_e51_validation_extended.py).

Workflow:
  A. Classify dataset images and save classifications to CSV, OR
     reuse an existing classification CSV, OR skip classification.
  B. Compare Original paper baseline, E3-E, and E5.1.
  C. Save dehazed images, per-image metrics, summary CSVs, and graphs.

E5.1 is the E3-E preset followed by adaptive RGB chroma preservation.
The original paper baseline is DehazeConfig() with optional extensions off.
"""

from pathlib import Path
from datetime import datetime
import csv
import sys
import traceback

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_config import match_pairs, IMAGE_EXTS
from classifier.predict import load_model, classify_image
from dehazing.core import DehazeConfig, dehaze_image, read_rgb
import run_e51_validation_extended as e51


MODEL_PATH = PROJECT_ROOT / "models" / "best_haze_CNN.pth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "full_pipeline_comparison"
CLASSIFICATION_ROOT = OUTPUT_ROOT / "classification"
IMAGE_EXTENSIONS = IMAGE_EXTS

METRIC_NAMES = [
    "PSNR", "SSIM", "FSIM", "CIE76_DeltaE", "EPI", "Edge_Retention",
    "MS_SSIM", "BRISQUE", "NIQE", "PIQE"
]
# Higher is better for these; lower is better for all remaining metrics.
HIGHER_IS_BETTER = {"PSNR", "SSIM", "FSIM", "EPI", "Edge_Retention", "MS_SSIM"}


def create_unique_run_folder(dataset_name, mode_name):
    """Create a new timestamped directory for every experiment execution."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset_name}_{mode_name}_{stamp}"
    candidate = OUTPUT_ROOT / base_name
    suffix = 1
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def find_child(parent, names):
    if not parent.is_dir():
        return None
    lookup = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def get_dataset():
    base = PROJECT_ROOT / "dataset"
    print("\nSelect dataset:")
    print("  1. CNN_Dataset")
    print("  2. I-HAZE")
    print("  3. O-HAZY / O-HAZE")
    choice = input("Choice [1-3]: ").strip()

    if choice == "1":
        root = base / "CNN_Dataset"
        if not root.is_dir():
            root = base / "CNN Dataset"
        split = input("Split [train/validation/test] (default validation): ").strip().lower() or "validation"
        if split not in {"train", "validation", "test"}:
            raise SystemExit("Invalid CNN_Dataset split.")
        split_root = root / split
        hazy = find_child(split_root, ["hazy"])
        clear = find_child(split_root, ["clear", "gt", "ground_truth"])
        dataset_name = f"CNN_Dataset_{split}"
    elif choice == "2":
        root = base / "I-HAZE"
        hazy = find_child(root, ["ihaze_hazy", "hazy"])
        clear = find_child(root, ["ihaze_gt", "ihaze_GT", "gt", "clear"])
        split = "all"
        dataset_name = "I-HAZE"
    elif choice == "3":
        root = base / "O-HAZY"
        if not root.is_dir():
            root = base / "O-HAZE"
        hazy = find_child(root, ["ohaze_hazy", "ohazy_hazy", "hazy"])
        clear = find_child(root, ["ohaze_GT", "ohaze_gt", "ohazy_gt", "gt", "clear"])
        split = "all"
        dataset_name = "O-HAZE"
    else:
        raise SystemExit("Choose 1, 2, or 3.")

    if not root.is_dir() or hazy is None or clear is None:
        raise SystemExit(
            f"Could not find expected hazy/clear folders for {dataset_name}.\n"
            f"Dataset root checked: {root}"
        )
    return dataset_name, root, hazy, clear, split


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_classification_csv(path):
    mapping = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = Path(row.get("filename", "")).name
            if key:
                mapping[key] = row
    if not mapping:
        raise ValueError(f"No classification records found in {path}")
    return mapping


def classify_and_save(hazy_dir, dataset_name, run_dir):
    if not MODEL_PATH.is_file():
        raise SystemExit(f"CNN checkpoint not found: {MODEL_PATH}")

    images = sorted(
        p for p in hazy_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No images found in {hazy_dir}")

    print(f"\nLoading CNN checkpoint: {MODEL_PATH}")
    model, device = load_model(MODEL_PATH)
    rows = []
    for i, image_path in enumerate(images, 1):
        try:
            label, score, memberships = classify_image(image_path, model, device)
            rows.append([
                str(image_path.relative_to(hazy_dir)), image_path.name,
                score, label, memberships["clear"],
                memberships["uncertain"], memberships["hazy"]
            ])
            print(f"[{i}/{len(images)}] {image_path.name}: {label} ({score:.4f})")
        except Exception as exc:
            rows.append([
                str(image_path.relative_to(hazy_dir)), image_path.name,
                "", "ERROR", "", "", ""
            ])
            print(f"[{i}/{len(images)}] ERROR {image_path.name}: {exc}")

    out = run_dir / "classification" / f"{dataset_name}_classification.csv"
    write_csv(
        out,
        ["relative_path", "filename", "haze_score", "classification",
         "clear_membership", "uncertain_membership", "hazy_membership"],
        rows
    )
    print(f"\nClassification CSV saved: {out}")
    print(f"Classified images: {len(rows)}")
    return out


def save_rgb_u8(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = e51.as_u8(image)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save output image: {path}")


def metric_vector(gt, prediction):
    """Combine established full-reference and no-reference metrics."""
    full = e51.metrics(gt, prediction)
    extra = e51.extended_metrics(gt, prediction)
    return tuple(float(x) for x in (full + extra))


def plot_comparisons(summary_rows, dataset_name, output_dir):
    """Create individual report graphs with moderately zoomed y-axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "Adaptive Pixel-Wise Transmission Estimation (APTE)",
        "Proposed Method - 1",
        "Proposed Method - 2",
    ]
    x = np.arange(len(methods), dtype=float)

    for metric in METRIC_NAMES:
        selected = next((r for r in summary_rows if r[0] == metric), None)
        if selected is None:
            continue

        values = np.asarray([selected[1], selected[2], selected[3]], dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            print(f"Skipping {metric}: fewer than two finite method values.")
            continue

        finite_values = values[valid]
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))
        spread = vmax - vmin

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            x[valid], finite_values,
            marker="o", markersize=8, linewidth=2.2
        )

        # Numeric labels make the exact values readable in print.
        y_span = spread if spread > 1e-12 else max(abs(vmax) * 0.02, 0.01)
        annotation_gap = y_span * 0.035
        for xpos, value in zip(x[valid], finite_values):
            ax.annotate(
                f"{value:.4f}",
                (xpos, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=0, ha="center", fontsize=9)
        # Keep the categorical x-axis close to its data range, with small margins.
        ax.set_xlim(-0.18, 2.18)

        # Balanced scale: show the deviation without either compressing it
        # against zero or tightly cropping to the min/max values.
        # The same 35% range-padding rule is applied to every metric.
        if spread > 1e-12:
            y_padding = spread * 0.35
        else:
            y_padding = max(abs(vmax) * 0.05, 0.01)
        lower = max(0.0, vmin - y_padding)
        upper = vmax + y_padding
        if upper <= lower:
            upper = lower + max(abs(vmax) * 0.1, 0.1)
        ax.set_ylim(lower, upper)

        ax.set_xlabel("Dehazing method")
        ax.set_ylabel(metric.replace("_", " "))
        # No title: dataset and metric should be stated in the report caption.
        ax.grid(True, axis="both", linestyle="--", alpha=0.4)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", pad=12)
        fig.tight_layout()

        filename = f"{metric}_line.png"
        fig.savefig(graph_dir / filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

    print(f"Individual line graphs saved in: {graph_dir}")

def evaluate_pairs(dataset_name, hazy_dir, clear_dir, run_dir, classification_map=None):
    pairs, unmatched_hazy, unmatched_clear = match_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise SystemExit("No filename-matched hazy/ground-truth pairs were found.")

    out = run_dir
    original_dir = out / "01_ORIGINAL_BASELINE"
    e3e_dir = out / "02_E3E_REFERENCE"
    e51_dir = out / "03_E5_1"
    routed_dir = out / "04_CNN_ROUTED_OUTPUT"
    for folder in (original_dir, e3e_dir, e51_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if classification_map is not None:
        routed_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitializing PSNR, SSIM, FSIM, CIE76_DeltaE, EPI, Edge Retention, MS-SSIM, BRISQUE, NIQE and PIQE...")
    e51.initialize_extended_metrics()

    header = ["image"]
    for metric in METRIC_NAMES:
        header += [f"{metric}_Original", f"{metric}_E3E", f"{metric}_E5_1"]
    header += ["predicted_class", "haze_score", "routed_action"]
    rows, skipped = [], []

    # Frozen configurations: original baseline vs selected E3-E and E5.1.
    original_config = DehazeConfig()
    e3e_config = e51.E3E_CONFIG

    print(f"Matched pairs: {len(pairs)}")
    for index, (hazy_path, gt_path) in enumerate(pairs, 1):
        try:
            hazy = read_rgb(hazy_path)
            gt = e51.as_u8(read_rgb(gt_path))
            if hazy.shape != gt.shape:
                skipped.append([hazy_path.name, "hazy/GT dimension mismatch"])
                continue

            original = dehaze_image(hazy, original_config)["dehazed"]
            e3e = dehaze_image(hazy, e3e_config)["dehazed"]
            e51_image, alpha = e51.apply_adaptive_chroma(hazy, e3e)

            original_u8 = e51.as_u8(original)
            e3e_u8 = e51.as_u8(e3e)
            e51_u8 = e51.as_u8(e51_image)

            vectors = [
                metric_vector(gt, original_u8),
                metric_vector(gt, e3e_u8),
                metric_vector(gt, e51_u8)
            ]

            predicted_class, haze_score, action = "", "", ""
            if classification_map is not None:
                record = classification_map.get(hazy_path.name)
                if record is None:
                    predicted_class = "MISSING_CLASSIFICATION"
                    routed_u8 = hazy
                    action = "NO_CLASSIFICATION_RECORD"
                else:
                    predicted_class = record.get("classification", "")
                    haze_score = record.get("haze_score", "")
                    if predicted_class == "CLEAR":
                        routed_u8 = hazy
                        action = "BYPASSED_CLEAR"
                    elif predicted_class in {"HAZY", "UNCERTAIN"}:
                        routed_u8 = e51_u8
                        action = "DEHAZED_E5_1"
                    else:
                        routed_u8 = hazy
                        action = "BYPASSED_INVALID_OR_ERROR"
                save_rgb_u8(routed_u8, routed_dir / f"{hazy_path.stem}_routed.png")

            row = [hazy_path.name]
            for metric_index in range(len(METRIC_NAMES)):
                row.extend([vectors[0][metric_index],
                            vectors[1][metric_index],
                            vectors[2][metric_index]])
            row.extend([predicted_class, haze_score, action])
            rows.append(row)

            save_rgb_u8(original_u8, original_dir / f"{hazy_path.stem}.png")
            save_rgb_u8(e3e_u8, e3e_dir / f"{hazy_path.stem}.png")
            save_rgb_u8(e51_u8, e51_dir / f"{hazy_path.stem}.png")

            if index % 10 == 0 or index == len(pairs):
                write_csv(out / "per_image_metrics.csv", header, rows)
                write_csv(out / "skipped.csv", ["image", "reason"], skipped)
                print(f"Progress {index}/{len(pairs)} | successful {len(rows)} | skipped {len(skipped)}")

        except Exception as exc:
            skipped.append([hazy_path.name, repr(exc)])
            print(f"ERROR {hazy_path.name}: {exc}")

    if not rows:
        raise SystemExit("All image pairs failed. Check skipped.csv for details.")

    summary = []
    for metric_index, metric in enumerate(METRIC_NAMES):
        values = []
        for method_index in range(3):
            col = 1 + metric_index * 3 + method_index
            arr = np.asarray([float(r[col]) for r in rows], dtype=float)
            values.append(float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan"))
        summary.append([metric, *values, len(rows)])

    write_csv(
        out / "summary_metrics.csv",
        ["metric", "Original_mean", "E3E_mean", "E5_1_mean", "n"],
        summary
    )
    plot_comparisons(summary, dataset_name, out)

    if classification_map is not None:
        # Summarize classifier routing counts and keep classification CSV separately.
        counts = {}
        for row in rows:
            label = row[-3] or "UNKNOWN"
            counts[label] = counts.get(label, 0) + 1
        write_csv(
            out / "classification_routing_counts.csv",
            ["predicted_class", "image_count"],
            [[k, v] for k, v in sorted(counts.items())]
        )

    write_csv(out / "unmatched_pairs.csv",
              ["unmatched_hazy", "unmatched_ground_truth"],
              [[str(unmatched_hazy[i]) if i < len(unmatched_hazy) else "",
                str(unmatched_clear[i]) if i < len(unmatched_clear) else ""]
               for i in range(max(len(unmatched_hazy), len(unmatched_clear)))])
    print(f"\nEvaluation complete: {out}")
    print(f"Per-image metrics: {out / 'per_image_metrics.csv'}")
    print(f"Summary metrics  : {out / 'summary_metrics.csv'}")
    print(f"Graphs           : {out / 'graphs'}")
    return out


def main():
    print("=" * 68)
    print("CNN CLASSIFICATION + ORIGINAL / E3-E / E5.1 RGB DEHAZING EVALUATION")
    print("=" * 68)
    dataset_name, root, hazy_dir, clear_dir, split = get_dataset()
    print(f"\nDataset: {dataset_name}")
    print(f"Hazy folder: {hazy_dir}")
    print(f"GT folder  : {clear_dir}")

    print("\nChoose how to provide classification:")
    print("  1. Run CNN now, save classifications, then use them")
    print("  2. Reuse an existing classification CSV")
    print("  3. Skip CNN classification; evaluate all paired dataset images")
    mode = input("Choice [1-3]: ").strip()

    if mode not in {"1", "2", "3"}:
        raise SystemExit("Invalid choice.")

    mode_name = {
        "1": "CNN_CLASSIFIED",
        "2": "REUSED_CLASSIFICATION",
        "3": "DIRECT_NO_CLASSIFIER",
    }[mode]

    # A fresh directory is created before any output is written.
    run_dir = create_unique_run_folder(dataset_name, mode_name)
    print("\\nNEW EXPERIMENT FOLDER (previous results are preserved):")
    print(run_dir)

    classification_map = None
    csv_path = None
    if mode == "1":
        csv_path = classify_and_save(hazy_dir, dataset_name, run_dir)
        classification_map = load_classification_csv(csv_path)
    elif mode == "2":
        raw = input("Enter full path to the saved classification CSV: ").strip().strip('"')
        csv_path = Path(raw).expanduser()
        if not csv_path.is_file():
            raise SystemExit(f"Classification CSV not found: {csv_path}")
        classification_map = load_classification_csv(csv_path)
        print(f"Reusing classification CSV: {csv_path}")
    else:
        print("CNN skipped. Every matched hazy/GT pair will be evaluated.")

    # Record the settings and source paths for traceability.
    write_csv(
        run_dir / "run_info.csv",
        ["field", "value"],
        [
            ["dataset", dataset_name],
            ["split", split],
            ["classification_mode", mode_name],
            ["classification_csv", str(csv_path) if csv_path else "Not used"],
            ["hazy_folder", str(hazy_dir)],
            ["ground_truth_folder", str(clear_dir)],
            ["run_started", datetime.now().isoformat(timespec="seconds")],
        ],
    )

    result_dir = evaluate_pairs(
        dataset_name, hazy_dir, clear_dir, run_dir, classification_map
    )
    print("\nALL DONE.")
    print(f"Outputs: {result_dir}")


if __name__ == "__main__":
    main()
