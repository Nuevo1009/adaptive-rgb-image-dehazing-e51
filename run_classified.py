from pathlib import Path
import csv
import shutil

import numpy as np
from PIL import Image

from classifier.predict import load_model, classify_image
from dehazing.core import DehazeConfig, dehaze_image, read_rgb
from settings import PROJECT_ROOT

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "models" / "best_haze_CNN.pth"

# Separate output directory prevents overwriting earlier baseline/E3-E runs.
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cnn_guided_dehazing_e3e_selected"
CLEAR_OUTPUT = OUTPUT_ROOT / "clear_bypassed"
DEHAZED_OUTPUT = OUTPUT_ROOT / "dehazed"
LOG_PATH = OUTPUT_ROOT / "classification_results.csv"

# Selected validation configuration: adaptive airlight + guided refinement
# (r=30, eps=0.015) + RGB chroma preservation (alpha=0.40).
# The paper's original atmospheric, transmission and reconstruction
# equations remain unchanged. Use DehazeConfig() for the baseline instead.

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def choose_input_folder():
    """Ask the user for a dataset name and return its hazy-image folder."""
    dataset_root = PROJECT_ROOT / "dataset"

    datasets = {
        "CNN_DATASET": dataset_root / "CNN_Dataset",
        "I-HAZE": dataset_root / "I-HAZE",
        "O-HAZY": dataset_root / "O-HAZY",
    }

    print("\nSelect dataset:")
    print("1. CNN_Dataset")
    print("2. I-HAZE")
    print("3. O-HAZY")

    choice = input(
        "\nEnter dataset name or number: "
    ).strip().upper()

    aliases = {
        "1": "CNN_DATASET",
        "CNN_DATASET": "CNN_DATASET",
        "CNN DATASET": "CNN_DATASET",
        "2": "I-HAZE",
        "I-HAZE": "I-HAZE",
        "IHAZE": "I-HAZE",
        "3": "O-HAZY",
        "O-HAZY": "O-HAZY",
        "OHAZY": "O-HAZY",
    }

    key = aliases.get(choice)
    if key is None:
        raise SystemExit(
            "Invalid choice. Enter CNN_Dataset, I-HAZE, or O-HAZY."
        )

    dataset_path = datasets[key]
    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset folder not found: {dataset_path}")

    if key == "CNN_DATASET":
        print("\nSelect split:")
        print("1. train")
        print("2. test")
        split = input("Enter train or test: ").strip().lower()

        if split not in ("train", "test"):
            raise SystemExit("Invalid split. Enter train or test.")

        source_path = dataset_path / split / "hazy"
    else:
        hazy_folder_names = {
            "I-HAZE": "ihaze_hazy",
            "O-HAZY": "ohaze_hazy",
        }
        source_path = dataset_path / hazy_folder_names[key]

    if not source_path.is_dir():
        raise SystemExit(f"Hazy folder not found: {source_path}")

    print(f"\nSelected dataset: {key}")
    print(f"Input folder: {source_path}")
    return source_path


def save_rgb_float(image, output_path):
    """Save an RGB image returned by the dehazing algorithm."""
    image = np.asarray(image)

    if image.dtype != np.uint8:
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image + 0.5, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def process_image(image_path, model, device, config, source_root):
    """Classify one image and route it to the correct branch."""
    label, score, memberships = classify_image(
        image_path, model, device
    )

    # Preserve relative subfolders if the source contains duplicate filenames.
    try:
        relative_path = image_path.relative_to(source_root)
    except ValueError:
        relative_path = Path(image_path.name)

    if label == "CLEAR":
        output_path = CLEAR_OUTPUT / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_path)
        action = "BYPASSED"

    elif label in ("UNCERTAIN", "HAZY"):
        output_path = (
            DEHAZED_OUTPUT
            / relative_path.parent
            / f"{relative_path.stem}_dehazed.png"
        )

        # Use the original-resolution image for dehazing.
        rgb = read_rgb(image_path)
        result = dehaze_image(rgb, config)
        save_rgb_float(result["dehazed"], output_path)
        action = "DEHAZED"

    else:
        raise ValueError(f"Unexpected classifier label: {label}")

    print(
        f"{image_path.name} | "
        f"score={score:.4f} | "
        f"class={label} | "
        f"action={action}"
    )

    return {
        "filename": str(relative_path),
        "haze_score": score,
        "classification": label,
        "clear_membership": memberships["clear"],
        "uncertain_membership": memberships["uncertain"],
        "hazy_membership": memberships["hazy"],
        "action": action,
        "output_path": str(output_path),
    }


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"CNN model not found:\n{MODEL_PATH}\n"
            "Check MODEL_PATH and your checkpoint filename."
        )

    print("CNN-guided image dehazing")
    print("-------------------------")

    source_path = choose_input_folder()

    images = sorted(
        p for p in source_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise SystemExit(f"No supported images found in: {source_path}")

    print(f"Images found: {len(images)}")

    CLEAR_OUTPUT.mkdir(parents=True, exist_ok=True)
    DEHAZED_OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load the CNN once, not once per image.
    model, device = load_model(MODEL_PATH)

    config = DehazeConfig.e3e_selected()
    print(
        "\nSelected dehazing preset: E3-E validation candidate | "
        f"GF radius={config.guided_filter_radius}, "
        f"epsilon={config.guided_filter_eps}, "
        f"RGB chroma alpha={config.color_preserve_strength}"
    )

    results = []

    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}]", end=" ")

        try:
            results.append(
                process_image(
                    image_path, model, device, config, source_path
                )
            )
        except Exception as exc:
            print(f"\n[ERROR] {image_path.name}: {exc}")
            results.append({
                "filename": str(image_path),
                "haze_score": "",
                "classification": "ERROR",
                "clear_membership": "",
                "uncertain_membership": "",
                "hazy_membership": "",
                "action": "FAILED",
                "output_path": "",
            })

    columns = [
        "filename",
        "haze_score",
        "classification",
        "clear_membership",
        "uncertain_membership",
        "hazy_membership",
        "action",
        "output_path",
    ]

    with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print("\nProcessing complete.")
    print(f"Images processed: {len(results)}")
    print(f"Clear images bypassed: {sum(r['action'] == 'BYPASSED' for r in results)}")
    print(f"Images dehazed: {sum(r['action'] == 'DEHAZED' for r in results)}")
    print(f"Failed images: {sum(r['action'] == 'FAILED' for r in results)}")
    print(f"Results CSV: {LOG_PATH}")


if __name__ == "__main__":
    main()