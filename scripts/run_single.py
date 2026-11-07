import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from pathlib import Path
import cv2

from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dataset_config import select_dataset, IMAGE_EXTS, PROJECT_ROOT
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, save_rgb


def main():
    # ---------------------------------------------------------
    # 1. Select dataset using dataset_config.py
    # ---------------------------------------------------------
    root, hazy_dir, clear_dir, split = select_dataset()

    # ---------------------------------------------------------
    # 2. Discover available hazy images
    # ---------------------------------------------------------
    images = sorted(
        p for p in hazy_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        raise SystemExit(f"No images found in: {hazy_dir}")

    # ---------------------------------------------------------
    # 3. Let the user select one image
    # ---------------------------------------------------------
    print("\nAvailable hazy images:")
    display_limit = min(30, len(images))

    for i, image_path in enumerate(images[:display_limit], start=1):
        print(f"{i}. {image_path.name}")

    choice = input(
        f"\nEnter image number (1-{display_limit}): "
    ).strip()

    try:
        index = int(choice) - 1
        if index < 0 or index >= display_limit:
            raise ValueError
    except ValueError:
        raise SystemExit("Invalid image number.")

    src = images[index]

    # ---------------------------------------------------------
    # 4. Configure and execute the dehazing algorithm
    # ---------------------------------------------------------
    cfg = DehazeConfig(
        DOWNSAMPLE_FACTOR,
        PATCH_SIZE,
        OMEGA,
        GAMMA,
        T_MIN
    )

    print(f"\nProcessing: {src.name}")
    result = dehaze_image(read_rgb(src), cfg)

    # ---------------------------------------------------------
    # 5. Create output directory
    # ---------------------------------------------------------
    output_dir = PROJECT_ROOT / "outputs" / "single" / root.name / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 6. Save dehazing outputs
    # ---------------------------------------------------------
    dehazed_path = output_dir / f"{src.stem}_dehazed.png"
    atmospheric_path = output_dir / f"{src.stem}_atmospheric.png"
    transmission_path = output_dir / f"{src.stem}_transmission.png"

    save_rgb(dehazed_path, result["dehazed"])
    save_rgb(atmospheric_path, result["atmospheric_map"])

    transmission = (
        result["transmission_map"] * 255
    ).round().clip(0, 255).astype("uint8")

    if not cv2.imwrite(str(transmission_path), transmission):
        raise SystemExit(f"Could not save transmission map: {transmission_path}")

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------
    print("\n" + "=" * 55)
    print("SINGLE-IMAGE DEHAZING COMPLETED")
    print("=" * 55)
    print(f"Dataset root       : {root}")
    print(f"Dataset split      : {split}")
    print(f"Input image        : {src}")
    print(f"Dehazed image      : {dehazed_path}")
    print(f"Atmospheric map    : {atmospheric_path}")
    print(f"Transmission map   : {transmission_path}")
    print(f"A_far RGB          : {result['A_far']}")
    print(f"A_near RGB         : {result['A_near']}")
    print(f"Mean gray threshold: {result['threshold']:.6f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
