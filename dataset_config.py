"""Interactive dataset discovery for CNN Dataset, I-HAZE, and O-HAZE.

Place this file in the project root beside dehazing/ and the run scripts.
"""
from pathlib import Path
import re

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PARENT = PROJECT_ROOT / "dataset"


def _existing_child(parent: Path, candidates):
    if not parent.is_dir():
        return None
    children = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for candidate in candidates:
        if candidate.lower() in children:
            return children[candidate.lower()]
    return None


def _image_files(folder: Path):
    if not folder or not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _normalise_stem(path: Path) -> str:
    """Normalize common haze/ground-truth prefixes and suffixes for pair matching."""
    name = path.stem.lower().strip()
    # Remove common dataset tags only at the start/end, not in the middle.
    patterns = [r"^(hazy|haze|gt|clear|groundtruth|ground_truth)[_\- ]*",
                r"[_\- ]*(hazy|haze|gt|clear|groundtruth|ground_truth)$"]
    previous = None
    while previous != name:
        previous = name
        for pattern in patterns:
            name = re.sub(pattern, "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def match_pairs(hazy_dir: Path, clear_dir: Path):
    """Return (pairs, unmatched_hazy, unmatched_clear); never guesses by sort order."""
    hazy_files = _image_files(hazy_dir)
    clear_files = _image_files(clear_dir)
    clear_by_stem = {}
    for p in clear_files:
        clear_by_stem.setdefault(_normalise_stem(p), []).append(p)

    pairs, unmatched_hazy, used_clear = [], [], set()
    for hp in hazy_files:
        candidates = clear_by_stem.get(_normalise_stem(hp), [])
        candidates = [p for p in candidates if p not in used_clear]
        if len(candidates) == 1:
            gp = candidates[0]
            pairs.append((hp, gp))
            used_clear.add(gp)
        elif len(candidates) == 0:
            unmatched_hazy.append(hp)
        else:
            # Ambiguous matching is unsafe: don't silently pair the wrong images.
            unmatched_hazy.append(hp)

    unmatched_clear = [p for p in clear_files if p not in used_clear]
    return pairs, unmatched_hazy, unmatched_clear


def _select_dataset_root():
    print("\nSelect dataset:")
    print("  1. CNN_Dataset (train/test/hazy and clear)")
    print("  2. I-HAZE (ihaze_hazy and ihaze_gt)")
    print("  3. O-HAZE (ohaze_hazy and ohaze_GT)")
    print("  4. Enter a dataset root path manually")
    choice = input("Choice [1-4]: ").strip()

    if choice == "1":
        root = DEFAULT_DATASET_PARENT / "CNN_Dataset"
        if not root.is_dir():
            root = DEFAULT_DATASET_PARENT / "CNN Dataset"
        split = input("Split [train/test] (default test): ").strip().lower() or "test"
        if split not in {"train", "test"}:
            raise SystemExit("Split must be 'train' or 'test'.")
        split_root = root / split
        hazy = _existing_child(split_root, ["hazy"])
        clear = _existing_child(split_root, ["clear", "gt", "ground_truth"])
        return root, hazy, clear, split

    if choice == "2":
        root = DEFAULT_DATASET_PARENT / "I-HAZE"
        if not root.is_dir():
            root = Path(input("Enter full path to I-HAZE folder: ").strip().strip('"'))
        hazy = _existing_child(root, ["ihaze_hazy", "hazy"])
        clear = _existing_child(root, ["ihaze_gt", "ihaze_GT", "gt", "clear"])
        return root, hazy, clear, "all"

    if choice == "3":
        root = DEFAULT_DATASET_PARENT / "O-HAZY"
        if not root.is_dir():
            root = DEFAULT_DATASET_PARENT / "O-HAZE"
        if not root.is_dir():
            root = Path(input("Enter full path to O-HAZE/O-HAZY folder: ").strip().strip('"'))
        hazy = _existing_child(root, ["ohaze_hazy", "ohazy_hazy", "hazy"])
        clear = _existing_child(root, ["ohaze_gt", "ohazy_gt", "gt", "clear"])
        return root, hazy, clear, "all"

    if choice == "4":
        root = Path(input("Enter full path to dataset root folder: ").strip().strip('"')).expanduser()
        if not root.is_dir():
            raise SystemExit(f"Dataset folder not found: {root}")
        # Discover paired-folder dataset layouts.
        for hazy_name in ["ihaze_hazy", "ohaze_hazy", "ohazy_hazy", "hazy"]:
            hazy = _existing_child(root, [hazy_name])
            if hazy:
                clear = _existing_child(root, ["ihaze_gt", "ohaze_gt", "ohazy_gt", "gt", "clear", "ground_truth"])
                return root, hazy, clear, "all"
        # CNN Dataset layout
        split = input("CNN dataset split [train/test] (default test): ").strip().lower() or "test"
        split_root = root / split
        return root, _existing_child(split_root, ["hazy"]), _existing_child(split_root, ["clear", "gt"]), split

    raise SystemExit("Invalid choice. Run again and choose 1, 2, 3, or 4.")


def select_dataset():
    root, hazy_dir, clear_dir, split = _select_dataset_root()
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}\nEdit the paths or choose option 4.")
    if hazy_dir is None or not hazy_dir.is_dir():
        raise SystemExit(f"Could not find the hazy-image folder under: {root}")
    if clear_dir is None or not clear_dir.is_dir():
        raise SystemExit(f"Could not find the clear/ground-truth folder under: {root}")
    print(f"\nDataset root : {root}")
    print(f"Hazy images  : {hazy_dir}")
    print(f"Clear images : {clear_dir}")
    return root, hazy_dir, clear_dir, split
