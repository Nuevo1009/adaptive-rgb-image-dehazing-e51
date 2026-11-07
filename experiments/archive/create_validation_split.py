from pathlib import Path
import argparse
import random
import shutil
import sys

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Required folder not found: {folder}")
    return {
        p.stem.lower(): p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a paired validation split by MOVING a fraction of image pairs "
            "from train/hazy and train/clear into validation/hazy and validation/clear. "
            "Default is a dry run; pass --apply to move files."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(Path("dataset") / "CNN_Dataset"),
        help="Path to CNN_Dataset (default: dataset/CNN_Dataset, relative to current directory)",
    )
    parser.add_argument("--fraction", type=float, default=0.20,
                        help="Fraction of matched training pairs to move (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually move files. Without this flag, only preview the plan.")
    args = parser.parse_args()

    if not 0.05 <= args.fraction <= 0.50:
        sys.exit("ERROR: --fraction must be between 0.05 and 0.50.")

    root = Path(args.dataset_root).expanduser().resolve()
    train_hazy = root / "train" / "hazy"
    train_clear = root / "train" / "clear"
    val_hazy = root / "validation" / "hazy"
    val_clear = root / "validation" / "clear"

    try:
        hazy = image_files(train_hazy)
        clear = image_files(train_clear)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    common = sorted(set(hazy) & set(clear))
    unmatched_hazy = sorted(set(hazy) - set(clear))
    unmatched_clear = sorted(set(clear) - set(hazy))

    if not common:
        sys.exit(
            "ERROR: No paired hazy/clear images matched by filename stem. "
            "Do not continue until the pairing convention is checked."
        )

    # Avoid mixing into an existing validation split or overwriting files.
    for folder in (val_hazy, val_clear):
        if folder.exists() and any(folder.iterdir()):
            sys.exit(
                f"ERROR: Validation folder is not empty: {folder}\n"
                "This script will not merge with or overwrite an existing split. "
                "Inspect it manually before proceeding."
            )

    rng = random.Random(args.seed)
    shuffled = common[:]
    rng.shuffle(shuffled)
    count = max(1, round(len(shuffled) * args.fraction))
    selected = sorted(shuffled[:count])

    print("=" * 68)
    print("PAIRED TRAIN / VALIDATION SPLIT")
    print("=" * 68)
    print(f"Dataset root       : {root}")
    print(f"Matched train pairs: {len(common)}")
    print(f"Pairs to validation: {len(selected)} ({len(selected)/len(common):.1%})")
    print(f"Pairs left in train: {len(common) - len(selected)}")
    print(f"Unmatched hazy files: {len(unmatched_hazy)}")
    print(f"Unmatched clear files: {len(unmatched_clear)}")
    print(f"Random seed        : {args.seed}")
    print(f"Mode               : {'APPLY (files will be moved)' if args.apply else 'DRY RUN (no files changed)'}")
    print()

    if unmatched_hazy:
        print("Examples of unmatched hazy filenames:")
        for name in unmatched_hazy[:10]:
            print(f"  {hazy[name].name}")
    if unmatched_clear:
        print("Examples of unmatched clear filenames:")
        for name in unmatched_clear[:10]:
            print(f"  {clear[name].name}")

    print("First 10 selected pairs:")
    for stem in selected[:10]:
        print(f"  hazy: {hazy[stem].name}  |  clear: {clear[stem].name}")

    if not args.apply:
        print()
        print("Nothing has been changed.")
        print("If the pairing and selected files look correct, rerun with --apply.")
        return

    # Create destination folders only after all checks and preview details.
    val_hazy.mkdir(parents=True, exist_ok=True)
    val_clear.mkdir(parents=True, exist_ok=True)

    moved = []
    try:
        for stem in selected:
            src_h, dst_h = hazy[stem], val_hazy / hazy[stem].name
            src_c, dst_c = clear[stem], val_clear / clear[stem].name
            if dst_h.exists() or dst_c.exists():
                raise FileExistsError(f"Destination already exists for pair: {stem}")
            shutil.move(str(src_h), str(dst_h))
            moved.append((src_h, dst_h))
            shutil.move(str(src_c), str(dst_c))
            moved.append((src_c, dst_c))
    except Exception as exc:
        print(f"\nERROR while moving files: {exc}")
        print("Attempting to roll back files already moved...")
        for original, destination in reversed(moved):
            try:
                if destination.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(destination), str(original))
            except Exception as rollback_exc:
                print(f"ROLLBACK WARNING: {destination} -> {original}: {rollback_exc}")
        sys.exit("Split was not completed cleanly; inspect the folders before retrying.")

    print()
    print("SUCCESS: validation split created.")
    print(f"Validation hazy: {val_hazy}")
    print(f"Validation clear: {val_clear}")
    print(f"Moved pairs     : {len(selected)}")
    print(f"Training pairs remaining: {len(common) - len(selected)}")
    print()
    print("Your test folder was not touched.")
    print("Keep the validation split fixed while comparing dehazing configurations.")


if __name__ == "__main__":
    main()
