from pathlib import Path
import argparse

def main():
    p = argparse.ArgumentParser(description="Print commands for this project and CNN Dataset.")
    p.add_argument("--dataset-root", required=True,
                   help='Path to the "CNN Dataset" folder (the folder containing train/ and test/)')
    args = p.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    for split in ("train", "test"):
        for kind in ("hazy", "clear"):
            folder = root / split / kind
            print(f"{split:5} {kind:5}: {folder}  [{'OK' if folder.is_dir() else 'MISSING'}]")
    project = Path(__file__).resolve().parent
    test_hazy = root / "test" / "hazy"
    test_clear = root / "test" / "clear"
    train_hazy = root / "train" / "hazy"
    train_clear = root / "train" / "clear"
    print("\nRun these from the project directory:")
    print(f'python run_batch.py --input-dir "{test_hazy}" --output-dir outputs/test_dehazed')
    print(f'python evaluate.py --hazy-dir "{test_hazy}" --gt-dir "{test_clear}" --output-dir outputs/test_evaluation/dehazed --csv outputs/test_evaluation/metrics.csv')
    print(f'python evaluate.py --hazy-dir "{train_hazy}" --gt-dir "{train_clear}" --output-dir outputs/train_evaluation/dehazed --csv outputs/train_evaluation/metrics.csv')
    print(f'python gamma_ablation.py --hazy-dir "{train_hazy}" --gt-dir "{train_clear}" --gammas 1.0 1.5 2.0 2.5 --csv outputs/train_ablation/gamma_results.csv')
    print("\nDo not use the test set to choose gamma if you want an unbiased final test result.")

if __name__ == "__main__":
    main()
