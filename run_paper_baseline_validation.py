"""Run only PAPER_BASELINE on the validation split. Saves a row after every image."""
from experiment_common import run_one

if __name__ == "__main__":
    run_one("PAPER_BASELINE", "PAPER_BASELINE", fsim=True)
