"""Run only E5_ADAPTIVE_RGB on the validation split. Saves a row after every image."""
from experiment_common import run_one

if __name__ == "__main__":
    run_one("E5_ADAPTIVE_RGB", "E5", fsim=True)
