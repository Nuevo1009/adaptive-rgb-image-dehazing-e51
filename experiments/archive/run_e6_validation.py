"""Run only E6_GUIDED_TUNE on the validation split. Saves a row after every image."""
from experiment_common import run_one

if __name__ == "__main__":
    run_one("E6_GUIDED_TUNE", "E6", fsim=True)
