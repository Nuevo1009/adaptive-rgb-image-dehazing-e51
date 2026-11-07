"""Run only E7_DARK_TFLOOR on the validation split. Saves a row after every image."""
from experiment_common import run_one

if __name__ == "__main__":
    run_one("E7_DARK_TFLOOR", "E7", fsim=True)
