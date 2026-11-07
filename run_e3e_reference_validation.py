"""Run only E3E_REFERENCE on the validation split. Saves a row after every image."""
from experiment_common import run_one

if __name__ == "__main__":
    run_one("E3E_REFERENCE", "E3E", fsim=True)
