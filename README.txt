UNIVERSAL DATASET SELECTION FOR IMAGE DEHAZING

Files:
- dataset_config.py: prompts for dataset and detects its folders/pairs.
- gamma_ablation.py: evaluates gamma values [1.0, 1.5, 2.0, 2.5].
- evaluate.py: evaluates the current GAMMA setting on paired images.
- run_batch.py: dehazes every image in the selected hazy folder.

HOW TO USE
1. Copy these four .py files into the root of your existing image_dehazing_python project,
   beside settings.py and the dehazing/ folder. Back up your old files first.
2. Keep your existing dehazing/core.py.
3. Ensure settings.py defines these values (retain any additional settings needed by other scripts):
   GAMMA = 2.0
   OMEGA = 0.9375
   DOWNSAMPLE_FACTOR = 3
   PATCH_SIZE = 5
   T_MIN = 0.05
4. Install requirements if needed:
   pip install numpy opencv-python scikit-image
5. Run any of:
   python run_batch.py
   python evaluate.py
   python gamma_ablation.py
6. Choose the dataset from the prompt.

EXPECTED FOLDER STRUCTURES
CNN_Dataset/ or CNN Dataset/
  train/hazy, train/clear, test/hazy, test/clear
I-HAZE/
  ihaze_hazy, ihaze_gt
O-HAZY/ or O-HAZE/
  ohaze_hazy, ohaze_GT

IMPORTANT PAIRING NOTE
The matcher uses normalized filename stems and refuses to guess based only on sort order.
If zero pairs match, compare hazy and ground-truth filenames and adjust the normalization
in dataset_config.py if your dataset uses a different naming convention.

OUTPUTS
- outputs/batch_dehazed/<dataset>/<split>/
- outputs/evaluation/<dataset>/<split>/evaluation_results.csv
- outputs/gamma_ablation/<split>/gamma_results.csv

This package relies on your existing dehazing/core.py API:
DehazeConfig(...), dehaze_image(read_rgb(path), cfg)["dehazed"], read_rgb(path).
It has not been tested against your local files, so run on a small dataset first.
