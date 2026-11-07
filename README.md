# Adaptive RGB Image Dehazing — E5.1

Python research repository for a still-image dehazing pipeline based on the paper
**“VLSI Architecture for an Adaptive Transmission Estimation-Based Video Dehazing Method.”**

The selected candidate is **E5.1: Confidence-Guided Adaptive RGB Chroma Preservation**.
E5.1 is a post-processing extension of E3-E. It retains the E3-E dehazing output and
adds a spatially adaptive RGB-chroma blend. It does not replace atmospheric-light,
transmission-estimation, or physical reconstruction stages.

> Scope: Python implementation and evaluation only. Hardware/FPGA implementation is
> intentionally outside this repository release.

## Repository contents

- `dehazing/core.py` — shared dehazing implementation and E3-E preset.
- `run_e51_validation.py` — canonical validation comparison of E3-E and E5.1.
- `run_paper_baseline_validation.py` — paper/baseline validation runner.
- `dataset_config.py` — dataset discovery and filename-based pair matching.
- `classifier/`, `train.py`, `run_classified.py` — optional CNN classification pipeline.
- `evaluate_*.py` — evaluation utilities.
- `experiments/archive/` — exploratory/ablation scripts, not the selected method.
- `docs/` — methodology, equations, evaluation notes and limitations.
- `results/` — summary of the reported validation averages.

## Selected methods

1. **Paper baseline** — original dehazing configuration in `DehazeConfig`.
2. **E3-E** — enhanced RGB reference preset in `DehazeConfig.e3e_selected()`.
3. **E5.1** — applies confidence-guided adaptive RGB chroma preservation to E3-E output.

The canonical E5.1 validation runner explicitly constructs its E3-E configuration
with guided-filter radius 20 and epsilon 1e-3. Keep that configuration unchanged
when reproducing the reported E5.1 validation result. See `experiments/README.md`
for a mismatch in an older standalone reference runner.

## Installation

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependencies:

```powershell
pip install -r requirements-optional-fsim.txt
pip install -r requirements-optional-classifier.txt
```

Some archived experiments may also require:

```powershell
pip install -r requirements-experiments.txt
```

## Dataset layout

The validation runner expects:

```text
dataset/
└── CNN_Dataset/
    └── validation/
        ├── hazy/
        └── clear/
```

The dataset is **not included**. Obtain it from its original provider and follow its
licence and terms. `dataset_config.py` matches pairs by normalized filename stems;
it does not pair images merely by sorted order.

The default paths are configured in the validation scripts. If your dataset is
stored elsewhere, update the paths carefully and record any changes.

## Run E5.1 validation

From the repository root:

```powershell
python run_e51_validation.py
```

The script writes per-image metrics and dehazed examples under its configured output
directory. Do not commit the full dataset or large generated output directories.
Commit only selected, clearly labelled examples and the CSV/plots needed to support
a release.

## Reported validation averages

The previously reported 1,200-pair validation comparison:

| Metric | E3-E | E5.1 | Preferred direction |
|---|---:|---:|---|
| PSNR (dB) | 19.709367 | 19.743819 | Higher |
| SSIM | 0.821066 | 0.823807 | Higher |
| FSIM | 0.926025 | 0.926175 | Higher |
| CIE76 DeltaE | 11.798564 | 11.676600 | Lower |
| EPI | 0.907286 | 0.907465 | Higher |
| Edge retention | 0.875162 | 0.874515 | Interpret with care |

E5.1 improves five of these six reported averages, with a small reduction in edge
retention. The PSNR gain is modest (+0.034452 dB).

**Reproducibility qualification:** The uploaded project ZIP did not contain the
original per-image validation CSVs or generated comparison plots. This table records
previously reported aggregate values; it is not a substitute for raw per-image
results. Add the original validated CSV and plots before claiming that this repository
independently reproduces the table.

The test split was used during earlier development iterations. Do not describe it as
an untouched independent test set. Prefer a genuinely independent dataset for a
future generalisation evaluation.

## Experimental archive

`experiments/archive/` contains E5.2–E5.4, E6/E7 and other exploratory scripts.
They are retained for research traceability and are **not selected release methods**.
Some archived scripts depend on intermediate CSV files or outputs not included here.
See `experiments/README.md` before running them. The later E5.3/E5.4 comparison
tables were affected by incorrect inherited E5.1 values; do not treat those columns
as authoritative.

## CNN classifier

Classifier source is retained, but model checkpoint files (`.pth`) are not included.
Train a checkpoint using `train.py` or provide a compatible checkpoint and update
the path in the runtime configuration. E5.1 dehazing validation does not require
the classifier.

## Citation and attribution

Cite the original research paper when using the baseline algorithm. Add the complete
author list, venue, year and DOI after checking the paper's title page. Cite the
dataset provider according to its terms.

## Limitations

- E5.1's reported average improvements are small, not a claim of universal
  per-image improvement.
- Edge retention is slightly lower than E3-E in the reported validation summary.
- The confidence map is a heuristic derived from the input and E3-E output; it is
  not semantic understanding or a ground-truth colour estimator.
- The reported aggregate table requires the original raw CSVs for independent audit.
