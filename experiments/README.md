# Experimental archive

These scripts document exploratory versions and ablations. They are not the selected
release algorithm. Several scripts depend on intermediate CSVs, generated images, or
root-level helper modules that are not included in this archive. Treat this folder as
research history, not as a guaranteed one-command runnable suite.

Important reproducibility note: `run_e51_validation.py` explicitly constructs its
E3-E configuration with guided-filter radius 20 and epsilon 1e-3. The older
`experiment_common.py` / `run_e3e_reference_validation.py` path obtains the default
E3-E preset (radius 30 and epsilon 0.015 in `dehazing/core.py`). Therefore that older
standalone reference runner is not numerically equivalent to the E3-E branch inside
the canonical E5.1 comparison. Use the E3-E rows produced by `run_e51_validation.py`
as the matching E3-E reference for the reported E5.1 comparison unless the configs
are deliberately unified and the validation is rerun.

The later E5.3/E5.4 comparison tables were previously affected by incorrect inherited
E5.1 values. Do not treat those inherited columns as authoritative. The original
`run_e51_validation.py` and its original per-image CSV are the authority for E5.1.

The original raw E5.1 per-image CSV and generated output images were not present in
the uploaded project ZIP. Scripts depending on missing intermediate artifacts cannot
be independently reproduced from this release package alone.
