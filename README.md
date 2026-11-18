# Adaptive RGB Image Dehazing with CNN-Based Haze Classification

An RGB image dehazing framework implementing the Adaptive Pixel-Wise Transmission Estimation (APTE) method and two proposed chroma-preservation extensions: E3-E and E5.1. The project integrates CNN-based haze classification, RGB image dehazing, and quantitative image-quality evaluation.

## 1. Project Overview

Haze reduces image visibility, contrast, colour fidelity, and structural information. These degradations can negatively affect subsequent computer vision tasks.

This project implements an RGB image dehazing framework based on the research paper:

**VLSI Architecture for an Adaptive Transmission Estimation-Based Video Dehazing Method**

The original APTE algorithm is retained as the baseline. Two additional RGB chroma-preservation methods are incorporated to improve colour fidelity while maintaining the dehazing characteristics of the original algorithm.

The framework supports:

- CNN-based haze classification.
- Original APTE image dehazing.
- Proposed Method 1 – E3-E.
- Proposed Method 2 – E5.1.
- CNN-guided dehazing and clear-image bypass.
- Full-reference and no-reference image-quality evaluation.
- Per-image metric calculation.
- Comparative CSV reports and individual metric graphs.
- Evaluation using CNN_Dataset, I-HAZE, and O-HAZE.

## 2. Methodology

The framework evaluates three dehazing approaches.

### 2.1 Original APTE

The original Adaptive Pixel-Wise Transmission Estimation method is implemented as the baseline.

It performs transmission estimation and atmospheric-scattering-based image restoration to recover visibility from hazy RGB images.

The original algorithm is retained without replacing its underlying dehazing methodology.

### 2.2 Proposed Method 1 – E3-E

E3-E introduces fixed RGB chroma preservation as an additional processing stage.

The chroma-preservation operation is expressed as:

\[
C_{\mathrm{new}} =
(1-\alpha)C_{\mathrm{dehazed}}
+\alpha C_{\mathrm{input}}
\]

where:

- \(C_{\mathrm{input}}\) represents the original RGB chroma information.
- \(C_{\mathrm{dehazed}}\) represents the chroma information after the original dehazing operation.
- \(\alpha\) represents the chroma-preservation strength.

The selected E3-E configuration uses:

\[
\alpha=0.40
\]

This combines the dehazed result with the original image's colour information.

### 2.3 Proposed Method 2 – E5.1

E5.1 extends the E3-E approach using adaptive RGB chroma preservation.

Instead of using a fixed preservation strength, E5.1 calculates a spatially varying preservation parameter:

\[
C_{\mathrm{new}}(x,y)=
(1-\alpha(x,y))C_{\mathrm{E3E}}(x,y)
+\alpha(x,y)C_{\mathrm{input}}(x,y)
\]

The adaptive preservation mechanism incorporates:

- RGB chroma disagreement.
- Luminance-dependent adjustment.
- Saturation-related information.
- Gaussian smoothing of the adaptive preservation map.

The selected E5.1 configuration uses:

| Parameter | Value |
|---|---:|
| Base alpha | 0.40 |
| Alpha gain | 0.20 |
| Minimum alpha | 0.20 |
| Maximum alpha | 0.60 |
| Dark luminance threshold | 0.16 |
| Bright luminance threshold | 0.88 |

E5.1 aims to adapt chroma preservation according to local image characteristics rather than applying the same preservation strength to every pixel.

## 3. System Architecture

The framework supports two processing modes.

### Mode A – CNN-Based Classification and Dehazing

1. Input hazy RGB image.
2. CNN-based image classification.
3. Classification into CLEAR, HAZY, or UNCERTAIN.
4. CLEAR images bypass dehazing.
5. HAZY and UNCERTAIN images undergo E5.1 dehazing.
6. Save classification results and processed images.
7. Evaluate image quality against the corresponding ground truth.

### Mode B – Direct Dehazing Evaluation

1. Load hazy RGB image and corresponding ground-truth image.
2. Apply original APTE.
3. Apply proposed E3-E.
4. Apply proposed E5.1.
5. Calculate evaluation metrics for each method.
6. Save dehazed outputs, per-image metrics, summary metrics, and comparison graphs.

This mode evaluates all matched image pairs without CNN-based routing.

## 4. Repository Structure

```text
adaptive-rgb-image-dehazing-e51/
│
├── classifier/
│   ├── __init__.py
│   ├── predict.py
│   └── model.py
│
├── dehazing/
│   ├── __init__.py
│   └── core.py
│
├── models/
│   ├── best_haze_CNN.pth
│   └── haze_CNN_final.pth
│
├── dataset_config.py
├── dataset_paths.py
├── run_e51_validation_extended.py
├── run_full_pipeline.py
├── train.py
├── test.py
├── .gitignore
└── README.md
```

**Note:** Dataset images and generated experiment outputs are maintained separately and should not be committed to the repository.

## 5. Dataset Organization

The framework supports:

- CNN_Dataset
- I-HAZE
- O-HAZE / O-HAZY

Expected dataset organization:

```text
dataset/
│
├── CNN_Dataset/
│   ├── train/
│   │   ├── hazy/
│   │   └── clear/
│   ├── validation/
│   │   ├── hazy/
│   │   └── clear/
│   └── test/
│       ├── hazy/
│       └── clear/
│
├── I-HAZE/
│   ├── ihaze_hazy/
│   └── ihaze_gt/
│
└── O-HAZY/
    ├── ohaze_hazy/
    └── ohaze_GT/
```

Dataset images are not included in this repository. Obtain the datasets separately and configure their paths using `dataset_config.py`.

The evaluation requires correctly matched hazy and ground-truth image pairs.

## 6. Installation and Environment Setup

### Requirements

- Windows 10 or Windows 11.
- Python 3.10 or a compatible Python environment.
- PyTorch.
- NumPy.
- SciPy.
- OpenCV.
- Pillow.
- Pandas.
- Matplotlib.
- Scikit-image.
- Scikit-learn.
- PyIQA.

### Create a virtual environment

Open Windows PowerShell in the project directory:

```powershell
cd "C:\path\to\adaptive-rgb-image-dehazing-e51"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip

pip install numpy scipy pillow opencv-python matplotlib pandas scikit-image scikit-learn torch torchvision pyiqa
```

Additional packages may be required depending on the installed versions and metric implementations.

## 7. Running the Project

The complete pipeline is executed through:

**`run_full_pipeline.py`**

Run the following command from the project root directory:

```powershell
python run_full_pipeline.py
```

The program provides interactive dataset and classification options.

### Dataset Selection

```text
1. CNN_Dataset
2. I-HAZE
3. O-HAZY / O-HAZE
```

### Classification Selection

```text
1. Run CNN now, save classifications, then use them
2. Reuse an existing classification CSV
3. Skip CNN classification; evaluate all paired dataset images
```

### Recommended execution configurations

**CNN-based classification and dehazing**

- Select the required dataset.
- Choose option 1 to run CNN classification or option 2 to reuse previous classifications.
- The pipeline evaluates APTE, E3-E, and E5.1 and generates the CNN-routed output.

**Direct dehazing evaluation**

- Select the required dataset.
- Choose option 3.
- All matched hazy and ground-truth pairs are evaluated without CNN classification.

## 8. Evaluation Metrics

The framework evaluates image quality using ten metrics.

| Metric | Description | Preferred direction |
|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio | Higher |
| SSIM | Structural Similarity Index | Higher |
| FSIM | Feature Similarity Index | Higher |
| CIE76 DeltaE | Colour difference | Lower |
| EPI | Edge Preservation Index | Higher |
| Edge Retention | Edge preservation measurement | Higher |
| MS-SSIM | Multi-Scale Structural Similarity | Higher |
| BRISQUE | Blind image-quality assessment | Lower |
| NIQE | Natural Image Quality Evaluator | Lower |
| PIQE | Perception-Based Image Quality Evaluator | Lower |

The first seven metrics provide full-reference or reference-based image-quality information. BRISQUE, NIQE, and PIQE provide no-reference image-quality assessments.

Metric interpretation should consider the characteristics and limitations of each metric. No single metric is sufficient to establish overall dehazing performance.

## 9. Output Organization

Every execution creates a separate experiment directory to preserve previous results.

Example:

```text
outputs/
└── full_pipeline_comparison/
    │
    ├── CNN_Dataset_CNN_CLASSIFIED_YYYYMMDD_HHMMSS/
    │   ├── classification/
    │   ├── per_image_metrics.csv
    │   ├── summary_metrics.csv
    │   ├── graphs/
    │   └── dehazed/
    │
    ├── I-HAZE_DIRECT_NO_CLASSIFIER_YYYYMMDD_HHMMSS/
    │   ├── per_image_metrics.csv
    │   ├── summary_metrics.csv
    │   ├── graphs/
    │   └── dehazed/
    │
    └── O-HAZE_DIRECT_NO_CLASSIFIER_YYYYMMDD_HHMMSS/
        ├── per_image_metrics.csv
        ├── summary_metrics.csv
        ├── graphs/
        └── dehazed/
```

The exact generated subfolders depend on the selected execution mode.

### Generated results

- Per-image evaluation metrics.
- Summary metric comparisons.
- CNN classification CSV, when classification is enabled.
- Individual metric comparison graphs.
- Dehazed RGB image outputs.
- Separate timestamped experiment directories.

Previous experiment folders are preserved rather than overwritten.

## 10. Research Evaluation Considerations

For meaningful comparisons:

- Use identical image pairs across APTE, E3-E, and E5.1.
- Keep the evaluation protocol consistent across methods.
- Preserve original image resolutions for primary high-resolution dataset evaluations.
- Report average metrics and per-image results.
- Interpret full-reference and no-reference metrics separately.
- Avoid claiming universal improvement based only on average metric differences.
- Evaluate classification accuracy separately from dehazing quality.
- Distinguish direct dehazing comparisons from CNN-routed pipeline results.

## 11. Current Project Scope

The current implementation focuses on:

1. Reproducing the original APTE RGB dehazing baseline.
2. Introducing fixed RGB chroma preservation through E3-E.
3. Developing adaptive RGB chroma preservation through E5.1.
4. Integrating CNN-based haze classification.
5. Comparing dehazing methods using ten image-quality metrics.
6. Evaluating performance on paired hazy and ground-truth datasets.

The project is intended for research experimentation and comparative evaluation of RGB image dehazing techniques.

## 12. License and Citation

This repository implements a research-oriented image dehazing framework based on the referenced APTE research methodology, with additional proposed RGB chroma-preservation modules.

If you use the implementation or build upon the research, please cite the original APTE research paper:

**VLSI Architecture for an Adaptive Transmission Estimation-Based Video Dehazing Method**

Please also acknowledge the original datasets and relevant metric implementations used in your experiments.

---

**Repository:** `Nuevo1009/adaptive-rgb-image-dehazing-e51`

**Project:** Adaptive RGB Image Dehazing with CNN-Based Haze Classification and Adaptive Chroma Preservation.
