# Experimental results and interpretation

## E3-E versus E5.1

Reported aggregate validation means over 1,200 paired images:

| Metric | E3-E | E5.1 | Difference |
|---|---:|---:|---:|
| PSNR (dB) | 19.709367 | 19.743819 | +0.034452 |
| SSIM | 0.821066 | 0.823807 | +0.002741 |
| FSIM | 0.926025 | 0.926175 | +0.000150 |
| CIE76 DeltaE | 11.798564 | 11.676600 | -0.121964 |
| EPI | 0.907286 | 0.907465 | +0.000179 |
| Edge retention | 0.875162 | 0.874515 | -0.000647 |

PSNR, SSIM, FSIM and EPI are higher-is-better; DeltaE is lower-is-better.
Edge retention is a ratio and should be interpreted alongside visual results and EPI,
not optimized in isolation.

## Interpretation

The reported averages favour E5.1 on five of six measures, but the PSNR gain is
small (+0.034452 dB). This supports describing E5.1 as a modest refinement, not a
large or universally beneficial improvement. Per-image results should accompany
aggregate means to show how often the candidate helps or hurts.

## Evidence currently present in the release package

The project ZIP supplied for cleanup contained source scripts and model checkpoints,
but no original validation CSVs, plots or sample outputs. The aggregate table above
was already reported during the project conversation. It has been retained as a
reported result, but must be paired with the original CSV and regenerated plots
before an external reader can independently audit it.

The test split was used in earlier development iterations; it should not be described
as an untouched final test set.
