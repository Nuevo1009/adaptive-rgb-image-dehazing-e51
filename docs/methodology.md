# Methodology

## Pipeline

The selected processing chain is:

1. Read hazy RGB input and normalize it to floating point `[0, 1]`.
2. Run the E3-E dehazing preset in `dehazing/core.py`.
3. Compute an adaptive confidence map from the hazy input RGB and E3-E output.
4. Convert the input and E3-E output into luminance/chroma representations using the
   implementation's RGB weighted luminance.
5. Blend the E3-E chroma with input chroma using a spatially varying alpha.
6. Recombine with E3-E luminance, clip to `[0, 1]`, and evaluate.

E5.1 is a post-processing extension to E3-E; it does not alter the atmospheric-light,
transmission-estimation or physical reconstruction stages.

## E5.1 confidence construction

In `run_e51_validation.py`, let `C_src = RGB_src - Y_src` and
`C_out = RGB_E3E - Y_E3E`, where `Y` uses weights `[0.299, 0.587, 0.114]`.

The implementation computes:
- mean absolute chroma disagreement, normalized by `0.15` and clipped to `[0,1]`;
- a dark-luminance gate based on `0.16` and a transition width of `0.18`;
- a bright-luminance gate based on `0.88` and a transition width of `0.12`;
- a saturation gate from the input chroma norm, normalized by `0.20`;
- confidence = disagreement confidence × luminance gate × `(0.65 + 0.35 × saturation gate)`;
- alpha = `0.40 + 0.20 × (confidence - 0.35)`, clipped to `[0.20, 0.60]`;
- Gaussian smoothing with sigma `2.0`, followed by clipping to `[0.20, 0.60]`.

The exact source implementation is authoritative.

## Adaptive chroma reconstruction

Let `alpha(x,y)` be the E5.1 map. The source code computes:

`C_new = (1 - alpha) * C_E3E + alpha * C_input`

and combines this with E3-E luminance:

`RGB_E5.1 = clip(Y_E3E + C_new, 0, 1)`

This is RGB processing; it does not convert the image to YCbCr.
