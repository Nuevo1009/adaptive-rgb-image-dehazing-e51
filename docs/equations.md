# Equations and implementation notes

## RGB luminance and chroma

For normalized RGB values:

`Y = 0.299 R + 0.587 G + 0.114 B`

`C = RGB - Y`

The same luminance weights are used by the E3-E and E5.1 colour-preservation code.

## E3-E chroma preservation

For constant strength `alpha_E3E = 0.40`:

`C_E3E,new = (1 - alpha_E3E) C_dehazed + alpha_E3E C_input`

The implementation recombines this chroma with the dehazed luminance and clips RGB to
`[0, 1]`.

## E5.1 confidence and adaptive alpha

Define:

- `D = clip(mean(abs(C_input - C_E3E)) / 0.15, 0, 1)`
- `G_dark = clip((Y_input - 0.16) / 0.18, 0, 1)`
- `G_bright = clip((0.88 - Y_input) / 0.12, 0, 1)`
- `S = clip(norm(C_input) / 0.20, 0, 1)`

Then:

`confidence = D * G_dark * G_bright * (0.65 + 0.35 S)`

`alpha_raw = clip(0.40 + 0.20 (confidence - 0.35), 0.20, 0.60)`

The alpha map is Gaussian-smoothed with sigma `2.0` and clipped again to `[0.20, 0.60]`.

## E5.1 output

`C_E5.1 = (1 - alpha) C_E3E + alpha C_input`

`RGB_E5.1 = clip(Y_E3E + C_E5.1, 0, 1)`

## Atmospheric dehazing reconstruction

The core implementation reconstructs scene radiance using the atmospheric scattering
form:

`J(x) = (I(x) - A(x)) / max(t(x), t_min) + A(x)`

The precise atmospheric-light map, transmission and correction-factor calculations
are implemented in `dehazing/core.py`. Consult that source and the original paper for
the exact equation-to-code mapping. `t_min=0.05` is documented in the code as a
numerical safeguard, not a paper-specified parameter.
