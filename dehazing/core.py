from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass
class DehazeConfig:
    downsample_factor: int = 3
    patch_size: int = 5
    omega: float = 0.9375
    gamma: float = 2.0
    t_min: float = 0.05  # Numerical safeguard; not a paper-specified parameter.

    # Research experiment. False preserves the original atmospheric-light estimator.
    adaptive_airlight: bool = False
    adaptive_strength: float = 0.75
    candidate_fraction: float = 0.01
    near_white_threshold: float = 0.88
    bright_threshold: float = 0.92
    large_region_fraction: float = 0.08
    low_texture_threshold: float = 0.025
    large_region_weight: float = 0.25

    # Optional experiment: edge-aware refinement of the transmission map.
    # Disabled by default so the original algorithm remains reproducible.
    refine_transmission: bool = False
    guided_filter_radius: int = 20
    guided_filter_eps: float = 1e-3

    # E2 experiment: spatially adaptive omega. Disabled by default.
    adaptive_omega: bool = False
    omega_min: float = 0.80
    omega_max: float = 0.98
    omega_haze_gain: float = 0.08
    omega_contrast_gain: float = 0.08
    omega_contrast_radius: int = 7

    # E3 experiment: blend input RGB chroma with dehazed RGB chroma.
    # Disabled by default; strength=0 preserves the original output.
    color_preserve_strength: float = 0.0

    @classmethod
    def e3e_selected(cls, **overrides):
        """Return the selected E3-E validation configuration.

        The default dataclass configuration above remains the reproducible
        paper/baseline path. This named preset enables the validated optional
        extensions without changing the paper's atmospheric, transmission,
        or reconstruction equations.
        """
        values = {
            "downsample_factor": 3,
            "patch_size": 5,
            "omega": 0.9375,
            "gamma": 2.0,
            "t_min": 0.05,
            "adaptive_airlight": True,
            "adaptive_strength": 0.75,
            "large_region_weight": 0.25,
            "refine_transmission": True,
            "guided_filter_radius": 30,
            "guided_filter_eps": 0.015,
            "adaptive_omega": False,
            "color_preserve_strength": 0.40,
        }
        values.update(overrides)
        return cls(**values)


def validate_rgb(image):
    if image is None:
        raise ValueError("Could not read image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel color image.")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= 255.0
    return np.clip(image, 0, 1)


def gray(rgb):
    return cv2.cvtColor(
        np.ascontiguousarray(rgb.astype(np.float32)),
        cv2.COLOR_RGB2GRAY
    )


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")


def dark_channel(rgb, patch_size=5):
    # Baseline Equation (2): minimum across RGB, then local patch minimum.
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")
    per_pixel_min = np.min(rgb, axis=2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size)
    )
    return cv2.erode(
        per_pixel_min, kernel, borderType=cv2.BORDER_REPLICATE
    )


def estimate_airlight(rgb_down, dark):
    """Original baseline estimator; kept unchanged for controlled comparison."""
    g = gray(rgb_down)
    threshold = float(np.mean(g))
    far = g >= threshold
    near = ~far

    def pick(mask):
        if not np.any(mask):
            y, x = np.unravel_index(int(np.argmax(dark)), dark.shape)
            return rgb_down[y, x].copy()
        candidates = np.where(mask, dark, -np.inf)
        y, x = np.unravel_index(int(np.argmax(candidates)), dark.shape)
        return rgb_down[y, x].copy()

    return (
        pick(far).astype(np.float32),
        pick(near).astype(np.float32),
        threshold
    )


def _large_uniform_near_white_mask(rgb, cfg):
    """
    Experimental diagnostic mask. A component is marked only if it is:
    near-white, sufficiently large, and locally low-texture.
    It is a reliability cue, not proof that a region is not atmospheric light.
    """
    g = gray(rgb)
    near_white = (
        (np.min(rgb, axis=2) >= cfg.near_white_threshold)
        & (g >= cfg.bright_threshold)
    )
    mask_u8 = near_white.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    grad = np.sqrt(gx * gx + gy * gy) / 4.0
    total_pixels = max(1, g.shape[0] * g.shape[1])
    large_uniform = np.zeros(g.shape, dtype=bool)

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area / total_pixels < cfg.large_region_fraction:
            continue
        component = labels == component_id
        if float(np.mean(grad[component])) <= cfg.low_texture_threshold:
            large_uniform[component] = True

    return large_uniform


def _robust_candidate_airlight(rgb, dark, candidate_mask, penalty_mask, cfg):
    """
    Region-aware airlight estimate.

    Important difference from the previous implementation:
    the bright-region penalty is applied to the candidate RANKING before
    top-k selection, not only to weights after selection. Thus penalized
    candidates can be replaced by stronger unpenalized alternatives.

    Returns: (estimate, selected_count, diagnostics).
    """
    ys, xs = np.where(candidate_mask)
    if len(ys) == 0:
        y, x = np.unravel_index(int(np.argmax(dark)), dark.shape)
        return rgb[y, x].astype(np.float32), 0, {
            "group_candidate_count": 0,
            "group_penalty_candidate_count": 0,
            "selected_penalty_candidate_count": 0,
        }

    values = dark[ys, xs].astype(np.float32)
    candidate_is_penalized = penalty_mask[ys, xs]
    k = max(1, int(np.ceil(len(values) * cfg.candidate_fraction)))
    k = min(k, len(values))

    penalty_weight = float(np.clip(cfg.large_region_weight, 0.0, 1.0))

    # Apply penalty BEFORE top-k selection. This changes which pixels are
    # selected, unlike the previous post-selection-only weighting.
    ranking_values = values.copy()
    ranking_values[candidate_is_penalized] *= penalty_weight

    selected_idx = np.argpartition(ranking_values, len(ranking_values) - k)[-k:]
    cy, cx = ys[selected_idx], xs[selected_idx]
    selected_dark = dark[cy, cx].astype(np.float32)
    selected_penalized = penalty_mask[cy, cx]

    d_min = float(np.min(selected_dark))
    d_max = float(np.max(selected_dark))
    if d_max - d_min > 1e-8:
        score = 0.5 + 0.5 * (selected_dark - d_min) / (d_max - d_min)
    else:
        score = np.ones_like(selected_dark, dtype=np.float32)

    # Keep a reduced residual weight for any penalized pixels that still
    # make the top-k set (for example, when alternatives are insufficient).
    weights = score.astype(np.float32)
    weights[selected_penalized] *= penalty_weight
    if float(np.sum(weights)) <= 1e-8:
        weights[:] = 1.0

    candidate_rgb = rgb[cy, cx].astype(np.float32)
    estimate = np.sum(candidate_rgb * weights[:, None], axis=0) / np.sum(weights)
    diagnostics = {
        "group_candidate_count": int(len(values)),
        "group_penalty_candidate_count": int(np.count_nonzero(candidate_is_penalized)),
        "selected_penalty_candidate_count": int(np.count_nonzero(selected_penalized)),
    }
    return np.clip(estimate, 0, 1).astype(np.float32), int(k), diagnostics


def estimate_airlight_adaptive(rgb_down, dark, cfg, baseline_values=None):
    """
    Experimental bright-region-aware estimate.
    Returns A_far, A_near, mean-gray threshold, diagnostics.
    """
    g = gray(rgb_down)
    threshold = float(np.mean(g))
    far = g >= threshold
    near = ~far
    penalty_mask = _large_uniform_near_white_mask(rgb_down, cfg)

    robust_far, k_far, far_diag = _robust_candidate_airlight(
        rgb_down, dark, far, penalty_mask, cfg
    )
    robust_near, k_near, near_diag = _robust_candidate_airlight(
        rgb_down, dark, near, penalty_mask, cfg
    )

    if baseline_values is None:
        base_far, base_near, _ = estimate_airlight(rgb_down, dark)
    else:
        base_far, base_near = baseline_values

    strength = float(np.clip(cfg.adaptive_strength, 0.0, 1.0))
    a_far = (1.0 - strength) * base_far + strength * robust_far
    a_near = (1.0 - strength) * base_near + strength * robust_near

    diagnostics = {
        "large_uniform_near_white_fraction": float(np.mean(penalty_mask)),
        "far_candidate_count": int(k_far),
        "near_candidate_count": int(k_near),
        "adaptive_strength": strength,
        "far_group_candidate_count": far_diag["group_candidate_count"],
        "far_group_penalty_candidate_count": far_diag["group_penalty_candidate_count"],
        "far_selected_penalty_candidate_count": far_diag["selected_penalty_candidate_count"],
        "near_group_candidate_count": near_diag["group_candidate_count"],
        "near_group_penalty_candidate_count": near_diag["group_penalty_candidate_count"],
        "near_selected_penalty_candidate_count": near_diag["selected_penalty_candidate_count"],
        "large_region_weight": float(np.clip(cfg.large_region_weight, 0.0, 1.0)),
    }
    return (
        np.clip(a_far, 0, 1).astype(np.float32),
        np.clip(a_near, 0, 1).astype(np.float32),
        threshold,
        diagnostics
    )


def atmospheric_map(rgb, a_far, a_near):
    # Baseline Equations (9)-(10).
    g = gray(rgb)
    i_max = float(gray(a_far.reshape(1, 1, 3))[0, 0])
    if i_max <= 1e-8:
        return np.broadcast_to(a_far, rgb.shape).copy()
    eta = np.clip((i_max - g) / i_max, 0, 1)
    blended = (1 - eta[..., None]) * a_far + eta[..., None] * a_near
    return np.clip(
        np.where((g >= i_max)[..., None], a_far, blended), 0, 1
    ).astype(np.float32)


def correction_factor(rgb, gamma=2.0):
    # Baseline Equation (13).
    cmin = np.min(rgb, axis=2)
    cmax = np.max(rgb, axis=2)
    denom = cmin + cmax
    ratio = np.divide(
        cmin, denom, out=np.zeros_like(cmin), where=denom > 1e-8
    )
    return 1 - np.power(np.clip(ratio, 0, 1), gamma)


def adaptive_omega_map(rgb, min_ratio, correction, cfg):
    """Build a bounded spatial omega map from haze and local-contrast cues.

    Haze cue: robustly normalized min_ratio * correction (larger tends to
    produce lower raw transmission in the original equation).
    Contrast cue: local grayscale standard deviation, robustly normalized.
    The gains are deliberately modest and the map is bounded.
    """
    q = np.clip(min_ratio * correction, 0.0, 1.0).astype(np.float32)
    q_lo, q_hi = np.percentile(q, [5, 95])
    haze = np.clip((q - q_lo) / max(float(q_hi - q_lo), 1e-6), 0, 1)

    g = gray(rgb).astype(np.float32)
    radius = int(cfg.omega_contrast_radius)
    if radius < 1:
        raise ValueError("omega_contrast_radius must be >= 1.")
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean = cv2.boxFilter(g, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(g * g, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    c_lo, c_hi = np.percentile(local_std, [5, 95])
    contrast = np.clip((local_std - c_lo) / max(float(c_hi - c_lo), 1e-6), 0, 1)

    omega_map = (
        float(cfg.omega)
        + float(cfg.omega_haze_gain) * (haze - 0.5)
        - float(cfg.omega_contrast_gain) * (contrast - 0.5)
    )
    return np.clip(omega_map, float(cfg.omega_min), float(cfg.omega_max)).astype(np.float32), haze, contrast


def transmission_map(rgb, a_map, omega=0.9375, gamma=2.0, omega_map=None):
    # Baseline Equation (12), with correction factor from Equation (13).
    ratios = np.divide(
        rgb, a_map, out=np.zeros_like(rgb), where=a_map > 1e-8
    )
    min_ratio = np.min(ratios, axis=2)
    k = correction_factor(rgb, gamma)
    omega_term = float(omega) if omega_map is None else omega_map
    return np.clip(1 - omega_term * min_ratio * k, 0, 1).astype(np.float32)


def guided_filter_gray(guide_rgb, source, radius=20, eps=1e-3):
    """
    Fast grayscale-guided filter implemented with OpenCV box filters.

    guide_rgb: RGB guidance image in [0, 1].
    source: scalar transmission map in [0, 1].
    radius: box-filter radius in pixels.
    eps: regularization parameter.

    This is an optional post-estimation refinement; it does not alter the
    paper's original transmission equation.
    """
    if radius < 1:
        raise ValueError("guided_filter_radius must be >= 1.")
    if eps <= 0:
        raise ValueError("guided_filter_eps must be > 0.")

    guide = gray(validate_rgb(guide_rgb)).astype(np.float32)
    p = np.asarray(source, dtype=np.float32)
    if p.ndim != 2 or p.shape != guide.shape:
        raise ValueError("source transmission and guide dimensions must match.")

    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)

    def mean(x):
        return cv2.boxFilter(
            x, ddepth=-1, ksize=ksize, normalize=True,
            borderType=cv2.BORDER_REFLECT
        )

    mean_i = mean(guide)
    mean_p = mean(p)
    corr_i = mean(guide * guide)
    corr_ip = mean(guide * p)
    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = mean(a)
    mean_b = mean(b)
    refined = mean_a * guide + mean_b
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def preserve_rgb_chroma(input_rgb, dehazed_rgb, strength=0.0):
    """Blend input-image RGB chroma into the dehazed result without YCbCr.

    Luminance is computed directly from RGB. The output luminance remains
    unchanged before final gamut clipping; only the zero-sum RGB chroma vector
    is blended. strength=0 is an exact no-op; strength=1 uses input chroma.
    """
    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha <= 0.0:
        return np.asarray(dehazed_rgb, dtype=np.float32).copy()
    src = validate_rgb(input_rgb)
    out = validate_rgb(dehazed_rgb)
    if src.shape != out.shape:
        raise ValueError("input and dehazed images must have matching shapes")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y_src = np.sum(src * weights, axis=2, keepdims=True)
    y_out = np.sum(out * weights, axis=2, keepdims=True)
    chroma_src = src - y_src
    chroma_out = out - y_out
    blended_chroma = (1.0 - alpha) * chroma_out + alpha * chroma_src
    return np.clip(y_out + blended_chroma, 0.0, 1.0).astype(np.float32)


def reconstruct(rgb, a_map, t_map, t_min=0.05):
    # Baseline Equation (6); t_min avoids unstable division.
    t = np.maximum(t_map, t_min)[..., None]
    return np.clip((rgb - a_map) / t + a_map, 0, 1).astype(np.float32)


def dehaze_image(image, config=None):
    cfg = config or DehazeConfig()
    rgb = validate_rgb(image)
    h, w = rgb.shape[:2]
    factor = max(1, int(cfg.downsample_factor))
    small = rgb.copy() if factor == 1 else cv2.resize(
        rgb,
        (max(1, round(w / factor)), max(1, round(h / factor))),
        interpolation=cv2.INTER_AREA
    )

    dark = dark_channel(small, cfg.patch_size)
    baseline_far, baseline_near, baseline_threshold = estimate_airlight(small, dark)
    adaptive_diagnostics = {}

    if cfg.adaptive_airlight:
        a_far, a_near, threshold, adaptive_diagnostics = estimate_airlight_adaptive(
            small, dark, cfg, baseline_values=(baseline_far, baseline_near)
        )
    else:
        a_far, a_near, threshold = (
            baseline_far, baseline_near, baseline_threshold
        )

    a_map = atmospheric_map(rgb, a_far, a_near)
    omega_map = None
    omega_diagnostics = {}
    if cfg.adaptive_omega:
        ratios = np.divide(rgb, a_map, out=np.zeros_like(rgb), where=a_map > 1e-8)
        min_ratio = np.min(ratios, axis=2)
        correction = correction_factor(rgb, cfg.gamma)
        omega_map, haze_cue, contrast_cue = adaptive_omega_map(
            rgb, min_ratio, correction, cfg
        )
        omega_diagnostics = {
            "omega_mean": float(np.mean(omega_map)),
            "omega_min_observed": float(np.min(omega_map)),
            "omega_max_observed": float(np.max(omega_map)),
            "haze_cue_mean": float(np.mean(haze_cue)),
            "contrast_cue_mean": float(np.mean(contrast_cue)),
        }
    t_map_raw = transmission_map(rgb, a_map, cfg.omega, cfg.gamma, omega_map=omega_map)
    if cfg.refine_transmission:
        t_map = guided_filter_gray(
            rgb, t_map_raw,
            radius=cfg.guided_filter_radius,
            eps=cfg.guided_filter_eps,
        )
    else:
        t_map = t_map_raw
    output = reconstruct(rgb, a_map, t_map, cfg.t_min)
    if cfg.color_preserve_strength > 0:
        output = preserve_rgb_chroma(rgb, output, cfg.color_preserve_strength)

    return {
        "dehazed": output,
        "atmospheric_map": a_map,
        "transmission_map": t_map,
        "transmission_map_raw": t_map_raw,
        "dark_channel": dark,
        "A_far": a_far,
        "A_near": a_near,
        "baseline_A_far": baseline_far,
        "baseline_A_near": baseline_near,
        "threshold": threshold,
        "adaptive_diagnostics": adaptive_diagnostics,
        "omega_map": omega_map,
        "omega_diagnostics": omega_diagnostics,
    }
