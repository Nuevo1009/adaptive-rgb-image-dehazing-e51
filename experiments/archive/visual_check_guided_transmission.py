from pathlib import Path
import argparse
import cv2
import numpy as np

from dataset_config import PROJECT_ROOT, match_pairs
from settings import GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN
from dehazing.core import DehazeConfig, dehaze_image, read_rgb


def to_u8(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def label_panel(image, label, width=480, height=300):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height + 42, width, 3), dtype=np.uint8)
    y, x = (height - nh) // 2, (width - nw) // 2
    canvas[y:y+nh, x:x+nw] = resized
    cv2.putText(canvas, label, (10, height + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Save visual E0/E1 comparisons and raw/refined transmission maps."
    )
    parser.add_argument("--count", type=int, default=12,
                        help="Number of validation pairs to visualize (default 12)")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Optional exact hazy image filenames to inspect")
    args = parser.parse_args()

    root = PROJECT_ROOT / "dataset" / "CNN_Dataset"
    hazy_dir, clear_dir = root / "validation" / "hazy", root / "validation" / "clear"
    if not hazy_dir.is_dir() or not clear_dir.is_dir():
        raise SystemExit(f"Validation folders not found: {hazy_dir} / {clear_dir}")

    pairs, uh, uc = match_pairs(hazy_dir, clear_dir)
    if args.names:
        wanted = {n.lower() for n in args.names}
        pairs = [(h, g) for h, g in pairs if h.name.lower() in wanted]
    else:
        pairs = pairs[:max(1, args.count)]

    if not pairs:
        raise SystemExit("No matching validation pairs selected.")

    common = dict(
        downsample_factor=DOWNSAMPLE_FACTOR,
        patch_size=PATCH_SIZE,
        omega=OMEGA,
        gamma=GAMMA,
        t_min=T_MIN,
        adaptive_airlight=True,
        adaptive_strength=0.75,
        large_region_weight=0.25,
    )
    cfg_e0 = DehazeConfig(**common, refine_transmission=False)
    cfg_e1 = DehazeConfig(
        **common, refine_transmission=True,
        guided_filter_radius=20, guided_filter_eps=1e-3
    )

    out_dir = PROJECT_ROOT / "outputs" / "guided_transmission_visual_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (hazy_path, gt_path) in enumerate(pairs, 1):
        hazy = read_rgb(hazy_path)
        clear = read_rgb(gt_path)
        r0 = dehaze_image(hazy, cfg_e0)
        r1 = dehaze_image(hazy, cfg_e1)

        panels = [
            label_panel(to_u8(hazy), "Hazy input"),
            label_panel(to_u8(r0["dehazed"]), "E0: Adaptive airlight"),
            label_panel(to_u8(r1["dehazed"]), "E1: Guided transmission"),
            label_panel(to_u8(clear), "Clear ground truth"),
            label_panel(cv2.cvtColor(
                np.clip(r1["transmission_map_raw"] * 255, 0, 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB), "Raw transmission"),
            label_panel(cv2.cvtColor(
                np.clip(r1["transmission_map"] * 255, 0, 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB), "Guided-refined transmission"),
        ]
        top = np.hstack(panels[:3])
        bottom = np.hstack(panels[3:])
        sheet = np.vstack([top, bottom])
        out_path = out_dir / f"{idx:03d}_{hazy_path.stem}_comparison.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"[{idx}/{len(pairs)}] Saved {out_path}")

    print(f"\nVisual comparisons saved in:\n{out_dir}")
    print("Inspect for halos, over-smoothing, colour shifts, lost detail, and unnatural edges.")


if __name__ == "__main__":
    main()
