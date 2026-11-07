import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
import csv, cv2
from settings import DATASET_ROOT, GAMMA, OMEGA, DOWNSAMPLE_FACTOR, PATCH_SIZE, T_MIN, dataset_dir
from dehazing.core import DehazeConfig, dehaze_image, read_rgb, save_rgb

EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Install optional dependency: pip install -r requirements-optional-detection.txt")
    if not DATASET_ROOT.is_dir():
        raise SystemExit(f"Dataset root not found: {DATASET_ROOT}\nEdit DATASET_ROOT in settings.py.")
    weights=input("Enter path to YOLO weights file: ").strip().strip('"')
    model=YOLO(weights)
    src=dataset_dir("test","hazy")
    out=Path("outputs/detection")
    cfg=DehazeConfig(DOWNSAMPLE_FACTOR,PATCH_SIZE,OMEGA,GAMMA,T_MIN)
    rows=[]
    for p in sorted(x for x in src.rglob("*") if x.suffix.lower() in EXTS):
        rgb=read_rgb(p)
        dehazed=dehaze_image(rgb,cfg)["dehazed"]
        save_rgb(out/"dehazed"/f"{p.stem}.png",dehazed)
        for label,img in [("hazy",rgb),("dehazed",(dehazed*255+0.5).astype("uint8"))]:
            result=model.predict(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),verbose=False)[0]
            boxes=result.boxes
            rows.append([p.name,label,len(boxes),
                         ";".join(str(int(x)) for x in boxes.cls.tolist()) if boxes is not None else "",
                         ";".join(f"{float(x):.5f}" for x in boxes.conf.tolist()) if boxes is not None else ""])
            target=out/"annotated"/label/f"{p.stem}.jpg"
            target.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(target),result.plot())
    out.mkdir(parents=True,exist_ok=True)
    with (out/"detection_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["image","version","num_detections","class_ids","confidences"]);w.writerows(rows)
    print(f"Results saved under {out.resolve()}")

if __name__=="__main__":
    main()
