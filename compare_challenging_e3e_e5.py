from __future__ import annotations
import argparse, csv, math, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def norm_key(p):
    s = p.stem.lower()
    return re.sub(r"_dehazed$", "", s)

def image_map(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    result = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            k = norm_key(p)
            if k not in result or len(p.name) < len(result[k].name):
                result[k] = p
    return result

def read_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0

def lum(a):
    return .299*a[...,0] + .587*a[...,1] + .114*a[...,2]

def psnr(a,b):
    mse = float(np.mean((a-b)**2))
    return 99.0 if mse <= 1e-12 else 10*math.log10(1/mse)

def get_features(hazy, gt, e3, e5):
    yh, yg = lum(hazy), lum(gt)
    def green_fraction(a):
        r,g,b = a[...,0],a[...,1],a[...,2]
        return float(np.mean((g > r*1.08) & (g > b*.92) & (g > .16)))
    def chroma(a):
        return a - lum(a)[...,None]
    return {
        "dark_fraction_gt": float(np.mean(yg < .22)),
        "dark_fraction_hazy": float(np.mean(yh < .20)),
        "bright_fraction_hazy": float(np.mean(yh > .78)),
        "bright_region_mean_hazy": float(np.mean(yh[yh > .62])) if np.any(yh > .62) else float(np.mean(yh)),
        "vegetation_proxy": green_fraction(hazy),
        "e5_e3_chroma_change": float(np.mean(np.abs(chroma(e5)-chroma(e3)))),
        "colour_strength_hazy": float(np.mean(np.sqrt(np.sum(chroma(hazy)**2,axis=2)))),
        "psnr_e3": psnr(e3,gt),
        "psnr_e5": psnr(e5,gt),
    }

def scores(f):
    return {
        "dark_regions_buildings": .7*f["dark_fraction_gt"]+.3*f["dark_fraction_hazy"],
        "bright_skies": .7*f["bright_fraction_hazy"]+max(0,f["bright_region_mean_hazy"]-.65),
        "trees_vegetation": f["vegetation_proxy"],
        "strong_colour_differences": f["colour_strength_hazy"]+3*f["e5_e3_chroma_change"],
    }

def contact_sheet(rows, destination, title, thumb_w=280):
    if not rows: return
    labels = ["Hazy input","Ground truth","E3-E reference","E5 Adaptive RGB"]
    margin,gap,head = 12,10,48
    try: font=ImageFont.truetype("arial.ttf",14)
    except OSError: font=ImageFont.load_default()
    prepared=[]; row_heights=[]
    for r in rows:
        ims=[]
        for k in ("hazy_path","gt_path","e3_path","e5_path"):
            im=Image.open(r[k]).convert("RGB")
            h=max(1,round(im.height*thumb_w/im.width))
            ims.append(im.resize((thumb_w,h),Image.Resampling.LANCZOS))
        prepared.append(ims); row_heights.append(max(x.height for x in ims)+45)
    W=margin*2+4*thumb_w+3*gap
    H=head+sum(row_heights)+gap*(len(rows)-1)+margin
    sheet=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(sheet)
    d.text((margin,12),title,fill="black",font=font)
    y=head
    for r,ims,rh in zip(rows,prepared,row_heights):
        d.text((margin,y),f"{r['key']} | E3-E {r['psnr_e3']:.2f} dB | E5 {r['psnr_e5']:.2f} dB",fill="black",font=font)
        y+=20; x=margin
        for label,im in zip(labels,ims):
            d.text((x,y),label,fill="black",font=font)
            sheet.paste(im,(x,y+18)); x+=thumb_w+gap
        y+=rh
    sheet.save(destination,quality=92)

def main():
    ap=argparse.ArgumentParser(description="Find and compare challenging E3-E/E5 validation images.")
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument("--top",type=int,default=8)
    ap.add_argument("--hazy",type=Path); ap.add_argument("--gt",type=Path)
    ap.add_argument("--e3",type=Path); ap.add_argument("--e5",type=Path)
    a=ap.parse_args(); root=a.root.resolve()
    base=root/"outputs"/"e5_e6_e7_separate"/"CNN_Dataset_validation"
    folders=[
        a.hazy or root/"dataset"/"CNN_Dataset"/"validation"/"hazy",
        a.gt or root/"dataset"/"CNN_Dataset"/"validation"/"clear",
        a.e3 or base/"E3E_REFERENCE"/"dehazed",
        a.e5 or base/"E5_ADAPTIVE_RGB"/"dehazed",
    ]
    for label,folder in zip(("Hazy","GT","E3-E","E5"),folders): print(f"{label}: {folder}")
    maps=[image_map(x) for x in folders]
    common=sorted(set.intersection(*(set(m) for m in maps)))
    if not common:
        raise SystemExit("No common filenames found. Check paths or pass --hazy --gt --e3 --e5.")
    records=[]
    for i,k in enumerate(common,1):
        paths=[m[k] for m in maps]
        size=Image.open(paths[1]).size
        arrays=[read_rgb(p,size) for p in paths]
        f=get_features(*arrays)
        f.update({"key":k,"hazy_path":str(paths[0]),"gt_path":str(paths[1]),"e3_path":str(paths[2]),"e5_path":str(paths[3])})
        f["psnr_delta_e5_minus_e3"]=f["psnr_e5"]-f["psnr_e3"]
        f["category_scores"]=scores(f); records.append(f)
        if i%100==0 or i==len(common): print(f"Analysed {i}/{len(common)}")
    out=root/"outputs"/"e5_e6_e7_separate"/"challenging_image_review"; out.mkdir(parents=True,exist_ok=True)
    csvfile=out/"challenging_image_features.csv"
    fields=[k for k in records[0] if k!="category_scores"]+[f"score_{k}" for k in scores(records[0])]
    with csvfile.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in records:
            row={k:v for k,v in r.items() if k!="category_scores"}
            row.update({f"score_{k}":v for k,v in r["category_scores"].items()}); w.writerow(row)
    titles={"dark_regions_buildings":"Dark regions / buildings","bright_skies":"Bright skies","trees_vegetation":"Trees / vegetation (green-colour heuristic)","strong_colour_differences":"Strong colour content / E5-E3 colour change"}
    selected=set()
    for cat,title in titles.items():
        chosen=sorted(records,key=lambda r:r["category_scores"][cat],reverse=True)[:max(1,a.top)]
        selected.update(r["key"] for r in chosen)
        contact_sheet(chosen,out/f"{cat}.jpg",title)
    combined=sorted((r for r in records if r["key"] in selected),key=lambda r:abs(r["psnr_delta_e5_minus_e3"]),reverse=True)[:24]
    contact_sheet(combined,out/"combined_review.jpg","Combined challenging-image review")
    print("\nDONE")
    print("Matched validation images:",len(common))
    print("CSV:",csvfile)
    print("Contact sheets folder:",out)
    print("Send combined_review.jpg, category JPGs, and challenging_image_features.csv.")

if __name__=="__main__": main()
