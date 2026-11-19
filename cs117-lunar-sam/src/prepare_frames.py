"""
Download a site's LROC NAC photometric frames and cut the same aligned crop from each.

    python src/prepare_frames.py --site apollo11

Reads  data/<site>/site.json   (LROC site ID, crop center lat/lon, crop size, resolution)
       data/<site>/frames.csv  (target_angle, product_id, incidence)
Writes data/<site>/crops/inc<target>.png  (one normalized 8-bit crop per sun angle)
       data/<site>/frames_meta.csv        (coverage and crop shift per frame)

The raw .IMG files (1-5 GB each) go to raw/<site>/ (not committed) and are deleted afterwards
unless --keep-raw is given. Frames are already registered to a terrain model by LROC, so cutting
the same map window from each gives pixel-aligned crops.
"""

import argparse
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform

from common import ROOT, crop_path, load_site

MOON_LONLAT = "+proj=longlat +R=1737400 +no_defs"   # lunar lat/lon (Moon radius 1737.4 km)


def find_img_url(site_id, pid):
    """Open the frame's LROC product page and return its PDS .IMG download link."""
    page = f"https://data.lroc.im-ldi.com/lroc/view_rdr_product/{site_id}_{pid}"
    html = urllib.request.urlopen(page, timeout=60).read().decode("utf-8", "ignore")
    links = re.findall(r'href="([^"]+\.IMG)"', html, flags=re.I)
    if not links:
        raise RuntimeError(f"No .IMG link on {page}")
    return urllib.parse.urljoin(page, links[0])


def read_window(src, ref_crs, transform, size):
    """Read one crop window from a frame, resampled onto our grid. Returns image + empty-pixel mask."""
    with WarpedVRT(src, crs=ref_crs, transform=transform, width=size, height=size,
                   resampling=Resampling.bilinear) as vrt:
        img = vrt.read(1).astype(np.float32)
    bad = ~np.isfinite(img) | (img < -1e30)
    if src.nodata is not None:
        bad |= img == src.nodata
    return img, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded .IMG files")
    args = ap.parse_args()

    cfg, frames, data, _ = load_site(args.site)
    raw_dir = ROOT / "raw" / args.site
    raw_dir.mkdir(parents=True, exist_ok=True)
    (data / "crops").mkdir(exist_ok=True)
    size, res = cfg["crop_px"], cfg["res_m"]

    # 1. Download each frame (skips files already downloaded)
    paths = {}
    for _, f in frames.iterrows():
        p = raw_dir / f"{cfg['lroc_site_id']}_{f.product_id}.IMG"
        paths[f.target_angle] = p
        if p.exists() and p.stat().st_size > 1e8:
            print(f"{f.target_angle:>3} deg: already downloaded")
            continue
        url = find_img_url(cfg["lroc_site_id"], f.product_id)
        print(f"{f.target_angle:>3} deg: downloading {url}")
        subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", str(p), url], check=True)

    # 2. Crop window centered on the site's lat/lon, slid west in 50 m steps
    #    until every frame fully covers it (a frame's strip can end near the center)
    srcs = {t: rasterio.open(p) for t, p in paths.items()}
    ref_crs = next(iter(srcs.values())).crs
    xs, ys = warp_transform(MOON_LONLAT, ref_crs, [cfg["center_lon"]], [cfg["center_lat"]])
    half = size * res / 2
    for shift in range(0, -1001, -50):
        t = from_origin(xs[0] + shift - half, ys[0] + half, res, res)
        worst = min(100 * (1 - read_window(s, ref_crs, t, size)[1].mean()) for s in srcs.values())
        print(f"shift {shift:>5} m west -> lowest coverage {worst:.1f}%")
        if worst == 100.0:
            break
    assert worst == 100.0, "No crop position is fully covered by every frame - pick different frames"

    # 3. Crop, normalize (1st-99th percentile stretch, same rule for every frame), save
    rows = []
    for _, f in frames.iterrows():
        src = srcs[f.target_angle]
        img, bad = read_window(src, ref_crs, t, size)
        lo, hi = np.percentile(img[~bad], [1, 99])
        img8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(crop_path(data, f.target_angle)), img8)
        rows.append({"target_angle": f.target_angle, "product_id": f.product_id,
                     "incidence": f.incidence, "orig_res_m": round(src.res[0], 2),
                     "pct_valid": round(100 * (~bad).mean(), 1), "shift_x_m": shift})
        src.close()
    meta = pd.DataFrame(rows)
    meta.to_csv(data / "frames_meta.csv", index=False)
    print(meta.to_string(index=False))

    if not args.keep_raw:
        shutil.rmtree(raw_dir)
        print(f"Deleted raw files in {raw_dir}")


if __name__ == "__main__":
    main()
