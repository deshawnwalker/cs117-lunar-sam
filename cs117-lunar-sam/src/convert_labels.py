"""
Convert a makesense.ai box export into crater circles and check them on every frame.

    python src/convert_labels.py --site apollo11

Reads  data/<site>/labels_export.csv  (makesense "Single CSV file" export, rectangles around crater rims)
Writes data/<site>/labels.csv          (x, y, r for EVERY box; the size floor is applied at evaluation)
       results/<site>/labels_check.png (circles drawn on all frames - they should sit on rims in every one)
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import MIN_R, load_crops, site_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    args = ap.parse_args()
    data, results = site_dirs(args.site)

    raw = pd.read_csv(data / "labels_export.csv", header=None).iloc[:, :8]
    raw.columns = ["label", "x", "y", "w", "h", "image", "img_w", "img_h"]
    for c in ["x", "y", "w", "h"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")      # header row -> NaN
    boxes = raw.dropna(subset=["x", "y", "w", "h"])

    labels = pd.DataFrame({
        "x": boxes.x + boxes.w / 2,       # center x = left edge + half the width
        "y": boxes.y + boxes.h / 2,       # center y = top edge + half the height
        "r": (boxes.w + boxes.h) / 4,     # radius = average half-size of the box
    })
    labels.to_csv(data / "labels.csv", index=False)
    kept = labels[labels.r >= MIN_R]
    print(f"{len(labels)} boxes converted; {len(kept)} are >= {2 * MIN_R} m (used in evaluation)")

    crops = load_crops(args.site)
    n = len(crops)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(18, 6 * ((n + 2) // 3)))
    for ax, (angle, im) in zip(np.ravel(axes), sorted(crops.items())):
        ax.imshow(im, cmap="gray")
        for x, y, r in kept[["x", "y", "r"]].values:
            ax.add_patch(plt.Circle((x, y), r, fill=False, color="lime", linewidth=0.8))
        ax.set_title(f"{angle} deg - {len(kept)} labels")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(results / "labels_check.png", dpi=110)
    print(f"Saved {data / 'labels.csv'} and {results / 'labels_check.png'}")


if __name__ == "__main__":
    main()
