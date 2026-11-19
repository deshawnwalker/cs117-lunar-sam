"""
Hough circle baseline.

  Tune on a site (picks one parameter set by mean F1 across all its sun angles, then saves it):
      python src/hough_baseline.py --site apollo11
  Run with frozen parameters (for check / held-out test sites - no tuning):
      python src/hough_baseline.py --site apollo16 --params results/apollo11/hough_params_frozen.json

Outputs in results/<site>/: hough_results.csv, hough_params_frozen.json (when tuning),
hough_curves.png, hough_overlays.png
"""

import argparse
import itertools
import json

import cv2
import numpy as np
import pandas as pd

from common import (MIN_R, MAX_R, circle_obj, draw_overlays, load_crops, load_labels,
                    plot_curves, result_row, score, site_dirs)

# Hyperparameter grid searched when tuning (2 x 2 x 2 x 3 x 4 = 96 combinations)
GRID = {
    "blur":    [1.5, 3.0],                 # Gaussian blur sigma before edge detection
    "dp":      [1.0, 1.5],                 # accumulator resolution
    "minDist": [MIN_R, 2 * MIN_R],         # minimum distance between circle centers (px)
    "param1":  [60, 100, 150],             # Canny edge threshold
    "param2":  [15, 20, 30, 40],           # votes needed to accept a circle
}


def run_hough(gray, p):
    g = cv2.GaussianBlur(gray, (0, 0), p["blur"])
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=p["dp"], minDist=p["minDist"],
                               param1=p["param1"], param2=p["param2"],
                               minRadius=MIN_R, maxRadius=MAX_R)
    return [] if circles is None else circles[0].tolist()


def tune(crops, gt, shape):
    """Pick the parameter set with the best mean F1 across all sun angles."""
    combos = list(itertools.product(*GRID.values()))
    print(f"Tuning: {len(combos)} parameter combinations x {len(crops)} frames")
    best, best_f1 = None, -1
    for k, vals in enumerate(combos, 1):
        p = dict(zip(GRID.keys(), vals))
        f1s = []
        for gray in crops.values():
            s = score(gt, [circle_obj(x, y, r, shape) for x, y, r in run_hough(gray, p)])
            pr, rc = s["precision"], s["recall"]
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
        if np.mean(f1s) > best_f1:
            best, best_f1 = p, np.mean(f1s)
        if k % 12 == 0:
            print(f"  {k}/{len(combos)} done, best mean F1 so far {best_f1:.3f}")
    print(f"Best parameters: {best} (mean F1 {best_f1:.3f})")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--params", help="frozen parameter JSON; omit to tune on this site")
    args = ap.parse_args()

    _, results = site_dirs(args.site)
    crops = load_crops(args.site)
    labels = load_labels(args.site)
    shape = next(iter(crops.values())).shape
    gt = [circle_obj(x, y, r, shape) for x, y, r in labels[["x", "y", "r"]].values]
    print(f"{args.site}: {len(gt)} labeled craters (>= {2 * MIN_R} m), {len(crops)} frames")

    if args.params:
        params = json.load(open(args.params))
        print(f"Using frozen parameters from {args.params}: {params}")
    else:
        params = tune(crops, gt, shape)
        json.dump(params, open(results / "hough_params_frozen.json", "w"), indent=2)

    rows, preds = [], {}
    for angle, gray in sorted(crops.items()):
        preds[angle] = [circle_obj(x, y, r, shape) for x, y, r in run_hough(gray, params)]
        rows.append(result_row(args.site, "hough", angle, score(gt, preds[angle]), len(gt)))
    df = pd.DataFrame(rows)
    df.to_csv(results / "hough_results.csv", index=False)
    print(df.drop(columns=["site", "method"]).round(3).to_string(index=False))

    plot_curves({"Hough": (df, "--o")}, results / "hough_curves.png", f"{args.site}: Hough")
    draw_overlays(crops, labels, preds, results / "hough_overlays.png", "Hough",
                  dict(zip(df.incidence, df.recall)))
    print(f"Saved results to {results}")


if __name__ == "__main__":
    main()
