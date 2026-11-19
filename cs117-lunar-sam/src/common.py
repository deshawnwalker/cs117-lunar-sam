"""Shared paths, settings, and scoring used by every script, so all methods are graded identically."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]   # repo root

# ---- Evaluation protocol (same for every site and method) ----
MIN_R = 6        # size floor: radius >= 6 px = craters >= 12 m diameter at 1 m/px
MAX_R = 110      # largest crater radius searched for (px)
IOU_THR = 0.5    # a crater counts as found if matched with IoU >= 0.5
N_BOOT = 1000    # bootstrap resamples for 95% CIs
SEED = 0


# ---- Paths ----
def site_dirs(site):
    """data/<site>/ holds inputs (frames list, labels, crops); results/<site>/ holds outputs."""
    data = ROOT / "data" / site
    results = ROOT / "results" / site
    results.mkdir(parents=True, exist_ok=True)
    return data, results


def load_site(site):
    data, results = site_dirs(site)
    cfg = json.load(open(data / "site.json"))
    frames = pd.read_csv(data / "frames.csv")     # target_angle, product_id, incidence
    return cfg, frames, data, results


def crop_path(data_dir, target_angle):
    return data_dir / "crops" / f"inc{int(target_angle)}.png"


def load_crops(site):
    """Return {actual incidence angle: grayscale crop} for a site."""
    _, frames, data, _ = load_site(site)
    crops = {}
    for _, f in frames.iterrows():
        p = crop_path(data, f.target_angle)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing crop {p} - run prepare_frames.py first")
        crops[float(f.incidence)] = img
    return crops


def load_labels(site, min_r=MIN_R):
    """Hand labels as circles (x, y, r), filtered to the size floor."""
    data, _ = site_dirs(site)
    labels = pd.read_csv(data / "labels.csv")
    return labels[labels.r >= min_r].reset_index(drop=True)


# ---- Masks as (x0, y0, cropped boolean array) so IoU only touches nearby pixels ----
def circle_obj(x, y, r, shape):
    h, w = shape
    x0, y0 = max(int(x - r), 0), max(int(y - r), 0)
    x1, y1 = min(int(x + r) + 1, w), min(int(y + r) + 1, h)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return x0, y0, (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def mask_obj(full_mask):
    ys, xs = np.nonzero(full_mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return int(x0), int(y0), full_mask[y0:y1, x0:x1]


def pair_iou(a, b):
    ax0, ay0, am = a
    bx0, by0, bm = b
    ax1, ay1 = ax0 + am.shape[1], ay0 + am.shape[0]
    bx1, by1 = bx0 + bm.shape[1], by0 + bm.shape[0]
    if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
        return 0.0
    ux0, uy0 = min(ax0, bx0), min(ay0, by0)
    ux1, uy1 = max(ax1, bx1), max(ay1, by1)
    A = np.zeros((uy1 - uy0, ux1 - ux0), bool)
    B = np.zeros_like(A)
    A[ay0 - uy0:ay1 - uy0, ax0 - ux0:ax1 - ux0] = am
    B[by0 - uy0:by1 - uy0, bx0 - ux0:bx1 - ux0] = bm
    union = np.logical_or(A, B).sum()
    return np.logical_and(A, B).sum() / union if union else 0.0


# ---- Scoring: one-to-one Hungarian matching ----
def score(gt_objs, pred_objs, thr=IOU_THR):
    n_gt, n_pred = len(gt_objs), len(pred_objs)
    per_gt = np.zeros(n_gt)
    if n_pred:
        iou = np.zeros((n_gt, n_pred))
        for i, g in enumerate(gt_objs):
            for j, p in enumerate(pred_objs):
                iou[i, j] = pair_iou(g, p)
        r, c = linear_sum_assignment(-iou)
        per_gt[r] = iou[r, c]
    matched = per_gt >= thr
    return {
        "recall": matched.mean(),
        "mean_iou": np.where(matched, per_gt, 0).mean(),
        "precision": matched.sum() / n_pred if n_pred else 0.0,
        "n_pred": n_pred,
        "per_gt": per_gt,
    }


def bootstrap_ci(per_gt, thr=IOU_THR, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    rec, miou = [], []
    for _ in range(n):
        s = per_gt[rng.integers(0, len(per_gt), len(per_gt))]
        m = s >= thr
        rec.append(m.mean())
        miou.append(np.where(m, s, 0).mean())
    return np.percentile(rec, [2.5, 97.5]), np.percentile(miou, [2.5, 97.5])


def result_row(site, method, angle, s, n_gt):
    (r_lo, r_hi), (m_lo, m_hi) = bootstrap_ci(s["per_gt"])
    return {"site": site, "method": method, "incidence": angle,
            "recall": s["recall"], "recall_lo": r_lo, "recall_hi": r_hi,
            "mean_iou": s["mean_iou"], "miou_lo": m_lo, "miou_hi": m_hi,
            "precision": s["precision"], "n_pred": s["n_pred"], "n_gt": n_gt}


def draw_overlays(crops, labels, preds_by_angle, out_png, pred_name, recall_by_angle=None):
    """Labels in green, predictions in red, one panel per sun angle."""
    import matplotlib.pyplot as plt
    n = len(crops)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    for ax, (angle, gray) in zip(np.ravel(axes), sorted(crops.items())):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        for x0, y0, m in preds_by_angle[angle]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, [c + [x0, y0] for c in cs], -1, (255, 50, 50), 2)
        for x, y, r in labels[["x", "y", "r"]].values:
            cv2.circle(vis, (int(x), int(y)), int(r), (50, 255, 50), 2)
        title = f"{angle} deg - {len(preds_by_angle[angle])} {pred_name} detections"
        if recall_by_angle:
            title += f", recall {recall_by_angle[angle]:.2f}"
        ax.imshow(vis)
        ax.set_title(title + "\n(green = labels, red = " + pred_name + ")")
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()


def plot_curves(dfs, out_png, title):
    """Recall and mean IoU vs incidence angle. dfs: {label: (dataframe, line style)}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (m, lo, hi, name) in zip(ax, [("recall", "recall_lo", "recall_hi", "Recall (IoU >= 0.5)"),
                                          ("mean_iou", "miou_lo", "miou_hi", "Mean IoU")]):
        for label, (df, style) in dfs.items():
            a.plot(df.incidence, df[m], style, label=label)
            a.fill_between(df.incidence, df[lo], df[hi], alpha=0.15)
        a.set_xlabel("Solar incidence angle (deg)")
        a.set_ylabel(name)
        a.set_ylim(0, 1)
        a.set_title(title)
        a.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
