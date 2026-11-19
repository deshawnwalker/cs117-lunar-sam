# Does sun angle change how well SAM finds lunar craters?

CS 117 (Computer Vision) final project, Team 15: Adhith Jacob, DeShawn Walker, Eshaan Punalekar.

**Question.** How does SAM 2.1's ability to find lunar craters change with solar incidence angle (≈16° to 82°),
how does it compare to a Hough circle baseline, and is the pattern the same on mare and highland terrain?

**Metric.** Recall at IoU ≥ 0.5 against circular crater annotations: the fraction of hand-labeled craters
(≥ 12 m diameter) matched one-to-one (Hungarian matching) by a prediction. Secondary: mean IoU and precision.
95% bootstrap confidence intervals over craters.

## Sites and splits

| Site | Terrain | Role | LROC photometric site |
|---|---|---|---|
| Apollo 11 | Mare | Tuning (validation) | NAC_PHO_E010N0230 |
| Apollo 17 | Mare valley near mountains | Check (validation) | NAC_PHO_E199N0308 |
| Apollo 16 | Highlands | Held-out test (run once, settings frozen) | NAC_PHO_E090S0155 |

Nothing is trained: SAM uses its released weights, and Hough has no learned weights. "Validation" means
hyperparameters were chosen on that site.

## Data

Images are LROC Narrow Angle Camera photometric products (ASU / NASA PDS), which LROC has already registered
to a terrain model, so the same map window lines up pixel-for-pixel across sun angles. For each site we cut one
1000 × 1000 px crop at 1 m/px from each of 6 frames spanning low to high sun. The frame IDs and exact incidence
angles are in `data/<site>/frames.csv`; raw `.IMG` files (1–5 GB each) are not stored here and are re-downloaded
by `prepare_frames.py`.

Craters were labeled by hand once per site on a mid-sun frame (rectangles fit to crater rims in
[makesense.ai](https://www.makesense.ai), converted to circles) and reused on every frame of that site.

## Results so far (Apollo 11, tuning site)

Recall at IoU ≥ 0.5 [95% CI], 154 craters ≥ 12 m:

| Method | 15.85° | 26.24° | 41.09° | 53.13° | 73.47° | 81.78° |
|---|---|---|---|---|---|---|
| Hough | 0.16 [0.10–0.22] | 0.25 [0.18–0.32] | 0.21 [0.15–0.27] | 0.27 [0.21–0.34] | 0.24 [0.18–0.31] | 0.34 [0.27–0.42] |

Produced by `python src/hough_baseline.py --site apollo11` → `results/apollo11/hough_results.csv`.

## Status

Done: Apollo 11 frames, labels, and Hough baseline. Next: SAM 2.1 evaluation on Apollo 11
(same frames, labels, and scoring code in `src/common.py`), then Apollo 17 and Apollo 16.

## Repository layout

```
src/
  common.py            shared settings + scoring (every method will be graded by this code)
  prepare_frames.py    download LROC frames, cut aligned crops
  convert_labels.py    makesense box export -> crater circles, overlay check
  hough_baseline.py    Hough baseline (tunes on a site, or runs with frozen params)
data/<site>/
  site.json            LROC site ID, crop center, crop size
  frames.csv           target angle, LROC product ID, exact incidence angle
  frames_meta.csv      coverage and crop shift per frame (written by prepare_frames.py)
  crops/inc<angle>.png the 6 aligned crops
  labels_export.csv    raw makesense export
  labels.csv           crater circles (x, y, r) in pixels
results/<site>/        CSVs, frozen parameters, plots, overlays
notebooks/             Colab notebook used to run the pipeline
```

## How to run (Google Colab)

```python
!git clone https://github.com/<deshawnwalker>/cs117-lunar-sam.git
%cd cs117-lunar-sam
!pip install -q -r requirements.txt

# Only needed for a new site (downloads ~20 GB, keeps only the crops):
!python src/prepare_frames.py --site apollo11

!python src/convert_labels.py --site apollo11                 # after labeling in makesense.ai
!python src/hough_baseline.py --site apollo11                 # tunes + scores (~10-30 min)

# Check and test sites reuse Apollo 11's frozen Hough parameters (no re-tuning):
!python src/hough_baseline.py --site apollo16 --params results/apollo11/hough_params_frozen.json
```

## Design decisions

- **Size floor 12 m** (radius ≥ 6 px): smaller craters span too few pixels to label or detect reliably.
- **Crop shift:** the Apollo 11 crop is centered 100 m west of the landing site because one frame's image
  strip did not fully cover a crop centered on the site (90.7% valid). The shift is found automatically.
- **Normalization:** each crop is stretched between its 1st and 99th brightness percentile.
- **Hough** parameters are chosen by mean F1 across all sun angles of the tuning site, then frozen.
- Circle labels are an approximation, so results describe agreement with circular crater annotations,
  not exact segmentation quality.

## External resources

- OpenCV `HoughCircles`
- LROC NAC photometric products, Arizona State University / NASA PDS: <https://data.lroc.im-ldi.com/lroc/rdr_product_select>
- Crater labels: our own, made in makesense.ai
