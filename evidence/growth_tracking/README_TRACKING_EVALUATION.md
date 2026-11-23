# Tracking and growth-history evaluation

**Paper:** *YOLO26-L-seg for Oyster Mushroom Cluster Segmentation and Growth Monitoring*

## Evaluation scope

This is the downstream tracking/measurement notebook. It consumes the cached YOLO26-L-seg prediction records from `full-inference.ipynb` and does not rerun model inference or repeat segmentation evaluation.

- Test frames: **528** (sequence 1: 261; sequence 3: 267).
- Ground-truth observations: **2663**.
- Frozen tracker: `C_current_baseline`.
- Tracker configuration SHA-256: `bcacebaf22ccc10198994faddd83acab15f838aa412f460365dcb35c9641c2ec`.
- Association IoU: **0.10**; frame-level evaluation IoU: **0.50**.
- Nominal evaluation interval: **2.0 hours**.
- Model inference rerun: **No**; test parameter tuning: **No**.

## Overall test results

| Metric | Result |
|---|---:|
| MOTA | 89.19% |
| IDF1 | 85.56% |
| Detection precision / recall / F1 | 98.61% / 90.69% / 94.48% |
| TP / FP / FN / ID switches | 2415 / 34 / 248 / 6 |
| Strict pooled / macro history coverage | 81.45% / 82.45% |
| Pooled area / width / height RAE | 4.36% / 1.54% / 1.65% |
| Macro growth-direction agreement | 89.75% |
| Adjacent relative-area-change MAE (pooled / macro) | 1.52 / 1.76 percentage points |

### Per-sequence results

| Sequence | Frames | GT observations | MOTA | IDF1 | IDSW | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 261 | 1014 | 94.58% | 96.59% | 1 | 43 |
| 3 | 267 | 1649 | 85.87% | 78.50% | 5 | 205 |

## Metric interpretation

Strict growth histories use a **90% whole-predicted-track purity threshold**. A track below the threshold is excluded in full; the mixed-identity failure is recorded in `test_mixed_identity_track_failures.csv`.

Area, width and height errors compare image-space mask geometry. **No physical-scale calibration is applied**, so the values do not establish centimetre dimensions or biological growth rates.

The normalized relative-area trajectory MAE is a separate **unitless trajectory-shape diagnostic**. It is not a percentage RAE and is not included among the primary geometric accuracy results. The adjacent-interval relative-area-change MAE provides the more direct error in estimated change between consecutive sampled observations.

## Provenance

The source inference manifest records notebook filename `FungiCirno_Real_Inference_Reproducibility.ipynb`. The current project filename supplied for the inference stage is `full-inference.ipynb`. Both are retained explicitly in `tracking_evaluation_run_manifest.json`; the original manifest is not rewritten. Raw source manifest and prediction-cache hashes are recorded alongside the frozen tracker configuration hash.

Charisis et al. values are contextual published references only. The models, datasets, filtering and evaluation protocols differ, so this archive does not claim a controlled head-to-head comparison.

See `METRIC_DEFINITIONS.md`, `FIGURE_CAPTIONS.md`, the machine-readable CSV files, `integrity_report.json` and `tracking_evaluation_run_manifest.json` for full details.
