# Metric definitions and interpretation

## Tracking

- **MOTA:** `1 - (FP + FN + IDSW) / GT`, with one-to-one frame-level mask-IoU matching at 0.50.
- **IDF1:** global identity-assignment F1 calculated from IDTP, IDFP and IDFN.
- **Association IoU:** 0.10. This is a tracker parameter and is not the frame-level evaluation threshold.

## Strict source-consistent histories

A predicted track is accepted only if at least 90% of its matched observations belong to its dominant GT identity. A failed track is excluded as a whole. Pooled coverage is based on all GT observations; macro coverage averages coverage across source identities, including the identity with no accepted history. The 90% threshold is applied to predicted-track purity before coverage is calculated; it is not a threshold on per-GT-identity coverage. Accordingly, the per-identity coverage figure does not include a 90% reference line.

## Geometric measurement

Pooled RAE is `100/N * sum(abs(predicted - GT) / max(GT, 1))`. It is computed for area, width and height on strict matched observations. The measurements are image-space px or px²; no physical-scale calibration was applied.

## Temporal area measurements

- **Adjacent relative-area-change MAE:** mean absolute difference between predicted and GT relative area changes over consecutive sampled observations within the same accepted predicted tracklet. The pooled value is 0.015222 (1.52 percentage points); the macro value is 0.017643 (1.76 percentage points) over 2140 transitions.
- **Growth-direction agreement:** per-identity agreement for increase/stable/decrease categories with a 5% deadband, macro-averaged over eligible identities.
- **Normalized relative-area trajectory MAE:** 0.548580, a unitless difference between independently normalized trajectories. It is supplementary and is not reported as a percentage or a relative absolute error.
