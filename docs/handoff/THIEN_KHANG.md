# Workstream B — Le Thien Khang: tracking and growth analytics verification

## Ownership

Validate and repair the existing tracking, grouping, growth, calibration and environmental modules. Do not build a separate tracker or redesign the application.

## Inputs

- Primary tracking input: predictions from `charisis_cluster_tracking`, with whole oyster cluster masks and Charisis's three stage labels (`immature`, `well-defined`, `overstayed`).
- Supplementary maturity input: `mushr_maturity`, with individual fruitbody masks and (`not-ready`, `ready`, `overdue`) labels. This model feeds per-image spatial grouping and flush readiness summaries; it is not the temporal cluster tracker.

The normalized Prediction schema is shared. Do not change model class IDs or taxonomy.

## What to verify and deliver

1. Use the public Charisis side-view time-lapse sequences with the same camera/view per series. Verify capture ordering, missing timestamp fallback, and that separate sequences are not mixed.
2. Verify the existing Hungarian association uses mask IoU, predicted centroid proximity and area ratio, one-to-one matching and a bounded missed-frame gap. Inspect crowded frames, overlapping masks, new pins, occlusions and partial harvesting.
3. Verify stable track keys and that each track observation links to its source prediction, preserving the per-capture predicted stage and confidence. Confirm stage transitions display alongside area/growth curves.
4. Build a small manually checked tracking reference subset from public time-lapse images if the source does not supply persistent object IDs. Report identity switches, IDF1/MOTA or another clearly defined sequence metric only when reference IDs exist. Do not label heuristic association scores as probabilities.
5. Verify Model B spatial groups are per capture/model version, not persistent track IDs. Verify class-area percentages and `ready_area_fraction` are derived from Model B's own taxonomy only.
6. Verify model-version-scoped rebuilds do not overwrite another model's groups or tracks. Check empty captures, gaps, reappearance and rerun idempotency.
7. Verify relative growth from source-image mask area. `mm^2` must remain unavailable without a valid camera-specific calibration and fixed reference plane. Document how calibration was measured.
8. Import environmental CSV files and test malformed timestamps, valid ranges, duplicate source rows, nearest-reading alignment and the 12-hour match limit. Keep analysis descriptive, not causal.
9. Fix demonstrated issues only inside your modules and focused UI outputs. Ask Hai Nam before shared schema/interface changes.

## Files you own

- `src/mushroom_monitor/tracking.py`
- `src/mushroom_monitor/maturity.py`
- `src/mushroom_monitor/environment.py`
- focused tracking/analytics UI and tests, coordinated with Hai Nam for `app.py`

## Definition of done

- Multi-capture oyster cluster sequence yields explainable stable tracks and per-observation stage history.
- Missed captures/temporary occlusion do not instantly destroy a track.
- Tracking results are evaluated against a labelled reference subset, or the absence of persistent public IDs is documented.
- Individual maturity and cluster-stage taxonomies remain separate.
- Growth and physical calibration outputs are traceable and honest.
- Environmental import/alignment works with real CSV examples and survives restart.

## Guardrails

Do not change model weights or class mappings. Do not treat per-image groups as persistent biological flush IDs. Do not infer 3D size or biomass from 2D area. Do not report a tracking metric without ground-truth identities.

## Return package

Branch `member/khang-growth-analytics`; changed files and commit hash; sequence/source manifest; overlays; track observation CSV; reference IDs and metric definitions; calibration evidence; environment test CSV/output; logs and known issues.
