# Coding-AI prompt — Le Thien Khang: verify tracking and growth analytics

You are validating an existing implementation. **Do not build another tracker, another app, or redesign the database.** Read `docs/FINAL_SYSTEM_ARCHITECTURE.md`, `docs/INTEGRATION_CONTRACTS.md`, `docs/ACCEPTANCE_CHECKLIST.md`, this handoff, and inspect the actual source before editing.

## Fixed model decisions

- `charisis_cluster_tracking` provides whole oyster cluster masks and the per-capture classes `immature`, `well-defined`, `overstayed`. This is the temporal tracking/growth-stage model.
- `mushr_maturity` provides individual fruitbody masks and the classes `not-ready`, `ready`, `overdue`. It feeds per-capture spatial grouping and flush readiness summary, not cluster tracking.
- Both models use the same normalized Prediction contract. Never merge class taxonomies.

## Your actual work

1. Validate ordered fixed-camera Charisis time-lapse series, mask-IoU/centroid/area Hungarian association, bounded missed-frame handling and stable track keys.
2. Verify that track observations retain source prediction IDs, class names/stages and confidence. Inspect dashboard stage progression alongside area/growth charts.
3. Inspect crowded/overlapped/new/occluded/partially harvested cases. Where public labels lack persistent identities, create a small manually checked reference-ID subset and report the metric definition and limitations.
4. Validate Model B's spatial grouping and `ready_area_fraction` as a descriptive area-weighted result from its own class taxonomy.
5. Verify model-version scoping, rebuild idempotency, growth math, camera calibration, CSV parsing and time alignment.
6. Fix demonstrable bugs only inside tracking/grouping/environment modules. Coordinate UI/shared schema changes with Hai Nam.

## Return evidence

Sequence IDs/source manifest; reference object IDs if available; overlays; per-track observation CSV; tracking errors/metrics; calibration method; environment CSV tests; commands/logs; commit hash and known issues. Never report physical size without calibration or a tracking metric without ground-truth identities.
