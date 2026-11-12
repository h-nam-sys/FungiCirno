# Workstream C — Nguyen Hai Nam (lead)

## Ownership

**Core application, integration, shared database/contracts, hardware execution policy, release control, system acceptance, report and presentation.**

## Files you own

- `app.py`
- `src/mushroom_monitor/db.py`
- `src/mushroom_monitor/config.py`
- shared service/contracts and migrations
- `README.md`, release packaging and final docs

## Responsibilities

1. Keep `main` protected. Members work on their dedicated branches and submit pull requests.
2. Review interface/schema change requests before accepting them.
3. Merge Minh Thinh's model branch and Khang's analytics branch independently; resolve conflicts centrally.
4. Install the trained `charisis_cluster_tracking` and `mushr_maturity` target-domain bundles when Thinh provides them. Install M18K WB-only only as a clearly labelled source/debug model; never present it as an oyster-stage model. Configure camera, unit and capture-series metadata.
5. Run the complete acceptance checklist on the target PC: CPU and AMD inference, capture persistence, segmentation, tracking, growth, environment, review, candidate training and rollback.
6. Keep backups of `data/mushroom_monitor.sqlite3`, `data/captures/`, `data/masks/`, and installed models before each integration release.
7. Capture screenshots/video only after the end-to-end workflow is populated with actual inference history.
8. Update the paper with actual side-view transfer, tracking and performance metrics only after corresponding experiments finish. Keep M18K in-domain results separate from side-view target-domain results.

## Definition of done

One clean project checkout can be installed from README, launched locally, run on CPU, run on AMD if supported, persist data across restart, review/correct detections, export reviewed data, install model candidates, show tracking/growth/environment history and restore a previous active model.

## Branch policy

- `main`: release baseline; only Hai Nam merges.
- `member/minh-thinh-ml`: model lifecycle fixes.
- `member/khang-growth-analytics`: tracking and analytics fixes.
- Use pull requests with a short description, files changed, run steps, evidence and known limitations.
