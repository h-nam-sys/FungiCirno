# FungiCirno — team verification handoff

This release contains the agreed application architecture, shared prediction contract, YOLO-seg model integration, tracking/grouping workflows, review/export lifecycle, training tools, and detailed acceptance tests. The two members are **validators and bounded fix owners**, not asked to redesign the system.

## Fixed architecture

- Production runtime: one Ultralytics YOLOv8-seg inference implementation.
- Model A `charisis_cluster_tracking`: oyster cluster instances, `immature` / `well-defined` / `overstayed`; tracking enabled. This is the primary growth-stage/time-series model.
- Model B `mushr_maturity`: individual oyster fruitbodies, `not-ready` / `ready` / `overdue`; tracking off by default, spatial flush aggregation enabled.
- M18K WB-only `best.pt`: source initialization candidate/debug model only; not an oyster-stage model.
- One normalized `Prediction` contract feeds SQLite, tracking, grouping, review and dashboard. No second application or separate tracker.
- Colab handles training. The local app handles inference, data, review, model registry and rollback. CPU is mandatory; AMD DirectML is the primary optional accelerator and must be verified end-to-end.

Read `docs/FINAL_SYSTEM_ARCHITECTURE.md` for the definitive decisions. Do not alter taxonomies or shared interfaces independently.

## Team ownership

- **Nguyen Hai Nam (lead):** main branch, app, DB/schema, shared contracts, release integration and final system acceptance. `docs/handoff/HAI_NAM.md`.
- **Nguyen Minh Thinh:** model/data/training/runtime validation and fixes. `docs/handoff/MINH_THINH.md`.
- **Le Thien Khang:** temporal tracking, maturity aggregation, growth/calibration/environment validation and fixes. `docs/handoff/THIEN_KHANG.md`.

## Setup

Use Python 3.12 on Windows.

**AMD PC:**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows_amd.ps1
.\run_windows.bat
```

**CPU-only machine:**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows_cpu.ps1
.\run_windows.bat
```

The weights, raw datasets, captures, masks and local SQLite database are external/local assets and are intentionally not committed.

## What members need to return

1. A short report of what works and what does not.
2. The exact commands, environment/package versions, model hashes, dataset/split IDs, logs and sample output artifacts.
3. Focused bug-fix commits only within their ownership area.
4. Any dependency that cannot be resolved from the public datasets or current hardware, stated explicitly.

Do not change DB schema, `app.py`, or shared interfaces without Hai Nam's approval. Do not claim completion from code presence alone.

## Final acceptance

Use `docs/ACCEPTANCE_CHECKLIST.md`. The release is not a scientifically validated production system until both oyster model families have target-domain weights and the CPU/AMD, sequence-tracking, review-learning, and rollback workflows have execution evidence.
