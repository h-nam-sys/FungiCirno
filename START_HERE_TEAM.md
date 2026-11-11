# FungiCirno — team handoff

This is the integrated implementation handoff for the three-person team. It gives each member a bounded subsystem to validate and repair. It is **not** a claim that every path has already run successfully on the target PC.

## Team ownership

- **Nguyen Hai Nam (lead):** main branch, app, database/schema, shared contracts, integration, system acceptance, report and presentation. See `docs/handoff/HAI_NAM.md`.
- **Nguyen Minh Thinh:** M18K model lifecycle, inference adapter, side-view transfer learning, human-reviewed learning, candidate evaluation and rollback. See `docs/handoff/MINH_THINH.md`.
- **Le Thien Khang:** spatial grouping, temporal tracking, growth measurement, calibration and environmental analysis. See `docs/handoff/THIEN_KHANG.md`.

## Rules that prevent integration damage

1. `main` is owned by Hai Nam. Protect it and do not let members push directly.
2. Both members start from the same release commit and use separate branches:
   - `member/minh-thinh-ml`
   - `member/khang-growth-analytics`
3. Do not change the shared database schema or public contracts without a request to Hai Nam.
4. Do not independently rewrite `app.py`, the DB layer, or the other member's modules.
5. Every submitted fix must include: changed files, exact run command, actual output/log, sample artifacts, known issues, and commit hash.
6. Do not commit model weights, raw datasets, captures, masks, local SQLite databases, or exported review data.
7. Do not call a subsystem complete merely because its code exists. Use the acceptance checklist and attach execution evidence.

## First setup

Use Python 3.12 on Windows.

**AMD PC:**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows_amd.ps1
.\run_windows.bat
```

**CPU-only fallback / second machine:**

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows_cpu.ps1
.\run_windows.bat
```

The app creates a new local SQLite database on first launch. This release intentionally does not include the lead's old local database or any personal capture/mask paths. The team checkpoint is not included either; it remains in the team's authorized Drive/Colab storage.

## First model installation

Obtain the team's M18K WB-only `best.pt`, then package it:

```powershell
python tools/package_yolo_model.py --weights path\to\best.pt --out m18k_wb_bundle.zip
```

Open **Models** in FungiCirno, upload the bundle, and activate it after reviewing the model metadata. WB-only predicts White Button instances; it does not predict oyster-mushroom maturity stages.

## Work order

1. Both members read `docs/INTEGRATION_CONTRACTS.md` and their own handoff document.
2. Minh Thinh validates the M18K bundle, CPU path, AMD path, target-domain fine-tuning and review-to-candidate workflow.
3. Khang validates multi-capture IDs, missed-frame handling, growth measurements, calibration and environment imports.
4. Hai Nam integrates approved branches and runs `docs/ACCEPTANCE_CHECKLIST.md` from a clean checkout.
5. Only after the integrated workflow is populated with real captures and predictions should the team record the final demo video and screenshots.

## What is implemented vs. what still requires evidence

The code provides the application workflows, YOLO-seg and legacy TorchVision model adapters, SQLite persistence, review/export, an offline candidate-training script, model-version activation/rollback, spatial grouping, mask-based temporal association, growth-history views, and CSV environmental ingestion.

The following remain **validation/deployment work**, not established results: full AMD DirectML YOLO inference; full CPU inference on the team's real checkpoint; side-view transfer-learning accuracy; tracking identity metrics on manually annotated sequences; physical calibration accuracy; and an actual incremental retraining + regression + rollback run. Live sensor hardware integration is not included; environmental data enter through CSV.

The `m18k` weights and source/target datasets are external assets and are deliberately not inside this ZIP.
