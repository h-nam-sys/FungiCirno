# Workstream A — Nguyen Minh Thinh

## Ownership

**M18K model lifecycle: checkpoint, transfer learning, inference adapter, human-reviewed learning, candidate evaluation and rollback.** You are the subsystem validator and bug-fix owner. The integrated implementation already exists in this release; do not start a new model registry or rewrite the whole app.

## Files you own

- `src/mushroom_monitor/inference/`
- `src/mushroom_monitor/model_registry.py`
- `src/mushroom_monitor/learning.py`
- `tools/train_candidate.py`
- `tools/prepare_reviewed_dataset.py`
- `tools/package_yolo_model.py`
- model/training docs and Colab notebook updates

## Required work

1. Locate the team's existing M18K WB-only `best.pt` checkpoint. The weight file is intentionally not committed in this repository.
2. Package it with `python tools/package_yolo_model.py --weights <best.pt> --out m18k_bundle.zip` and install through **Models**.
3. Run a representative image on CPU and AMD DirectML. Confirm source-image mask dimensions, boxes, zero-based class mapping, confidence, count and saved mask areas. DirectML failure must be recorded; CPU must still work.
4. Check the side-view transfer-learning path. Preserve the M18K source baseline; use the target-domain annotation set for fine-tuning and a sequence-/farm-separated holdout for evaluation.
5. Verify reviewed mask/class exports from **Learning lab**. Inspect the actual ZIP contents and several labels before training.
6. Run `tools/train_candidate.py` from Colab using the previous checkpoint and an independent holdout. Confirm the script's base and candidate metrics are measured on the same holdout.
7. Import the candidate bundle. Confirm that a bundle with `promotion_ready=false` cannot be activated unless an explicit override is selected. Confirm the old active model is still available for rollback.
8. Fix bugs only within your subsystem. If you need a schema/interface change, submit the change request to Hai Nam before implementing it.

## Definition of done

- A reproducible M18K bundle installs into the model registry.
- CPU inference works end-to-end.
- AMD DirectML result is either successful with evidence or a documented reproducible blocker.
- Side-view fine-tuning/evaluation artifacts are delivered, or the unavailable dataset/annotation dependency is explicitly reported.
- Reviewed data export, candidate training, holdout comparison, bundle packaging and rollback workflow have been exercised.

## Return package

Provide: source changes on `member/minh-thinh-ml`, checkpoint SHA-256, class mapping, dependency versions, notebook/script, target holdout metrics, sample overlay images, inference logs and a short list of known issues. Do not commit large weights or raw datasets; provide authorized download locations and hashes.

## Do not

- Change SQLite schema or app navigation independently.
- Rename M18K class IDs without updating bundle metadata.
- claim WB-only model output is maturity classification.
- use adjacent time-lapse frames in both train and final test.
- activate a candidate based on training loss alone.
