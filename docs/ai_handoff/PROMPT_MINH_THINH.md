# Coding-AI prompt — Minh Thinh: verify the model workstream

You are validating an existing implementation. **Do not redesign FungiCirno or invent a second inference system.** Read `docs/FINAL_SYSTEM_ARCHITECTURE.md`, `docs/INTEGRATION_CONTRACTS.md`, `docs/ACCEPTANCE_CHECKLIST.md`, this handoff, and then inspect the actual source code before changing anything.

## Fixed model decisions

- `charisis_cluster_tracking`: YOLOv8-seg; oyster cluster masks; zero-based classes exactly `immature`, `well-defined`, `overstayed`; tracking on.
- `mushr_maturity`: YOLOv8-seg; individual oyster fruitbody masks; zero-based classes exactly `not-ready`, `ready`, `overdue`; temporal tracking off, spatial flush summary on.
- `m18k_source_wb`: one-class WB source model only. It is not an oyster classifier.
- One Ultralytics adapter and one normalized Prediction contract serve all three. Do not mix the taxonomies.

## Your actual work

1. Inspect the real Charisis and MushR public datasets, annotation formats, licences/terms, class names, camera/sequence metadata and split manifests.
2. Validate/fix the generic COCO-to-YOLO conversion where appropriate; write a scoped converter only if the actual annotation format requires it. Inspect labels and masks after conversion.
3. Train Charisis Model A twice on identical target splits: M18K initialization and COCO YOLOv8n-seg initialization. Compare both on the same untouched oyster test sequences. Do not compare WB AP against oyster AP.
4. Train MushR Model B as an independent task/model family. Do not use Model A as its baseline.
5. Verify model bundles and exact class order; CPU app inference is mandatory. Test AMD DirectML full inference on the actual RX 5500 XT; report exact blocker if unsupported.
6. Exercise human-reviewed export, offline candidate training, same-taxonomy holdout regression, promotion gate and rollback.
7. Submit only evidence-backed fixes in your ownership area. Preserve shared schema/contracts.

## Return evidence

Dataset and licence/source info; split manifests; exact commands; training settings; checkpoint hashes; holdout metrics; overlays; model ZIPs (weights shared outside Git); CPU/AMD logs; RAM/latency; dependency versions; commit hash and known issues. Do not claim target-domain readiness from M18K metrics.
