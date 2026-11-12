# Model storage and taxonomy

- `incoming/`: uploaded model ZIPs staged before installation.
- `registry/<model_key>/<version>/`: installed immutable model versions.

Production model families:

- `charisis_cluster_tracking`: cluster-level oyster stages (`immature`, `well-defined`, `overstayed`), temporal tracking enabled.
- `mushr_maturity`: individual oyster fruitbody maturity (`not-ready`, `ready`, `overdue`), spatial flush summary enabled.
- `m18k_source_wb`: optional source-domain WB-only model (`WB`), not a production oyster model.

All use the same Ultralytics YOLO-seg runtime and normalized Prediction contract. Keep their weights and class mappings separate. See `docs/FINAL_SYSTEM_ARCHITECTURE.md` and `README.md` for package commands and promotion rules.

Weights are not committed to this repository. Every installed candidate must carry model metadata, class mapping and target-domain evaluation metrics before promotion.
