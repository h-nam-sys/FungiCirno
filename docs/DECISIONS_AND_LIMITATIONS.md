# Architecture decisions and claim limits

## Final decisions

1. **One inference implementation:** Ultralytics YOLOv8-seg is the production runtime for both model families. The legacy TorchVision loader remains only for importing older experimental bundles; it is not required for the production workflow.
2. **Two distinct model families:**
   - `charisis_cluster_tracking`: whole oyster clusters, three Charisis cluster-stage labels (`immature`, `well-defined`, `overstayed`), persistent tracking enabled.
   - `mushr_maturity`: individual oyster fruitbodies, three MushR maturity labels (`not-ready`, `ready`, `overdue`), spatial flush aggregation enabled.
3. **M18K role:** the WB-only YOLOv8n-seg checkpoint is a source-domain initialization candidate. It is not the deployed oyster model and does not predict maturity. Compare it with COCO-pretrained YOLOv8n-seg on the same Charisis target split before selecting an initialization.
4. **Target domain:** fixed-camera side-view oyster mushroom monitoring, primarily *Pleurotus ostreatus*. Target-domain results must come from oyster-labelled data and sequence/farm-separated evaluation.
5. **Research taxonomy:** Wang's seven-stage taxonomy concerns *Pleurotus pulmonarius* and remains separate. Do not map it into the *P. ostreatus* class sets without a new labelled training/evaluation design.
6. **Tracking:** Charisis informs mask-based association and missed-detection handling. The local Hungarian tracker is a project implementation, not an exact reproduction of the paper.
7. **Runtime:** CPU inference must work. AMD DirectML is the primary optional accelerator and must be demonstrated with the real YOLO model end-to-end. No silent fallback.
8. **Training:** offline in Colab. The local application performs inference, persistence, review and model management; it does not train in production.
9. **Persistence:** SQLite stores captures, model versions, inference runs, predictions, masks, tracks, observations, review records and environmental measurements.
10. **Learning lifecycle:** human-reviewed exports -> offline fine-tuning -> independent holdout evaluation -> candidate bundle -> explicit activation -> rollback.
11. **Environment:** CSV import and time alignment are included. Live sensor integration and causal environmental-response modelling are not.

## Model and taxonomy boundaries

- Charisis: cluster-level `immature`, `well-defined`, `overstayed`.
- MushR: individual-level `not-ready`, `ready`, `overdue`.
- M18K: source-domain `WB` only.
- Wang: separate seven-stage *P. pulmonarius* taxonomy.

Never compare these class IDs or class-area fractions across model families. A WB detection is not a maturity estimate. A cluster-stage label is not an individual fruitbody harvest label.

## Measurement limits

- `area_px2` is projected two-dimensional mask area in source-image pixels.
- `area_mm2` is computed only with a camera-specific calibration valid for the fixed reference plane.
- Relative area change is not biomass, volume, mass, or full 3D growth.
- Track association scores are heuristics, not probabilities.
- Spatial groups are per-image connected groups, not persistent biological flush identities.
- Group maturity summaries are descriptive aggregates of model outputs, not independent classifiers.
- Environmental correlations are descriptive, not causal.

## Evaluation limits

- M18K WB metrics are source-domain results, not oyster side-view accuracy.
- Charisis target evaluation must keep full time-lapse/farm sequences separate across train, validation and test.
- MushR training/holdout must retain its published dataset split or an equally strict sequence-separated split.
- Predictions require human inspection before they become learning labels.
- A validation split used for checkpoint selection is not the final untouched test set.
- Unit tests and mocked inference do not prove full-model AMD DirectML support, tracking accuracy or biological validity.

## Remaining evidence dependencies

- Download/licence confirmation and split manifest for Charisis and MushR datasets.
- Trained oyster cluster-stage YOLO bundle.
- Trained individual maturity YOLO bundle (unless deliberately deferred and reported as such).
- Full CPU inference and memory measurement on the real target bundles.
- Full AMD DirectML YOLO inference on the RX 5500 XT or a reproducible blocker.
- Tracking identity evaluation on annotated multi-capture sequences.
- Camera calibration validation.
- One end-to-end reviewed export -> fine-tune -> holdout gate -> activation -> rollback run.
