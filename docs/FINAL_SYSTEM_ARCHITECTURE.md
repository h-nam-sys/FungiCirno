# FungiCirno — final system architecture

**Architecture status:** implementation target and team acceptance contract  
**Deployment species:** oyster mushroom, *Pleurotus ostreatus*  
**Primary deployment view:** fixed-camera side view  
**Local target:** AMD Radeon RX 5500 XT 8 GB, with CPU inference always available  
**Training:** offline in Google Colab; the local application never trains in production

## 1. System objective

FungiCirno monitors oyster mushroom development from timestamped images. It must detect and segment mushroom clusters, assign a biological growth-stage label, preserve cluster identity across captures, measure image-plane growth, estimate individual fruitbody harvest maturity, summarize flush readiness, retain environmental context, and support human-reviewed model improvement.

The system must distinguish three different concepts:

1. **Cluster growth stage:** the development class assigned to a whole oyster cluster.
2. **Individual fruitbody maturity:** whether an individual mushroom is not ready, ready, or overdue for harvest.
3. **Growth measurement:** the change in visible mask area for the same tracked cluster over time.

A class label is a model prediction. A change in area is a visual measurement. Neither should be substituted for the other.

## 2. Final model architecture

Use one Ultralytics YOLOv8-seg inference implementation, shared by two model families. They have separate checkpoints, class taxonomies, model registry entries, and intended uses. They do not require separate applications or separate downstream processing pipelines.

### Model A — oyster cluster stage and tracking

| Property | Decision |
|---|---|
| Registry key | `charisis_cluster_tracking` |
| Runtime | `ultralytics_yolo_seg` |
| Input domain | Side-view oyster mushroom time-lapse images |
| Detection unit | Whole mushroom cluster |
| Taxonomy | `immature`, `well-defined`, `overstayed` |
| Training data | Charisis et al. public oyster-cluster instance-segmentation dataset |
| Track objects | Yes |
| Downstream | Persistent cluster IDs, per-capture stage history, area/growth curves |

Class semantics follow the Charisis annotation definitions: immature clusters correspond to early pinning; well-defined clusters show recognizable fruitbody structure including visible stems; overstayed clusters show cap edges bending upward and exposing gills. The labels are based on fruitbody characteristics, not a simple size threshold.

The public Charisis dataset contains 6,382 images and 37,004 annotated cluster instances. It is the primary target-domain training/evaluation source because its species, side-view production setting, cluster-level objects, masks, and growth-stage labels match the core FungiCirno task.

### Model B — individual fruitbody maturity

| Property | Decision |
|---|---|
| Registry key | `mushr_maturity` |
| Runtime | `ultralytics_yolo_seg` |
| Input domain | Oyster fruitbody images from the public MushR dataset |
| Detection unit | Individual fruitbody |
| Taxonomy | `not-ready`, `ready`, `overdue` |
| Training data | Public MushR annotated dataset |
| Track objects | No by default |
| Downstream | Per-fruitbody maturity and spatial flush aggregation |

This is a separate task from cluster-stage tracking. Its three labels must not be merged with the Charisis cluster taxonomy. Per-image spatial grouping combines nearby fruitbody detections; group-level `ready_area_fraction` is based on the predicted mask area belonging to the `ready` class, not the raw number of detections.

The app's common YOLO adapter and normalized prediction record are reused. Model B is run on demand because running both models increases inference time and memory use on the target PC.

### M18K's role

The current M18K checkpoint is a team-trained YOLOv8n-seg model with one class: `0: WB` (White Button). It is not an oyster model and does not predict maturity.

For Model A, use M18K `best.pt` as one possible initialization, but compare it against the standard COCO-pretrained YOLOv8n-seg initialization using the same Charisis train/validation/test split. The M18K source class head must be adapted to the three target classes. Select the initialization that performs better on the held-out oyster target domain; do not assume M18K is automatically superior because it is mushroom-specific.

For Model B, train a separate YOLOv8-seg checkpoint using the MushR annotated fruitbody dataset. The official MushR Mask R-CNN checkpoint may be used as a research/reference baseline, but it is not required by the production application. Keeping both production models on YOLOv8-seg avoids adding another inference backend.

### Taxonomies that are not interchangeable

- Charisis cluster classes: `immature`, `well-defined`, `overstayed`.
- MushR individual classes: `not-ready`, `ready`, `overdue`.
- M18K source class: `WB`.
- Wang's seven-stage taxonomy describes *Pleurotus pulmonarius*. It is a separate taxonomy/species reference, not the production class mapping for the *P. ostreatus* system.

Do not infer a stage from mask area. Do not map WB to one of the oyster stages.

## 3. Local application flow

```text
Site / growing unit / fixed side-view camera
                     |
                     v
      Timestamped capture ingestion
                     |
                     v
  Validation -> RGB conversion -> SHA-256 deduplication
                     |
                     v
               Model registry
                     |
          +----------+-----------+
          |                      |
          v                      v
 Model A: cluster YOLO-seg   Model B: fruitbody YOLO-seg
 3 Charisis cluster stages    3 MushR maturity classes
          |                      |
          v                      v
 Normalized Prediction objects (same interface)
          |                      |
          v                      v
 SQLite predictions + binary masks + model-version ID
          |                      |
          v                      v
 Mask/centroid/area tracking  Spatial DBSCAN-equivalent grouping
          |                      |
          v                      v
 Persistent cluster tracks    Area-weighted flush maturity
 Stage history + growth       Ready-area fraction
          |                      |
          +-----------+----------+
                      |
                      v
        Streamlit dashboard / review queue
                      |
                      v
 Human accept / reject / class correction / mask correction
                      |
                      v
          Reviewed YOLO-seg dataset export
                      |
                      v
             Colab candidate training
                      |
                      v
     Independent holdout evaluation and regression gate
                      |
                      v
 Candidate bundle -> registry -> explicit activation/rollback
```

## 4. Shared inference and data contract

Every model adapter returns the same normalized prediction fields:

- Model-local class ID and class name. YOLO class IDs are zero-based and are preserved exactly as written in `class_mapping.json`.
- Confidence score in `[0, 1]`.
- Bounding box in resized inference-image coordinates.
- Binary instance mask in resized inference-image coordinates.
- Detection index.

`ModelService` scales boxes, mask area and centroid to the source capture dimensions and stores them in SQLite with the capture ID, model version, run ID and mask path. Downstream logic must not inspect Ultralytics result objects or architecture-specific tensors.

The model bundle includes `best.pt`, `model_config.json`, `class_mapping.json`, `evaluation_metrics.json`, and `README.txt`. The config records `runtime_type=ultralytics_yolo_seg`, `prediction_role`, `track_objects`, species, taxonomy, thresholds, and model version.

Use `track_objects=true` for the cluster-stage family. Keep `track_objects=false` for the individual-fruitbody maturity family unless the team later validates individual identity tracking.

## 5. Tracking and growth

Tracking operates on Model A's cluster masks in a fixed camera/view and capture series. The current implementation uses mask IoU, predicted centroid proximity, area similarity, and one-to-one Hungarian assignment. Tracks tolerate a bounded number of missed captures.

A track is scoped by:

`(unit_id, camera_id, series_key, model_version_id)`

Each observation links to its source prediction. This preserves the stage predicted at each capture, even when the track's current class changes. The UI must show both:

- image-plane area and relative area change over time;
- the per-capture predicted stage and confidence.

A stage transition shown in the dashboard is a model-observed class change, not proof that the biological transition occurred at that exact timestamp. The tracker similarity score is a heuristic, not a probability.

`area_px2` is visible 2D mask area. `area_mm2` is calculated only with a valid camera-specific calibration and stable reference plane. These measurements are not volume, biomass, mass, or full 3D size.

## 6. Spatial grouping and maturity summary

Spatial grouping runs per capture and model version. It is not a persistent temporal ID.

- For Model A, class-area fractions describe the distribution of cluster growth-stage predictions within a spatial group.
- For Model B, `ready_area_fraction` is the share of predicted fruitbody mask area assigned to `ready`. It is a descriptive flush-level summary, not a separately trained flush classifier.

The class names and fractions are always interpreted within the active model's taxonomy. Do not compare the numeric values of different taxonomies as if they were the same outcome.

## 7. Human-reviewed incremental learning

The intended offline lifecycle is:

1. Run the active model and store predictions/masks.
2. A human inspects the source image and mask; accepts, rejects, corrects the class, or supplies a corrected mask.
3. Export only accepted/corrected examples for the corresponding model family.
4. Prepare a sequence-/farm-separated train/validation dataset. Keep an untouched target holdout outside the training export.
5. Fine-tune the previous model checkpoint in Colab.
6. Evaluate the candidate on the same untouched holdout as the active model when both have the same taxonomy.
7. For the first transfer from M18K to oyster stages, do not compare WB metrics to oyster-stage metrics. Use target-domain absolute evaluation gates instead.
8. Package as a candidate. Promotion requires passing metrics and explicit human approval.
9. Preserve the previous active version and verify rollback.

High-confidence predictions are not automatically ground truth. Acceptance must mean a human actually inspected the result. The app keeps inference review separate from automatic analysis inclusion.

## 8. Environment

Environmental measurements enter through CSV (`measured_at`, `temperature_c`, `humidity_pct`, `co2_ppm`, `notes`). They are aligned to capture/track observations by timestamp. Correlations are exploratory/descriptive only. Live sensor drivers and causal environmental-response models are outside the current release.

## 9. Compute and deployment

- Training and final model evaluation: Google Colab GPU.
- Guaranteed local execution path: CPU.
- Primary local GPU target: AMD RX 5500 XT 8 GB using DirectML where the complete YOLO graph and operators are supported.
- NVIDIA: future support, not a current acceptance requirement.
- The current 12 GB RAM constraint requires end-to-end memory measurement. Avoid loading both model families at once unless memory tests justify it.

The release includes `tools/test_amd_directml.py`, which attempts a real YOLO segmentation forward pass with the supplied checkpoint and image on DirectML and saves an overlay/metrics result. This is an end-to-end smoke test, not an accuracy benchmark or proof of sustained stability. The acceptance test must also run the real packaged checkpoint through the app and record the actual device, runtime, timing and memory. If DirectML fails, record the reproducible failure and use CPU; never label CPU execution as AMD inference.

## 10. Public data sources and split policy

1. **Charisis et al. oyster cluster time-lapse dataset:** DOI `10.5281/zenodo.15041243`. Use the cluster segmentation labels and three cluster-stage classes. Keep entire time-lapse sequences/farms separated between train, validation and final test.
2. **MushR dataset:** public project repository `https://github.com/ETCE-LAB/MushR`, which links the raw and annotated datasets on Kaggle. Use its annotated individual-fruitbody subset and three maturity classes for Model B. Respect Kaggle and repository dataset terms; do not redistribute raw images unless the dataset licence permits it.
3. **M18K V2:** source-domain WB checkpoint and dataset. Keep the original 450-image WB test split untouched for source-domain reporting; it is not an oyster target test set.

If the Charisis or MushR public annotations cannot be downloaded or redistributed under their terms, report that dependency. Do not invent a replacement taxonomy or claim target-domain validation without target labels.

## 11. Acceptance definition

The application implementation is complete when one clean checkout can:

- install and launch the app;
- import timestamped captures and deduplicate them;
- install either YOLO model bundle and preserve its class mapping;
- run inference and persist predictions/masks on CPU;
- run the actual packaged YOLO model end-to-end on AMD DirectML or document a reproducible blocker;
- display cluster-stage history and growth measurements for a multi-capture fixed-camera series;
- show individual maturity grouping and ready-area fraction for Model B;
- review/correct predictions and export the correct model-family dataset;
- train and package a candidate in Colab;
- gate candidate activation on independent target-domain evaluation;
- activate a candidate and restore the previous model;
- import and display environmental CSV readings descriptively.

The **research/deployment system** is complete only after both target model families have trained checkpoints and the acceptance checklist has real execution evidence. Application code cannot substitute for missing labels, weights, hardware runs or annotated tracking ground truth.

## 12. Research and dataset references

- Charisis et al., *A novel Mask R-CNN-based tracking pipeline for oyster mushroom cluster growth monitoring in time-lapse image datasets*, Computers and Electronics in Agriculture, 2025. Dataset DOI: `10.5281/zenodo.15041243`. The paper reports 6,382 images and 37,004 annotated clusters and defines the three cluster-stage labels.
- MushR, *A Smart, Automated, and Scalable Indoor Harvesting System for Gourmet Mushrooms*, 2023. The public `ETCE-LAB/MushR` repository links its raw and annotated oyster-mushroom datasets. The annotated task distinguishes individual fruitbodies as `not-ready`, `ready`, and `overdue`, then groups fruitbody masks to summarize flush readiness.
- M18K V2 source-domain report: the team's WB-only YOLOv8n-seg metrics apply only to the M18K WB holdout. Keep them separate from oyster results.

The public datasets are source material for model training, not proof of performance on a different farm, camera, cultivar or lighting condition. Confirm dataset licences/terms before redistribution. The local target camera still needs a deployment validation set and tracking-reference subset.
