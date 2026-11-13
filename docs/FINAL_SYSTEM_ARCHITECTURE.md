# FungiCirno — Charisis-first oyster growth monitoring architecture

**Primary deployment species:** oyster mushroom, *Pleurotus ostreatus*  
**Primary view:** fixed-camera side view  
**Primary model:** Charisis-trained TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN  
**Local runtime:** CPU and AMD DirectML paths through the existing TorchVision adapter  
**Training:** offline in Kaggle/Colab; local application performs inference, persistence, review and model management

## 1. System objective

FungiCirno turns timestamped side-view images of oyster mushroom cultivation into persistent cluster-growth histories. Its core workflow is:

1. Detect and segment each oyster mushroom cluster.
2. Assign the cluster's developmental stage.
3. Associate the same cluster across subsequent captures.
4. Measure changes in its visible mask area over time.
5. Preserve images, masks, predictions, tracks and environmental context for inspection.

The core system is **cluster-level growth monitoring**. It does not require a second maturity model to complete that core workflow.

## 2. Primary model — Charisis cluster-stage segmentation

| Property | Current decision |
|---|---|
| Registry family | `charisis_cluster_tracking` (older installed DB rows may use `legacy_charisis_maskrcnn`) |
| Architecture | TorchVision Mask R-CNN + ConvNeXt-Tiny-FPN |
| Runtime | `torchvision_state_dict` |
| Source | Charisis et al. oyster-cluster time-lapse dataset, DOI `10.5281/zenodo.15041243` |
| Object unit | Whole oyster mushroom cluster |
| Class IDs | 1, 2, 3 (TorchVision reserves 0 for background) |
| Dataset class names | `immature_cluster`, `mature_cluster`, `overstayed_cluster` |
| Downstream | Persistent cluster tracking, stage history, visible-area growth curves |

The training notebook preserves the annotation names `immature_cluster`, `mature_cluster`, and `overstayed_cluster`. The middle category corresponds to the mature/well-defined fruiting stage in the source research; do not rename a trained class without updating its mapping and retraining/evaluation metadata together.

The member's notebook is a TorchVision implementation adaptation of the Charisis Mask R-CNN/ConvNeXt-Tiny-FPN model family. It is not an exact reproduction of the original MMDetection 3.3.0 configuration. The trained checkpoint includes the pretrained ConvNeXt backbone weights and is loaded strictly into the matching project model builder. No M18K weights are required to load or run this checkpoint.

### Training data and split used in the submitted run

The full run merged four COCO time-lapse sequences:

| Sequence | Images | Cluster annotations | Use |
|---|---:|---:|---|
| 1 | 2,084 | 8,066 | Held-out test |
| 2 | 1,175 | 8,185 | Training/validation |
| 3 | 2,133 | 13,205 | Held-out test |
| 4 | 990 | 7,548 | Training/validation |
| **Total** | **6,382** | **37,004** | |

The notebook samples every fourth source frame, corresponding to one-hour intervals for the source's 15-minute capture cadence. The resulting sampled split was 433 training images and 109 validation images from sequences 2 and 4, plus 1,055 test images from sequences 1 and 3. Complete sequences are kept separate between training/validation and test to avoid leakage from near-identical adjacent frames.

### Submitted training and test evidence

The attached executed notebook completed all ten configured epochs on a Kaggle Tesla T4. Training loss decreased from 1.0546 to 0.1449; validation loss decreased from 0.6157 to 0.1649. The selected checkpoint was the epoch-10 best-validation checkpoint.

Held-out COCO mask evaluation reported:

| Metric | Overall result |
|---|---:|
| Mask AP@[0.50:0.95] | 0.650 |
| Mask AP@0.50 | 0.782 |
| Mask AP@0.75 | 0.759 |
| Mask AR@100 | 0.765 |

Per-class results from the notebook's category-order evaluation:

| Class | Mask AP@[0.50:0.95] | Mask AP@0.50 | Mask AP@0.75 |
|---|---:|---:|---:|
| `immature_cluster` | 0.711 | 0.844 | 0.821 |
| `mature_cluster` | 0.710 | 0.864 | 0.830 |
| `overstayed_cluster` | 0.529 | 0.638 | 0.626 |

The notebook ran inference on all 1,055 held-out images and exported 9,483 accepted detections. Its own tracking/export code produced 9,483 track observations across 213 predicted track IDs. Those are generated associations, **not** a measured tracking-accuracy score: no verified ground-truth track IDs or MOTA/IDF1/HOTA result is supplied.

## 3. Application tracking and growth measurement

The neural network supplies per-capture cluster masks, class labels and confidence scores. The FungiCirno application performs temporal association separately; it does not rely on the notebook's tracker implementation.

The application tracker:

- operates within one growing unit, camera and capture series;
- compares downsampled binary masks using IoU;
- combines mask overlap with predicted centroid proximity and mask-area similarity;
- uses one-to-one Hungarian assignment;
- tolerates a configurable number of missed captures;
- stores per-observation stage, area, timestamp and association score;
- scopes tracks by model version so one checkpoint cannot overwrite another checkpoint's tracks.

The association score is a heuristic, not a probability. A cluster that does not reappear within the allowed gap is marked as no longer observed; this does not prove it was harvested.

Visible-area growth is calculated as:

\[
G_t = 100\left(\frac{A_t}{A_0}-1\right)
\]

where `A0` and `At` are source-image mask areas for the same track. `area_px2` is a projected 2D measurement, not biomass, mass, volume or full 3D size. `area_mm2` is available only with valid camera-specific calibration and stable geometry.

## 4. Local ingestion and runtime compatibility

The current project already contains the matching `build_charisis_model()` implementation and TorchVision checkpoint loader. The loader:

1. Reads `model_checkpoint.pth` from the Charisis bundle.
2. Builds Mask R-CNN with ConvNeXt-Tiny-FPN using the bundle's class count and inference configuration.
3. Loads the checkpoint's `model` state dictionary with `strict=True`.
4. Returns the same normalized `Prediction` objects used by the database, review, tracking and dashboard layers.

No YOLO conversion or second application is required. The registry now assigns new ConvNeXt Charisis bundles the canonical `charisis_cluster_tracking` family key. Existing `legacy_charisis_maskrcnn` database records remain recognized for backward compatibility.

The model bundle contract is:

```text
model_checkpoint.pth
model_config.json
class_mapping.json
evaluation_metrics.json       # included when available
MODEL_IMPLEMENTATION_NOTE.txt
```

The class mapping preserves the trained TorchVision IDs 1–3. Class ID 0 is reserved for background by Mask R-CNN. Do not apply the YOLO zero-based remapping to this bundle.

The provided project archive does not include the binary `.pth` checkpoint. The executed notebook's final cell writes `model_bundle.zip` to the Kaggle notebook output directory; download that bundle separately for a clean installation. The uploaded notebook and project SQLite snapshot contain model/run metadata, not the checkpoint bytes.

### Local integration evidence in the supplied project snapshot

The supplied SQLite snapshot records 15 successful Charisis Mask R-CNN inference runs using AMD DirectML. Across those runs it records 96 candidate predictions, 43 predictions passing the configured 0.20 analysis threshold, and 22 application track records containing 43 observations. On the 15 runs, median model inference time was approximately 0.56 seconds and median end-to-end run time approximately 2.15 seconds; the input was 6080×3420 and the UI test used a 768-pixel inference long side. These are observations from a small local integration sample, **not** a controlled performance benchmark or accuracy evaluation.

The archive's earlier `legacy_charisis_maskrcnn` record is retained for compatibility. The project now treats this as the primary cluster model rather than a legacy experiment.

## 5. Optional individual fruitbody maturity extension

A separate individual-fruitbody classifier is needed only if the lead requires harvest-readiness estimates for each mushroom, rather than cluster-level developmental stages alone.

The optional MushR task uses individual fruitbody masks and the separate classes `not-ready`, `ready`, and `overdue`. Its spatial grouping output is a per-image flush summary, not a persistent temporal cluster identity. This task is not required for the core Charisis cluster-growth pipeline and must not be merged into the Charisis class taxonomy.

The project retains a TorchVision ResNet-50-FPN builder for a future compatible MushR Mask R-CNN bundle. The official Detectron2 TorchScript artifact is a different runtime contract and cannot be renamed to `.pth` or loaded by the current state-dictionary loader. Do not make MushR deployment a release blocker unless individual harvest readiness is explicitly in scope.

## 6. M18K's role

M18K V2 is a White Button mushroom dataset. The team's WB-only YOLOv8n-seg checkpoint and its reported metrics remain a historical source-domain segmentation experiment. It is not an oyster cluster model, does not predict Charisis growth stages, and is not required to initialize the trained Charisis checkpoint.

Retain M18K artifacts for benchmark/report history and optional generic YOLO runtime testing. Do not use M18K performance as evidence of oyster growth-monitoring performance.

## 7. Environmental context and human review

Environmental CSV measurements (`measured_at`, `temperature_c`, `humidity_pct`, `co2_ppm`, `notes`) are associated with growing units and timestamps. They support descriptive comparison with growth history; the project does not claim causal environmental-response modelling.

Human review keeps model confidence separate from annotation truth. Reviewers can accept, reject, correct class labels and replace masks. The generic COCO review export can preserve the Charisis one-based class IDs and taxonomy. The YOLO-seg export is only for YOLO candidates. The Charisis TorchVision notebook does not yet automatically ingest the review export; a sequence-aware reviewed-data merge is needed before claiming a complete Charisis incremental-learning loop.

## 8. Acceptance priorities

Core growth pipeline:

- install/load the Charisis Mask R-CNN bundle;
- run inference on CPU and AMD DirectML;
- inspect masks and per-class predictions;
- process timestamped fixed-camera sequences;
- verify Hungarian tracking and missed-frame handling;
- compare track IDs and growth curves with a manually checked reference subset;
- validate area scaling and camera calibration;
- confirm persistence, review, export and dashboard behavior.

Optional extension:

- train/integrate MushR individual maturity only if the lead requires per-fruitbody harvest readiness.

Do not claim validated production growth monitoring until tracking identity and growth-measurement quality have been checked against independent reference data.
