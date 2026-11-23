# Oyster Mushroom Cluster Monitoring

**Research code and reproducibility resources for instance segmentation, cluster tracking, and image-space growth monitoring of oyster mushrooms (*Pleurotus ostreatus*).**

This repository accompanies the research report [*YOLO26-L-seg for Oyster Mushroom Cluster Segmentation and Growth Monitoring*](paper/Oyster_Mushroom_Cluster_Monitoring_Report.pdf). It provides the inference and tracking notebooks, selected model checkpoints through Kaggle, and the evidence records used to analyse segmentation, geometric measurements, and temporal cluster histories.

## Start here

1. **Read the paper:** [PDF report](paper/Oyster_Mushroom_Cluster_Monitoring_Report.pdf).
2. **Get the model files:** [Oyster Mushroom Cluster Monitoring Models on Kaggle](https://www.kaggle.com/datasets/hnamedu/fungicirno-models). The existing Kaggle dataset URL is retained; its display name is updated in this documentation.
3. **Obtain the original images and annotations:** ["Oyster mushroom clusters growth monitoring timelapses" — Zenodo](https://doi.org/10.5281/zenodo.15041243). The source dataset is not redistributed here.
4. **Run the notebooks in order:** `notebooks/full-inference.ipynb`, then `notebooks/tracking-and-growth-history.ipynb`.

## What the project evaluates

The project compares two instance-segmentation checkpoints:

- **YOLO26-L-seg:** a fine-tuned, one-class cluster segmenter (`mushroom_cluster`).
- **Mask R-CNN:** a project implementation using a TorchVision Mask R-CNN with a ConvNeXt-Tiny-FPN backbone and three source maturity classes, collapsed to a common cluster category for class-agnostic comparison.

The resulting masks are used for image-space area and bounding-box measurements. A separate mask-IoU tracker associates cluster observations across sampled frames and evaluates whether the resulting histories remain consistent with the source identities.

### Selected results

The figures below summarize the reported evaluation under the paper's stated protocols. The tracking and growth-history results are retrospective: the test sequences were inspected during tracker development.

| Evaluation | Result |
|---|---:|
| YOLO26-L-seg mask AP50:95 | **89.02%** |
| Mask R-CNN baseline mask AP50:95 | 57.03% |
| Frozen tracker MOTA | **89.19%** |
| Frozen tracker IDF1 | **85.56%** |
| Strict source-consistent history coverage (pooled / macro) | 81.45% / **82.45%** |
| Strict-history area / width / height RAE | 4.36% / 1.54% / 1.65% |
| Macro growth-direction agreement | **89.75%** |
| Adjacent relative-area-change MAE (pooled / macro) | 1.52 / 1.76 percentage points |

MOTA is the project's custom frame-matching measure, calculated using the equation described in the paper; it is not presented as a standard TrackEval/CLEAR-MOTA implementation. The reported geometric errors are calculated on source-consistent matched observations and are image-space measurements, not physical-size errors.

## Models

The selected YOLO26-L-seg and Mask R-CNN checkpoints and the canonical split manifest are distributed separately:

**[Oyster Mushroom Cluster Monitoring Models — Kaggle](https://www.kaggle.com/datasets/hnamedu/fungicirno-models)**

Checkpoint SHA-256 values, architectures, class mappings, inference settings, and intended uses are documented in [`MODEL_CARD.md`](MODEL_CARD.md). The Kaggle URL intentionally remains the existing dataset URL.

## Reproducing the experiments

The two notebooks have separate roles:

### 1. Fresh inference — `notebooks/full-inference.ipynb`

This notebook locates the model artifacts and source data, verifies the checkpoints and split manifest, and performs or validates the fresh prediction records for the 961 validation/test frames. It exports the prediction caches, segmentation and matched-instance geometry results, figures, provenance, and an integrity report. It also retains the earlier tracker-comparison diagnostics in the full evidence record; those exploratory outputs should not be confused with the final frozen-tracker results reported in the paper.

### 2. Final tracking and growth evaluation — `notebooks/tracking-and-growth-history.ipynb`

Run this notebook using the evidence bundle produced by the inference notebook. It consumes the cached YOLO26-L-seg predictions and source `GT_TRACK_ID` annotations, verifies the frozen validation-selected tracker configuration, and evaluates the 528 sampled test frames. It does **not** rerun segmentation inference or tune tracker parameters on the test set. Its outputs include sequence-level tracking metrics, whole-track purity, strict source-consistent coverage, geometric history errors, adjacent area-change metrics, growth-direction agreement, figures, and an integrity report.

Step-by-step instructions and the expected outputs are in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Dataset access and handling are described in [`DATASET.md`](DATASET.md).

## Research evidence

The [`evidence/`](evidence/) directory contains the machine-readable results, prediction records, provenance, figures, and integrity checks. Start with [`evidence/README.md`](evidence/README.md) for an index and for the distinction between the final frozen-tracker results and historical exploratory tracker snapshots. The canonical final tracking outputs are under [`evidence/growth_tracking/`](evidence/growth_tracking/).

## Scope and limitations

- The YOLO26-L-seg model predicts one cluster class; it does not classify maturity.
- Mask area and bounding-box dimensions are in pixels or pixels squared. Physical dimensions require a validated camera calibration and reference.
- The training-derived relative-size index is not a maturity probability or harvest-readiness classifier.
- Tracking can fragment source identities or merge observations from different identities. The final analysis applies a 90% whole-predicted-track purity rule before using histories for strict growth measurements.
- The test sequences were inspected during exploratory tracker development. Re-running inference does not turn these retrospective test results into an untouched independent test estimate.
- The Mask R-CNN checkpoint is a project implementation using TorchVision and ConvNeXt-Tiny-FPN; it is not the original Charisis MMDetection checkpoint.

## Citation

Use [`CITATION.cff`](CITATION.cff) or [`CITATION.bib`](CITATION.bib) to cite the project. Cite the original Charisis dataset separately using its Zenodo record and follow the source record's citation and reuse terms.

## License

The repository's MIT license applies to the source code and documentation. It does not grant rights to the separately hosted model weights or the original source dataset; see [`MODEL_CARD.md`](MODEL_CARD.md) and [`DATASET.md`](DATASET.md).
