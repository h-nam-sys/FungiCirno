# FungiCirno

**Instance segmentation, geometric measurement, and cluster-level growth monitoring of oyster mushrooms (*Pleurotus ostreatus*).**

FungiCirno evaluates instance segmentation for delineating mushroom clusters and examines whether the resulting masks can support image-space measurement and per-cluster growth histories. The project compares a fine-tuned YOLO26-L-seg model with a Mask R-CNN implementation using a ConvNeXt-Tiny-FPN backbone.

## Research paper

Read the [FungiCirno research report](paper/FungiCirno_Final_Report.pdf) before using the models or interpreting the results.

## Pretrained models

The selected model checkpoints and canonical split manifest are distributed through the Kaggle model dataset:

**[FungiCirno Models — Kaggle](https://www.kaggle.com/datasets/hnamedu/fungicirno-models)**

The dataset contains the inference artifacts used by the reproducibility notebook. The notebook verifies the expected checkpoint and manifest SHA-256 hashes before evaluation.

## Run inference and evaluation

Use [`FungiCirno_Real_Inference_Reproducibility.ipynb`](notebooks/FungiCirno_Real_Inference_Reproducibility.ipynb) in a Kaggle notebook with a GPU accelerator.

You must attach both:

1. The [FungiCirno model artifacts](https://www.kaggle.com/datasets/hnamedu/fungicirno-models).
2. The original [Charisis oyster-mushroom dataset](https://doi.org/10.5281/zenodo.15041243), obtained separately under its own reuse terms.

The original image dataset is **not** redistributed in this repository or in the model-artifact dataset. Follow the source dataset's licence and citation requirements.

The notebook performs fresh inference, segmentation evaluation, geometric measurement, tracking, growth-history analysis, integrity checks, and evidence export. It is an experimental reproduction notebook, not a standalone application or production deployment.

## Experimental evidence

The complete evidence archive is published separately to keep this repository compact:

**[Download the full evidence archive — add GitHub Release URL](ADD_GITHUB_RELEASE_URL)**

The archive contains prediction records, evaluation tables, tracking and growth-history outputs, figures, provenance records, and integrity checks. It is separate from the pretrained model dataset.

## Scope and limitations

- Measurements are in image-space pixels; physical dimensions require camera calibration and a validated physical reference.
- The source maturity labels are not predicted by the YOLO model, and the size index is not a maturity classifier.
- Tracking and growth-history test results are retrospective because the test sequences were inspected during development.
- Identity association errors and incomplete histories remain limitations; consult the paper and evidence archive for per-identity results.
- The Mask R-CNN checkpoint is a project implementation using TorchVision and ConvNeXt-Tiny-FPN, not the original Charisis MMDetection checkpoint.

## Citation

See [`CITATION.cff`](CITATION.cff) or [`CITATION.bib`](CITATION.bib). GitHub uses `CITATION.cff` to display a suggested citation for the repository.

## Licence

No code or model-weight licence has been selected in this repository yet. Until a licence is added, do not assume that public availability grants permission to reuse, modify, or redistribute the code or weights. See [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) before making the repository public.
