# Reproducing FungiCirno

## Required resources

1. The [FungiCirno Kaggle model dataset](https://www.kaggle.com/datasets/hnamedu/fungicirno-models), containing the selected checkpoints and canonical split manifest.
2. The original [Charisis oyster-mushroom dataset](https://doi.org/10.5281/zenodo.15041243), obtained separately. It is not redistributed here.
3. [`notebooks/FungiCirno_Real_Inference_Reproducibility.ipynb`](notebooks/FungiCirno_Real_Inference_Reproducibility.ipynb).
4. A Kaggle notebook session with a GPU accelerator. The notebook intentionally requires CUDA for the full-resolution inference run.

## Execution

1. Create a Kaggle notebook and import `FungiCirno_Real_Inference_Reproducibility.ipynb`.
2. Attach the FungiCirno model-artifact dataset.
3. Obtain and attach the original source dataset separately, or provide its extracted root through `SOURCE_DATA_ROOT` in the notebook configuration cell.
4. If automatic artifact discovery is ambiguous, set `YOLO_CHECKPOINT_PATH`, `MASKRCNN_MODEL_ROOT`, and `SPLIT_MANIFEST_PATH` explicitly in the configuration cell.
5. Run all cells from top to bottom. The notebook checks the expected checkpoint and manifest SHA-256 values, reconstructs the source sequences and split, performs fresh inference, and computes the reported evaluations.
6. Download the evidence archive generated under `/kaggle/working` after the notebook completes.

The notebook pins Ultralytics to 8.4.214 and retains the Kaggle-provided PyTorch, TorchVision, and CUDA stack rather than reinstalling a potentially incompatible GPU build. It installs missing auxiliary packages such as `pycocotools`, `gdown`, and `psutil` when required.

## Expected outputs

The notebook exports fresh prediction records, segmentation and geometry metrics, tracking ablations, source-identity growth histories, figures, runtime records, provenance information, and an integrity report. The exact archive filename is printed by the final notebook cells.

## Interpretation

The test sequences were inspected during development. A fresh forward pass reproduces the inference procedure but does not make the test set an untouched independent test set. The custom MOTA calculation is project-specific and should not be described as a standard TrackEval/CLEAR-MOTA result. Measurements are image-space quantities unless a separate calibration is supplied.
