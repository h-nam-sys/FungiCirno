# FungiCirno — final course project report

This is the paper-only release. It contains the editable LaTeX source, the compiled 16-page A4 PDF, and every PDF figure referenced by the report. The full inference notebooks, prediction records, datasets, checkpoints, CSV audits, and run manifests are maintained separately for the project's GitHub research release.

## Files

- `FungiCirno_Final_Report.tex` — editable report source.
- `FungiCirno_Final_Report.pdf` — compiled 16-page report.
- `figures/` — the nine PDF figures referenced by the report.

## Compile

Use a TeX Live installation with the packages loaded by the document. From this directory, run:

```bash
pdflatex -interaction=nonstopmode -halt-on-error FungiCirno_Final_Report.tex
pdflatex -interaction=nonstopmode -halt-on-error FungiCirno_Final_Report.tex
```

The bibliography is embedded in the TeX source; no separate `.bib` file is required. Figure paths are relative to this directory.

## Dataset and reproducibility

The source dataset is not redistributed in this package or the public code repository. Obtain it from the original Zenodo record: <https://doi.org/10.5281/zenodo.15041243> and follow the reuse terms stated there.

The canonical split manifest selects 2,689 frames from the 6,382-image source collection: 1,728 training frames, 433 validation frames, and 528 test frames. Validation uses contiguous windows in sequences 2 and 4, with one guard frame on each side; all other indexed frames in those two sequences are assigned to training. The test sequences (1 and 3) are held out and sampled every eighth source frame. The exact intervals and frame indices are documented in the Methodology section.

The latest full-inference run performed fresh forward passes for both models on all 961 validation/test frames and passed 26 integrity checks. Appendix A identifies the inference notebook recorded in the run manifest, the Kaggle inputs, the model settings, and the analysis notebooks. The complete prediction records, notebooks, metric CSVs, run manifests, and integrity reports are maintained in the separate research evidence release, not duplicated here.

## Scope notes

The report presents an experimental evaluation rather than a deployed application; no system block diagram is included. The processing and evaluation stages are explained in the Methodology section. The main findings concern both instance segmentation and cluster-level growth monitoring. Tracking test results are labelled retrospective because the test sequences were inspected during tracker development.
