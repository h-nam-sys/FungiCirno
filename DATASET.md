# Dataset access and handling

FungiCirno uses the oyster-mushroom time-lapse dataset associated with Charisis et al.

- **Original source:** [Zenodo record, DOI 10.5281/zenodo.15041243](https://doi.org/10.5281/zenodo.15041243)
- **Source content described in the report:** four time-lapse sequences, 6,382 RGB images, 37,004 cluster annotations, and three source maturity labels.
- **Project benchmark:** 2,689 manifest-selected frames: 1,728 training, 433 validation, and 528 test frames.
- **Evaluation subset by sequence:** sequence 1: 261; sequence 2: 235; sequence 3: 267; sequence 4: 198.

The original dataset is not redistributed by FungiCirno. Users must obtain it from the original provider, review the record's current licence and reuse terms, and cite the dataset according to the provider's instructions.

The canonical `split_manifest.csv` is project metadata distributed with the model artifacts. It identifies the benchmark selection; it does not contain the source images or replace the original dataset.

The inference notebook expects the four sequence archives or equivalent extracted sequence directories and the tracked COCO annotation JSON files. It verifies the manifest and checks source files against the expected project provenance. Do not substitute a modified dataset or manifest and describe the run as an exact reproduction.
