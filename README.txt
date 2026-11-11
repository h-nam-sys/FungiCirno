FungiCirno — integrated implementation handoff

Read README.md first. This release contains the application, M18K YOLO-seg adapter, model registry, tracking/growth/environment workflows, reviewed-data export, candidate fine-tuning/evaluation scripts, and member runbooks.

No trained model weights or raw datasets are included. Obtain the team's M18K best.pt checkpoint from the authorized source and package it with tools/package_yolo_model.py. Side-view target-domain weights require transfer learning and an independent holdout evaluation.

Team ownership:
- Nguyen Minh Thinh: model lifecycle and incremental learning (docs/handoff/MINH_THINH.md)
- Le Thien Khang: tracking, measurements and environmental analytics (docs/handoff/THIEN_KHANG.md)
- Nguyen Hai Nam: core app, integration and final acceptance (docs/handoff/HAI_NAM.md)

The ZIP deliberately excludes the existing local SQLite database, captures, masks, and model checkpoints. Back up your local data before applying this release to an existing installation.
