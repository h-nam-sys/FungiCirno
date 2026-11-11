# Model storage

- `incoming/`: uploaded model ZIPs staged before installation.
- `registry/<model_key>/<version>/`: installed immutable model versions.

Model weights are not committed to this repository. Use `tools/package_yolo_model.py` to create an M18K/Ultralytics model ZIP. See the main README for the bundle format and candidate promotion procedure.
