# Model bundles

Import each Colab notebook's `model_bundle.zip` through **Models** in the dashboard. The bundle needs:

- `model_checkpoint.pth`
- `model_config.json`
- `class_mapping.json`

New imports are registered as candidates. Activate a version after checking its validation/test metrics. To roll back, activate the earlier version again.

Keep checkpoints and raw datasets out of Git.
