# Charisis model and FungiCirno temporal tracking

## Model role

The Charisis-trained Mask R-CNN detects whole oyster mushroom clusters and assigns one of three developmental labels: `immature_cluster`, `mature_cluster`, and `overstayed_cluster`. It outputs instance masks, boxes, class IDs and confidence scores. It does not assign persistent identities by itself.

## Training checkpoint

The member's executed notebook trained TorchVision Mask R-CNN with a ConvNeXt-Tiny-FPN backbone. It used the Charisis COCO annotation set, source sequences 2 and 4 for training/validation, and sequences 1 and 3 as held-out test. Frames were sampled every fourth capture (one-hour cadence). It completed 10 epochs on a Kaggle Tesla T4.

Held-out mask metrics:

- overall AP@[.50:.95]: 0.650
- AP50: 0.782
- AP75: 0.759
- AR100: 0.765

The notebook exported predictions and a separate sequence-tracking CSV. Its 213 predicted tracks are not a tracking accuracy score because ground-truth persistent IDs were not provided in the evidence.

## Application tracker

FungiCirno's tracker is a separate implementation that consumes the normalized prediction masks from the Charisis model:

1. Load only the latest inference run for each capture/model version and retain detections accepted for automatic analysis.
2. Resize masks to a bounded square canvas while preserving aspect ratio.
3. Predict the next normalized centroid from recent observations, with a short prediction horizon.
4. Score candidate associations using mask IoU (0.55), predicted-centroid affinity (0.30), and area ratio (0.15).
5. Reject pairs outside spatial/area gates, then solve one-to-one assignment with the Hungarian algorithm.
6. Keep unmatched tracks for a configurable missed-capture gap; empty captures still advance the gap counter.
7. Store each observation's linked prediction, per-frame class, timestamp, source-image area and relative change from the first observed area.

This is FungiCirno's tracker, not a verbatim implementation of Charisis et al. The notebook's own tracker uses a separate candidate-matching routine. The two outputs must not be conflated.

## App integration evidence

The supplied project database records 15 successful Charisis inference runs through AMD DirectML. It contains 96 candidate predictions; 43 passed the 0.20 automatic-analysis threshold. The application tracker produced 22 track records containing 43 observations. This establishes that the model was loaded and exercised through the application on a small capture sample. It does not establish tracking accuracy.

## Interpretation limits

- An association score is a heuristic, not a probability.
- A track closing after a missed-frame gap means it was no longer observed; it does not prove harvest/removal.
- Growth curves measure visible 2D mask area. They are not biomass, mass, volume or 3D growth.
- Calibration must be tied to the fixed camera and reference plane.
- Validate identity continuity against manually annotated ground-truth tracks before reporting IDF1, HOTA or MOTA.
