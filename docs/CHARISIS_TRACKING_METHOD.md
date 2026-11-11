# Charisis-inspired temporal cluster tracking

## Purpose

This module associates segmented oyster-mushroom cluster masks across captures from the same camera and capture series. It operationalizes the project's Charisis-inspired workflow: instance/cluster masks are the primary spatial evidence; temporal association then produces persistent cluster identities and area trajectories.

This is an implementation inspired by the described methodology, **not an exact reproduction** of the paper's complete pipeline or a claim that its private/training code was reproduced.

## Current association method

1. **Candidate detections:** use only predictions whose confidence passed the configured application threshold (`final_for_analysis=1`). Human review can accept, reject, or correct the class before rebuilding tracks.
2. **Mask representation:** load each binary mask and downsample it to a fixed square canvas while preserving its aspect ratio.
3. **Motion estimate:** extrapolate the last normalized centroid using the previous two observations, with a short prediction horizon capped at three frames.
4. **Association score:** combine mask IoU (0.55), predicted-centroid affinity (0.30), and mask-area ratio (0.15). Reject pairs outside a normalized spatial gate or with extreme area changes.
5. **Global assignment:** solve the frame-level assignment with the Hungarian algorithm rather than greedy pair ordering.
6. **Occlusion tolerance:** retain unmatched tracks for a configurable number of missed captures. Captures with zero detections still advance this counter.
7. **Growth:** report image-space mask area and percentage change relative to the first observation. Physical area is calculated only when the camera has an explicit mm-per-pixel calibration.
8. **Persistence:** track observations retain frame index, timestamp, area, growth and association score. Track rebuilding is isolated by model version.

Class changes are allowed because a cluster can develop through maturity stages. The latest observed class is stored on the track; per-frame classes remain available through linked predictions.

## Interpretation limits

- Association score is a heuristic similarity score, not a calibrated probability.
- Mask IoU can become unreliable under large camera movement, viewpoint changes, major occlusion, or large time gaps.
- The current tracker does not perform camera-motion compensation, substrate-plane homography, or learned re-identification.
- Area growth is only comparable within a consistent camera/view, framing, resolution and capture protocol.
- A track is not biological ground truth. Review masks and manually verify a sample of trajectories.
- Evaluation requires labelled sequences and identity metrics (e.g. IDF1/HOTA or track precision/recall), plus segmentation metrics. The current software tests validate mechanics, not biological accuracy.
