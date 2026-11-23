# Figure captions

1. **Tracking performance by sequence** (`fig_tracking_mota_idf1_by_sequence.png`). MOTA and IDF1 on test sequences 1 and 3, calculated using the frozen tracker and the stated evaluation protocol.
2. **Frame-level tracking detections and errors** (`fig_frame_level_errors_over_time.png`). True positives, false positives and false negatives per sampled test frame, plotted against nominal elapsed time.
3. **Purity of matched predicted tracks** (`fig_track_identity_purity.png`). Distribution of predicted-track source-identity purity; the dashed line indicates the 90% whole-track strict-history threshold.
4. **Predicted track-length distribution** (`fig_track_length_distribution.png`). Number of sampled observations assigned to each predicted track.
5. **Strict source-consistent geometric measurement error** (`fig_geometric_measurement_error.png`). Absolute relative error in area, width and height on observations retained after whole-track purity filtering.
6. **Growth-direction agreement by sequence** (`fig_growth_direction_agreement.png`). Macro agreement of increase/stable/decrease classifications across source identities with eligible adjacent observations.
7. **Representative strict growth histories** (`fig_representative_growth_trajectories.png`). Predicted and annotated visible mask area for one long accepted tracklet in each test sequence.
8. **Per-observation area agreement** (`fig_area_measurement_agreement.png`). Predicted versus annotated mask area on strict matched observations; values are image-space px².
9. **Temporal mask association progression** (`fig_temporal_mask_association_progression.png`). Predicted mask silhouettes and GT/prediction overlap at three observations of one accepted tracklet. This figure shows masks only, not original RGB frames.
10. **Matched mask-IoU distribution** (`fig_matched_mask_iou_distribution.png`). Mask IoU for frame-level TP matches, with the 0.50 evaluation threshold.
11. **Strict source-consistent coverage by identity** (`fig_strict_history_coverage_by_identity.png`). Coverage for each annotated source identity after whole-track purity filtering. No 90% reference line is shown: the 90% criterion applies to predicted-track purity, not to per-identity coverage.
12. **Mixed-identity track failure** (`fig_mixed_identity_track_failure.png`). Composition of the predicted track excluded in full because its source-identity purity fell below 90%.
13. **Adjacent-interval relative-area-change agreement** (`fig_adjacent_relative_area_change_agreement.png`). Predicted versus annotated relative area change between consecutive sampled observations within the same accepted tracklet; the diagonal denotes exact agreement.
14. **Adjacent-interval area-change error by sequence** (`fig_adjacent_area_change_error_by_sequence.png`). Distribution of absolute error in relative area change, expressed in percentage points, for each sequence.
