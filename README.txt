FungiCirno — oyster mushroom growth-monitoring application

Open README.md for install/run instructions.
Open docs/FINAL_SYSTEM_ARCHITECTURE.md for the fixed system design.
Open START_HERE_TEAM.md for the member verification plan.

Primary model: charisis_cluster_tracking (oyster cluster-stage segmentation + temporal tracking).
Secondary model: mushr_maturity (individual oyster fruitbody maturity + flush aggregation).
M18K WB-only best.pt is a source initialization candidate, not a production oyster model.

Trained oyster model weights and raw public datasets are external assets and are not included. Full CPU/AMD, target-domain, sequence-tracking, and learning/rollback evidence must be collected before calling the system validated.
