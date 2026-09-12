# Worklog #9 — GenPC downstream equivalence and fusion-boundary capture

Date: 2026-09-12  
Workspace: `C:\Projects\Kill-OSN-GS`  
Hardware: native Windows RTX 5080  
Sample: `07136` (`sofa`)  
Seed: `42`

## Scope respected

- Reused the completed baseline artifact `runs/baseline_07136/07136/img.png` byte-for-byte.
- Did not rerun Depth Prompting, cv2 inpainting, or ControlNet.
- Did not run TRELLIS, tuning, Structural Association, or Experiment 1.
- Used the authenticated gated RMBG-2.0 revision `5df4c9c76d8170882c34f6986e848ee07fd0ba43`.
- Kept the pristine official GenPC checkout at commit `bac9e7b59f4fea0eaaa936f24ef60f342f349829` unchanged.

## Downstream replay

The separate replay at `runs/instrumented_equivalence_07136_offline` completed RMBG → InstantMesh → registration/fusion. The preserved `img.png`, RMBG mask output, Zero123++ image, and `color_point.ply` matched the completed baseline exactly. The InstantMesh GLB did not match the completed baseline, and the resulting final transform and fused point cloud also did not match.

Strict author-metric comparison failed:

- Completed baseline: CD `0.07481462508440018`, EMD `0.09742892533540726`.
- Full downstream replay: CD `0.06143774464726448`, EMD `0.0757361352443695`.
- Replay result: `equivalence_pass=false`.

Evidence: `reference_models/genpc/runs/instrumented_equivalence_07136_offline/instrumented_equivalence_result.json`.

## Fusion boundary

The corrected boundary capture invokes the official `reg_xyz.remove_close_points` call with a read-only hook immediately before filtering. It exports:

- `O_aligned`: 88,916 points, SHA-256 `ea804a56fd4326707a4590b56da6097881ea217b8dda423acb4b5876d60a79bc`.
- `G_aligned`: 163,840 points, SHA-256 `5fc3cece7021929afc0e92c0cfa66e1b3a391e12e1bba9d826a53f435f45271d`.
- Complete `diff_transform`, `coarse_transformation`, `best_scales_transformation`, `best_transformation_xyz`, and derived inverse matrices.

Evidence:

- `reference_models/genpc/runs/instrumented_boundary_07136/pre_fusion/O_aligned.ply`
- `reference_models/genpc/runs/instrumented_boundary_07136/pre_fusion/G_aligned.ply`
- `reference_models/genpc/runs/instrumented_boundary_07136/pre_fusion/transform_lineage.json`
- `reference_models/genpc/runs/instrumented_boundary_07136/boundary_capture_result.json`

## Decision

Because the strict replay equivalence gate failed, the substrate is not frozen and no Experiment 1 run is allowed. The formal not-ready manifest is `reference_models/genpc/METHOD_VALID_SUBSTRATE_07136_NOT_READY_20260912.json`.

EXACT PUBLISHED BASELINE: BLOCKED  
GENPC METHOD-VALID SUBSTRATE: NOT READY  
EXPERIMENT 1: NOT RUN
