# Worklog #10 — Frozen-G GenPC substrate gate

Date: 2026-09-13 (Asia/Seoul)

## Scope and preserved history

This batch continued from Worklog #9 and did not rewrite it. Worklog #9 remains
the historical full-generator replay result: `equivalence_pass=false`, with the
first observed divergence at the InstantMesh GLB, before the fusion-boundary
hook. That result is preserved as generator replay evidence, not relabeled as
an instrumentation failure.

The completed baseline, full downstream replay, and prior boundary capture were
preserved. No existing run directory was deleted or overwritten. The official
upstream checkout remains at commit
`bac9e7b59f4fea0eaaa936f24ef60f342f349829` and was verified clean.

## Canonical frozen inputs

The canonical generated completion is the raw InstantMesh GLB from the first
chronologically successful method-valid baseline lineage, `baseline_07136`.
Selection was by chronology and provenance only; no CD/EMD comparison or quality
cherry-picking was used. InstantMesh was not rerun in this batch.

`G_raw_canonical`:

- Source: `reference_models/genpc/runs/baseline_07136/07136/07136_instantmesh.glb`
- Frozen copy: `reference_models/genpc/runs/frozen_g_substrate_07136_v6/frozen_inputs/G_raw_canonical_07136.glb`
- Byte SHA-256: `ebefc5f601cc99e1f60fe672356207798cbe00482ce295c86a6a67b335c2928d`
- Geometry SHA-256: `654fe0872e513fc74daba04994aff52845c58decbb7c25321086922b91daf75f`
- Geometry: 42,012 vertices, 84,016 faces, one mesh

The matching observed input was frozen from the same baseline lineage:

- `O_raw_registration`: `reference_models/genpc/runs/baseline_07136/07136/color_point.ply`
- Frozen copy: `reference_models/genpc/runs/frozen_g_substrate_07136_v6/frozen_inputs/O_raw_registration.ply`
- Byte SHA-256: `cfee459d78ec37192a6e999c41944d7e6082e5beb0e6157b49812d691f72f01a`
- Geometry SHA-256: `8b524609bcc94d484adf766a2cd9be1e9c5617a4859a7233a099074fd25cc641`
- Point count: 96,403

The frozen GT is `reference_models/genpc/upstream/data/GT/07136.ply`, SHA-256
`94c1d6e37ed3a282233d29882c45879b804902cf51a050a8859ed0d9f57ded72`, with
author evaluation FPS fixed at 16,384 points.

## Upstream generation was not rerun

The batch started only from the frozen O/G inputs. It did not execute Depth
Prompting, cv2 inpainting, ControlNet, Zero123++, or InstantMesh. Generator
provenance remains recorded: InstantMesh config SHA-256
`3f0786add80b310ee1c46f8790e2c796d3c2c0cb20808972a4622e62ef962d8d`, 75
diffusion steps, seed 42, preserved Zero123++ input SHA-256
`610f5a77696bc3cc6a942c9b1cd7761084d4db348cda1a6113a2de5254102690`, and
preserved Zero123++ output SHA-256
`0402eb1cfee08dfef712004b411a4f54af061e6781a936037d30403f1fa955e7`.

## Frozen-G control and instrumented registrations

Both runs used identical frozen O/G bytes, the same official registration
configuration (`redwood`, CUDA, `cd_inv_weight=0.5`, `diff_init=true`,
`reg_fine_xyz=true`, GLB sampling 163,840 points), and the same declared seed
42. Python, NumPy, PyTorch CPU/CUDA RNG state hashes matched before both
registrations. Existing GPU nondeterminism was not tuned away and the control
run was not selected by metric quality.

Control, with no boundary export:

- Root: `reference_models/genpc/runs/frozen_g_control_07136_v6`
- Final fused: 19,703 points; SHA-256
  `2d74001bfe71cebb5f4dbddefad22b4a5727c89cc41a4bcef1861dbee204dad4`
- CD: `0.0515192300081253`
- EMD: `0.0672045946121216`

Instrumented, with only the read-only boundary hook:

- Root: `reference_models/genpc/runs/frozen_g_instrumented_07136_v6`
- Final fused: 19,763 points; SHA-256
  `2f0e8400ffe3d91e86ff7bf36460ecc7d9340443e2945ae056893fc2535e5541`
- CD: `0.0554357208311558`
- EMD: `0.0769357979297638`
- Final transform SHA-256:
  `c99c7288ef2331960f80b152d659f7306d2a8e14e78986809c82de6e9474a6f4`

The control/instrumented metric difference is reported as GPU execution
variation. It is not used to choose a baseline or reject the frozen substrate.

## Boundary identity and transform lineage

The hook ran once immediately before the official
`remove_close_points(source_pcd, target_pcd, distance_threshold=0.0001)` call.
The complete lineage is in
`reference_models/genpc/runs/frozen_g_instrumented_07136_v6/pre_fusion/transform_lineage.json`.
It records `diff_transform`, `coarse_transformation`,
`best_scales_transformation`, `best_transformation_xyz`, and the relevant
inverse matrices.

The following all passed:

- O before/after in-memory identity: pass
- G before/after in-memory identity: pass
- O exported coordinate/order/count identity: pass
- G exported coordinate/order/count identity: pass
- O exact lossless sidecar identity: pass
- G exact lossless sidecar identity: pass

The PLY exports are retained for normal inspection. Because Open3D PLY writing
quantizes colors, the exact captured float point/color arrays are also frozen in
`O_aligned_exact.npz` and `G_aligned_exact.npz`; those sidecars are the future
Experiment 1 boundary inputs.

## Fusion-only sufficiency replay

The replay used only the captured exact sidecars. It did not rerun registration
or generation. It used the official downstream sequence: remove close points,
union, FPS 20,000, statistical noise removal with `std_ratio=2.5`, and final
PLY serialization.

Point accounting:

- O aligned: 88,916 points
- G aligned: 163,840 points
- After close-point removal: 152,968 generated points retained
- Pre-FPS union: 241,884 points
- FPS selected: 20,000 points
- Final: 19,763 points

The replay output is
`reference_models/genpc/runs/frozen_g_fusion_replay_07136_v6/07136/07136_fused.ply`.
Its SHA-256 is exactly the same as the same instrumented run's fused output:
`2f0e8400ffe3d91e86ff7bf36460ecc7d9340443e2945ae056893fc2535e5541`.
Canonical numerical geometry and point accounting both match exactly.

Independent author-metric calls report replay CD `0.0554357208311558` and EMD
`0.0772737637162209`; the independent EMD difference is
`0.000337965786457062`. Repeated read-only evaluation of the same artifact on
the RTX 5080 showed nondeterministic GPU EMD while CD stayed identical. Thus the
metric-input agreement proof is the exact common fused point-array identity;
the separate EMD values are preserved as diagnostics rather than quality
selection. No metric value was cherry-picked.

## Frozen substrate manifest

The complete manifest, including all artifact fingerprints, provenance, RNG
state hashes, configuration, transform lineage, and gate results, is:

`reference_models/genpc/METHOD_VALID_SUBSTRATE_07136_FROZEN_20260913.json`

The finalized execution evidence is:

`reference_models/genpc/runs/frozen_g_substrate_07136_v6/frozen_g_substrate_result_final.json`

The substrate is now frozen at the aligned O/G pair. No Experiment 1 variable,
Structural Association, A partition, or baseline/wrong/oracle A has been
introduced.

EXACT PUBLISHED BASELINE: BLOCKED
GENPC METHOD-VALID SUBSTRATE: READY
EXPERIMENT 1: NOT RUN
