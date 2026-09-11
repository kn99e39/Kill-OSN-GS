# 1 Experiment Environment Setup Batch

**Dates:** 2026-09-10 to 2026-09-11
**Scope:** Build an auditable Experiment 1 substrate and a GPU-capable reference environment. This batch explicitly excluded Experiment 1 execution, scientific evaluation, model inference, and automatic geometry processing.

## Objective

Prepare the LabServer63 RTX 3080 Ti host and the repository so that a reference-model reproduction can be attempted without changing the experimental contract or confusing setup evidence with scientific results.

## Work completed

- Created the immutable Experiment 1 contract, raw-artifact lineage, region-partition, fixed-evaluation, association-declaration, and content-addressed manifest components under `structural_experiment/`.
- Added read-only PLY/GLB geometry ingestion and real-data partition materialization that retains raw vertex and face ownership.
- Added synthetic invariant, artifact, substrate, and contract tests.
- Pinned the official LaS-Comp source checkout at `0da46f2569b484986d3737d97213428db962161f` under the server-side ignored reference-model directory.
- Added a bootstrap script that verifies the canonical source blobs before environment setup.
- Installed a user-local Miniforge environment on LabServer63 at `/home/nd/.local/conda-envs/lascomp-py310-cu121`.
- Verified the isolated runtime: Python 3.10, PyTorch `2.4.0+cu121`, torchvision `0.19.0+cu121`, CUDA runtime 12.1, and an available NVIDIA GeForce RTX 3080 Ti.

## Verification

- The bootstrap source-blob checks passed for the official environment file, requirements file, public entry point, and flow implementation.
- The full synthetic test suite passed: **20 passed**.
- Contract and artifact checks confirmed that evaluation selectors are independent of association declarations and that only an explicitly declared association may differ across compared manifests.
- The server reference source checkout and the outer repository worktree were clean after verification.

## Environment decision

The official Conda environment was not used unchanged because its current CPU Torch build encountered the known `iJIT_NotifyEvent`/MKL incompatibility. The isolated runtime instead uses the official PyTorch CUDA 12.1 wheel channel while retaining the upstream-requested PyTorch 2.4, torchvision 0.19, Python 3.10, and CUDA 12.1 compatibility target. This is recorded as an environment-distribution deviation, not a model-semantic change.

## Boundaries preserved

No checkpoint was downloaded, no official model was imported or executed, no inference was run, and no Experiment 1 result was generated. No automatic segmentation, graph construction, NURBS processing, or tuning was introduced.

## Outcome and handoff

The hardware-compatible core environment and experimental substrate are ready. Pristine LaS-Comp reproduction remains gated on resolving authoritative dependency and checkpoint provenance; that later prerequisite audit is recorded in worklog 2.

**Related commits:** `b595e9be7e68750dd9166c29d9f0b9b4aa0bdc51`, `3c27fe1a34557c63b2b273abd42fec392bb5207c`
