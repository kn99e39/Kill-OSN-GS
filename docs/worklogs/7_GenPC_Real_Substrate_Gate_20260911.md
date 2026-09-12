# 7 GenPC Real Substrate Gate

**Date:** 2026-09-11  
**Scope:** Build and audit the published CVPR 2025 GenPC reference path on the MAIN RTX 5080 workstation. This batch did not run Experiment 1.

## Outcome

The Blackwell PyTorch runtime is valid, and the author-provided ControlNet path loads successfully. The real GenPC substrate is **not ready** because the published depth-inpainting path is not fully mapped in the GenPC release and the InstantMesh path cannot load on this Windows host: the required nvdiffrast extension has no matching build toolchain (MSVC is absent and the installed CUDA toolkit is 13.3 while PyTorch is cu128).

## Frozen source and configuration

- GenPC is isolated at `reference_models/genpc/upstream`, pinned to `bac9e7b59f4fea0eaaa936f24ef60f342f349829`, detached and clean, with the author MIT license preserved.
- The published contract is recorded in `reference_models/genpc/PUBLISHED_METHOD_CONFIG_20260911.json`.
- The exact audit evidence is recorded in `reference_models/genpc/GENPC_REFERENCE_AUDIT_20260911.json`.
- InstantMesh and DDNM were fetched into isolated dependency directories only; the pristine GenPC checkout was not modified.

## Runtime gate

The isolated environment uses Python 3.11.9, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, and torchaudio 2.8.0+cu128. PyTorch reports one RTX 5080 at capability 12.0, 16,303 MiB, with `sm_120` in its architecture list. This is a documented hardware-compatibility/runtime deviation from the GenPC README’s Python 3.10 and PyTorch 2.6/cu126 instructions; method settings were not changed.

## Method mapping and checks

- Depth Prompting contract: 2,048 cameras, 256 resolution, distance 1.6, FOV 49.1°, point size 1/2, mask rate 3, and 256×256 inpainting.
- ControlNet: the author module identifies SDXL base, `xinsir/controlnet-depth-sdxl-1.0`, and the `madebyollin` VAE. All load successfully on the RTX 5080; peak load memory was 9,617,387,008 allocated bytes and 9,925,820,416 reserved bytes. No image was generated.
- InstantMesh: the author module identifies Zero123++ v1.2, the official InstantMesh base checkpoint, 75 diffusion steps, scale 1, distance 4.5, and six views. Loading stopped before model acquisition because `nvdiffrast` was unavailable. Building the official nvdiffrast source then failed on CUDA 13.3 versus PyTorch CUDA 12.8 mismatch and missing `cl`.
- Depth inpainting: the pristine source defaults to cv2 and references a missing `models.DDNM.ddnm_inpainting` adapter. The isolated official DDNM checkout provides the general guided-diffusion/ DDΝM implementation and checkpoint reference, but not the GenPC adapter. This remains a reproduction ambiguity.
- Entry point: `main.py` calls `getImage(..., img_gen=False)` while the next ScaleAdapter stage expects `img.png`. This is recorded as an un-applied `ENTRYPOINT_ORCHESTRATION_FIX`; the pristine source remains unchanged.

## Sample and experimental boundary

The deterministic official-entry sample is `07136` (`sofa`). Input and GT hashes are in the JSON audit manifest. No baseline completion, CD/EMD evaluation, pre-fusion export, equivalence run, O/G freeze, association definition, or Experiment 1 execution occurred.

## Handoff

Close the DDNM mapping and provide a matching CUDA/MSVC or approved Linux/WSL build path for nvdiffrast and Kaolin. Only then apply the minimal orchestration correction, run one published baseline, and instrument the identified boundary in `reg_xyz.py`.

**GENPC REAL SUBSTRATE: NOT READY**
