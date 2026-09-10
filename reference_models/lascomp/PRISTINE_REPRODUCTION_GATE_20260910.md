# LaS-Comp pristine-reproduction gate — 2026-09-10

This record concerns the reference-model environment only.  It is not an
Experiment 1 result and no official checkpoint, benchmark sample, inference,
attachment solver, segmentation, or graph procedure was run.

## Verified on `LabServer63`

- Official source: `https://github.com/DavidYan2001/LaS-Comp.git`
- Detached source commit: `0da46f2569b484986d3737d97213428db962161f`
- Working tree: clean after `bootstrap_reference.sh` validated the source Git
  blobs (not platform-dependent checkout bytes).
- GPU: NVIDIA GeForce RTX 3080 Ti, 12,288 MiB; driver `595.71.05`.
- Isolated core runtime: Python 3.10, `torch==2.4.0+cu121`,
  `torchvision==0.19.0+cu121`; `torch.cuda.is_available()` is `True` and CUDA
  reports `12.1`.
- The official environment pins the same Python/PyTorch/torchvision/CUDA minor
  versions.  The runtime uses official PyTorch CUDA 12.1 wheels because the
  Conda build encountered the documented `iJIT_NotifyEvent` / MKL incompatibility.
  This is recorded as a package-distribution deviation, not a model change.

## Gate result: BLOCKED — do not run pristine reproduction yet

The official `requirements.txt` currently contains 21 `file://` dependencies,
including private author-machine paths for `kaolin`, `xformers`, `nvdiffrast`,
`diffoctreerast`, and `diff-gaussian-rasterization`.  Those paths are absent on
the lab server, so `pip install -r requirements.txt` is not a reproducible
official installation here.  The server also has no system CUDA toolkit; the
README warns that CUDA-dependent extensions require CUDA 12.1 compatibility.

The source `ckpt/` directory contains only a 1-byte `ckpt/1` placeholder; the
README-required `ckpt/image-large/`, `ckpt/text-xlarge/`, and `ckpt/clip/`
assets are absent.  No checkpoint was downloaded because a portable, pinned
dependency recipe must be settled first.

Proceed only when all of the following are available and fingerprinted:

1. An author-provided or otherwise reviewable portable recipe for each
   non-public local dependency and extension, compatible with Python 3.10,
   PyTorch 2.4.0, CUDA 12.1, and the RTX 3080 Ti.
2. The official TRELLIS checkpoint assets, with source URLs, revisions, and
   SHA-256 fingerprints recorded before use.
3. A compatibility decision for CUDA-dependent extensions (verified compatible
   prebuilt CUDA 12.1 packages or an explicitly provisioned CUDA 12.1 toolkit).
4. A pristine reproduction equivalence check with all input/output artifacts,
   prompts, seeds, preprocessing, sampling configuration, and environment lock
   captured.

## Source-only causal-suitability precheck (not a runtime classification)

The locked public text path is:

```
partial PLY -> points -> voxelized ss + mask -> run_lascomp(mask, ss)
  -> sample_lascomp: decoder(x_0) -> insert partial occupancy -> re-encode
  -> sample_slat -> decoded mesh / GLB
```

At every denoising iteration in
`trellis/pipelines/samplers/flow_euler.py`, the public path sets
`pred_voxel[voxel_mask] = 1.0` before re-encoding; nonzero
`optimization_step` additionally applies a partial-voxel BCE optimization.
The public script writes `sparse_structure.ply`, `output_mesh.glb`, and sampled
output points only *after* this conditioned process.

Consequently, current public output artifacts cannot be asserted to be a
defensible independent `G_raw`.  A pre-insertion first-step candidate exists
only as an internal transient and has not been exposed, instrumented, or shown
equivalent in a pristine run.  No replacement `G_raw` is introduced by this
repository.  The required formal suitability classification is intentionally
deferred until the blocked pristine reproduction and read-only trace/equivalence
gate are complete.
