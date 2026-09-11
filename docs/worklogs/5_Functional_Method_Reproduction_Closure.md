# 5 Functional Method Reproduction Closure

**Date:** 2026-09-11

## Scope and preserved historical status

This batch opened the explicitly non-pristine `FUNCTIONAL_REPRODUCTION_CANDIDATE` gate. It did not alter the frozen LaS-Comp checkout, its provenance records, model source, checkpoints, prompt, seed, sampler, or Experiment 1 code.

Worklog #4 remains unchanged. Its historical result is still **`BLOCKED_PRE_INFERENCE`**: author-published material cannot reconstruct the exact historical dependency environment because key dependency identities are absent. That limitation is distinct from the functional-method question addressed here.

## Functional candidate environment

The candidate was created on LabServer63 in `/home/nd/.local/conda-envs/lascomp-functional-cu121` with the already validated RTX 3080 Ti (12 GiB), Python 3.10.21, `torch==2.4.0+cu121`, and `torchvision==0.19.0+cu121` core runtime.

- The compiler infrastructure is CUDA 12.1 (`nvcc 12.1.105`) with Conda GCC/G++ 11.4.0.
- PyTorch3D was built from the author-pinned commit `75ebeeaea0908c5527e7b1e305fbc7681382db47`; the produced wheel is `pytorch3d-0.7.8` with SHA-256 `1e02adbdd9ef09b66e553f266fb59a38d70f88b157a05f7e21cec0229634f15d`.
- The minimal CUDA 12.1 development libraries needed for that author-pinned build were added only after concrete compiler errors exposed missing headers: cuSPARSE 12.1.0.106, cuBLAS 12.1.3.1, and cuSOLVER 11.4.5.107, each with its matching development package.
- Exact author-listed xformers and Kaolin artifacts were used. `utils3d` used its author-pinned Git commit. The first runtime imports further established a real requirement for `pygltflib==1.16.2`, `warp-lang==1.8.1`, and `onnxruntime==1.22.1`; each is author-pinned in requirements.

These compiler and publicly resolvable package selections are **functional-environment deviations**, not claims about the authors' historical machine. Public resolver-selected transitive packages without an author artifact identity are retained in the server install reports. The complete candidate manifest is [FUNCTIONAL_REPRODUCTION_CANDIDATE_20260911.json](../../reference_models/lascomp/FUNCTIONAL_REPRODUCTION_CANDIDATE_20260911.json).

## Actual inference dependency closure

Starting from `run_lascomp_text_condition_single.py`, the following focused checks passed:

- The official entry point imported through argparse with the source-fixed `ATTN_BACKEND=xformers`, activating spconv and xformers.
- Kaolin, PyTorch3D, spconv, rembg/onnxruntime, Open3D, trimesh, PLY handling, and the TRELLIS pipeline imported without a source modification.
- The author-pinned PyTorch3D native extension built for `sm_86` and loaded along the import path.
- A separate load-only probe instantiated the official local pipeline and completed `pipeline.cuda()` successfully. It did not run sampling or inference.

`ipyevents` remains absent. Kaolin reports this only as a warning from its notebook visualization submodule; it does not prevent the official inference entry point import or checkpoint load. It is classified as tooling/visualization-only, alongside `ipycanvas`, Jupyter, USD, and related visualization packages.

`flash_attn` is unused because the source explicitly selects xformers. The three historically unpinned CUDA renderer extensions (`nvdiffrast`, `diffoctreerast`, and MIP-splatting `diff-gaussian-rasterization`) were not imported or loaded before the first source/configuration blocker. No functional substitute revision was selected and none was built.

## Checkpoints and official-format input

All seven provenance-identified files were acquired in server-only functional-reproduction paths and verified by SHA-256 against the existing checkpoint provenance manifest:

- Both `microsoft/TRELLIS-text-xlarge` flow weights at `e0b00432b8e3a8ecee0df806ab1df9f7281f2be4`.
- Four `JeffreyXiang/TRELLIS-image-large` decoders at `25e0d31ffbebe4b5a97464dd851910efc3002d96`.
- `openai/clip-vit-large-patch14` `model.safetensors` at `32bd64288804d66eefd0ccbe215aa642df71cc41`.

Every computed hash equals the source-declared hash. The literal `JeffreyXiang/TRELLIS-image-large` path in the official `pipeline.json` was preserved in the local checkpoint tree; `pipeline.json` was not edited.

The official Omni-Comp3D revision `9943b8b62106f0781b0b6809e30a00439a20672a` supplied `ycb_preprocessed/003` partial input and GT PLY. The author benchmark mapping names its prompt `A cracker box`.

## Single minimal author-faithful run

One actual invocation used:

- The official `run_lascomp_text_condition_single.py` entry point.
- The official YCB `003` partial PLY with `--dataset ycb_preprocessed`.
- Prompt `A cracker box`.
- The verified local text checkpoint root.
- All source defaults: seed 1; sparse denoise 100 steps; SLAT 25 steps; CFG 1.0 and interval `[0.5, 1.0]`; `rescale_t=3.0`; `alpha_eta=0.1`; `optimization_step=1`; `lr_sample=1e-5`; and 16,384 mesh points.

The first transport attempt stopped at argparse before any model activity because the remote shell split the prompt. The corrected invocation was the single actual inference attempt. It created only the source preprocessing artifacts `input_sample.ply` and `prompt.txt`; it did not materialize a sparse structure, mesh, GLB, or sampled output points. It was not repeated.

## First non-permissible semantic blocker

The fixed code and fixed official pipeline configuration contain a deterministic incompatibility on the attempted LaS-Comp path:

1. `TrellisTextTo3DPipeline.from_pretrained` builds `self.models` exclusively from the `models` map in the official `pipeline.json`.
2. That map includes `sparse_structure_decoder`, `sparse_structure_flow_model`, the three SLAT decoders, and `slat_flow_model`, but **does not include `sparse_structure_encoder`**.
3. The official `run_lascomp` path calls `lascomp_sample_sparse_structure`, which immediately performs `encoder = self.models['sparse_structure_encoder']`.

The successful load-only probe proves that this is not a checkpoint acquisition, CUDA toolkit, Python import, or ABI blocker. Supplying an encoder/checkpoint or changing the source access would alter the released configuration or Method behavior. Those actions are outside this functional reproduction gate.

## Fidelity assessment and decision

| Dimension | Result |
| --- | --- |
| Source fidelity | Pass: immutable official source commit used without modification. |
| Checkpoint fidelity | Pass: all seven official weights matched their declared SHA-256. |
| Configuration fidelity | Pass through the actual invocation: official input/prompt and source defaults retained. |
| Runtime deviations | Documented public-environment substitutions and CUDA build infrastructure; none presented as historical provenance. |
| Output-behavior fidelity | Fail: no inference output was materialized because the official inference path accesses an absent configuration key. |

**Functional reproduction classification: `FUNCTIONAL_REPRODUCTION_FAILED`.**

The evidence answers the completion question: incomplete historical provenance prevents pristine reproduction, and the public source/checkpoint configuration also currently prevents a scientifically defensible functional LaS-Comp inference without altering Method semantics. This is not an Experiment 1 result and not a suitability verdict.

## Boundaries and next gate

Runtime O-influence tracing, candidate-G inventory, G-suitability adjudication, Structural Attachment work, graph matching, NURBS work, and Experiment 1 remain **not permitted**. No dependency sweep, prompt tuning, seed search, sampling change, or model patch was performed.

**NEXT GATE: BLOCKED — an author-official source must reconcile the missing `sparse_structure_encoder` configuration/checkpoint with the released `run_lascomp` path before a method-faithful inference can be attempted.**
