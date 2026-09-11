# 4 LaS-Comp Pristine Reproduction and G Suitability Gate

**Date:** 2026-09-11

## Objective and scope

Close the next Experiment 1 readiness gate by determining whether the author-official LaS-Comp release can be reproduced faithfully from immutable public dependencies and checkpoints. Runtime O-influence tracing and G suitability adjudication are in scope only after a pristine reproduction pass. Experiment 1 itself is out of scope.

## Completed work

- Re-read the complete dependency-resolution, checkpoint-provenance, and previous reproduction-closure records against the frozen LaS-Comp source.
- Revalidated the author-official remote: `origin/HEAD` and `origin/main` both remain `0da46f2569b484986d3737d97213428db962161f`.
- Revalidated that the frozen upstream checkout is clean.
- Rechecked the complete reachable Git history for `environment.yml` and `requirements.txt`. It contains one published source revision, `222a06b7ec6858b94f9af5ab50ee2ddbf457c344`; there is no alternative public lockfile history to recover.
- Rechecked the author-referenced TRELLIS installer. It identifies the project origins of the three private CUDA extensions but clones each from an unpinned moving revision.
- Preserved the validated RTX 3080 Ti core runtime and all Experiment 1 contracts without altering their semantics.

## Observed evidence

### Dependency resolution

The frozen official requirements file has 21 non-portable `file://` records.

- Two are exact publicly hosted binary candidates with supplied version and SHA-256: Kaolin 0.18.0 and xformers 0.0.27.post2.
- Three identify authoritative upstream projects but no immutable source revision or build artifact: MIP-splatting's `diff-gaussian-rasterization`, `diffoctreerast`, and `nvdiffrast`.
- Sixteen provide a package name and a local Conda/build workspace path, but no version, build string, artifact filename, or artifact hash. The path timestamps are not package identities.

### Checkpoint provenance

The checkpoint record remains complete but intentionally unacquired: the two `microsoft/TRELLIS-text-xlarge` flow weights, four `JeffreyXiang/TRELLIS-image-large` shared decoders, and `openai/clip-vit-large-patch14` are identified by immutable repository revision, filename, size, and source-declared SHA-256. No checkpoint was downloaded or loaded before the environment gate passed.

### Portable environment

LabServer63 has a validated core runtime: RTX 3080 Ti, Python 3.10, `torch==2.4.0+cu121`, `torchvision==0.19.0+cu121`, and CUDA runtime 12.1. This does not supply the missing immutable identities or a CUDA toolkit for the unresolved source builds.

## Verification

- Frozen source remote and branch SHA equality: pass.
- Frozen upstream source worktree cleanliness: pass.
- Requirements/environment history has no additional published revision: pass.
- Dependency manifest structural check: 21 records comprising 2 exact public equivalents, 3 unpinned official upstream builds, and 16 unresolved identities.
- Checkpoint manifest structural check: 7 weight records with pinned revisions and source-declared hashes.

No dependency import, compiled CUDA-extension smoke test, checkpoint load, or pristine inference was run, because each is downstream of a portable, immutable environment.

## Inference and decision

The author-official material is insufficient to reconstruct the declared environment faithfully without selecting versions or source commits that the authors did not identify. Such selections would be a new, non-pristine environment rather than a validated reproduction.

**Pristine reproduction result: `BLOCKED_PRE_INFERENCE`.**

The runtime O-influence trace, candidate-G inventory, G-suitability analysis, and the three-way LaS-Comp suitability classification are therefore **deferred**, not negative findings. Issuing `UNSUITABLE_FOR_CURRENT_EXPERIMENT` without authentic runtime evidence would misstate the result; issuing either suitable classification would violate the prerequisite gate.

## Deviations

None in this batch. The previously documented official PyTorch CUDA-wheel distribution is retained as a core-runtime distribution deviation only; no model code, checkpoint, prompt, seed, conditioning, or sampling behavior was changed.

## Unresolved blockers

1. Immutable version/build identities for the sixteen local Conda/build workspace dependencies.
2. Exact commit, release tag, or original wheel plus SHA-256 for `nvdiffrast`, `diffoctreerast`, and MIP-splatting's `diff-gaussian-rasterization` submodule.
3. A CUDA 12.1 toolkit may be needed after those immutable source identities are supplied; provisioning it before then would not close the provenance gap.

## Preserved boundaries

No Experiment 1 run, baseline-versus-oracle comparison, Structural Attachment implementation, automatic segmentation, graph matching, NURBS work, model tuning, seed search, prompt alteration, instrumentation, arbitrary G-state naming, checkpoint acquisition, model import, or inference occurred.

## Outcome

The readiness gate is closed with a reproducibility block, not with experimental evidence or a suitability verdict. Author-published environment provenance is required before one pristine official run can be attempted. Relevant earlier commits are `3c27fe1a34557c63b2b273abd42fec392bb5207c`, `43414816beb0c4ae1f91a01f33a33e891c039884`, and `1b4aff0ec116b4c5cc3ac153ca3373db0ff33117`.

**NEXT GATE: BLOCKED — obtain the exact unresolved dependency artifacts and immutable CUDA-extension revisions from an author-official source.**
