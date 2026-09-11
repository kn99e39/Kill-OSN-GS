# LaS-Comp pristine-reproduction closure — 2026-09-11

## Decision

**PRISTINE_REPRODUCTION: BLOCKED_PRE_INFERENCE**

Faithful pristine reproduction cannot yet be established from the available
author-official public resources on `LabServer63`.  This is not a claim that
LaS-Comp is intrinsically unreproducible: two required binary packages and all
required model weights have authoritative public sources.  It is a narrow
reproducibility finding: the published environment does not identify enough
immutable dependency artifacts to construct the requested faithful reference
environment without making unrecorded choices.

No LaS-Comp checkpoint was downloaded or loaded; no author script, inference,
runtime trace, instrumentation, candidate-G selection, or Experiment 1 run was
performed in this closure.

## Evidence reviewed

1. The locked official source is
   `DavidYan2001/LaS-Comp@0da46f2569b484986d3737d97213428db962161f`.
   Its `requirements.txt` has 21 local `file://` records.
2. The complete reachable Git history for `requirements.txt` and
   `environment.yml` has one published version (introduced in
   `222a06b7ec6858b94f9af5ab50ee2ddbf457c344`); no earlier public lockfile,
   tag, or branch resolves the missing versions.
3. The official LaS-Comp repository has no releases or issues that add a
   portable dependency lock.  Its README only pins Python 3.10, CUDA 12.1,
   PyTorch 2.4.0, and torchvision 0.19.0.
4. The author-referenced Microsoft TRELLIS installer resolves the *project
   origins* of the three private CUDA build directories and provides exact
   PyTorch/CUDA 12.1 xFormers guidance.  It still clones the three CUDA
   extensions at an unpinned moving upstream revision.
5. `DEPENDENCY_RESOLUTION_20260910.json` records all 21 entries: two exact
   public equivalents, three unpinned official upstream builds, and sixteen
   entries with no recoverable exact package version.
6. `CHECKPOINT_PROVENANCE_20260910.json` identifies exact authoritative model
   repositories, revisions, files, sizes, and source-declared hashes.  Its
   `downloaded_sha256` fields deliberately remain null until the environment
   gate passes.

## What is already compatible

The isolated lab-server core runtime has been verified as Python 3.10,
`torch==2.4.0+cu121`, and `torchvision==0.19.0+cu121` on an RTX 3080 Ti.  The
wheel distribution rather than the failed Conda distribution is an
environment-source deviation only; it does not change model code, model
weights, arguments, sampling, or conditioning semantics.

Two private-wheel records have exact public candidates that must still be
downloaded and hash-verified before installation:

- `xformers==0.0.27.post2` from the official PyTorch CUDA 12.1 index;
- `kaolin==0.18.0` from NVIDIA's official Torch 2.4.0/CUDA 12.1 index.

## Blocking conditions

1. Sixteen `file://` entries name only local Conda/build workspaces; their
   exact versions are absent from every author-official artifact inspected.
   Selecting current public releases would violate the no-guessing rule.
2. `nvdiffrast`, `diffoctreerast`, and
   `diff_gaussian_rasterization` have author-referenced upstream projects, but
   no immutable commit, release, wheel hash, or build recipe specific to the
   LaS-Comp environment.  Building current upstream heads would not be a
   faithful reconstruction.
3. The source explicitly warns that CUDA-dependent extensions need CUDA 12.1.
   The server has a compatible PyTorch runtime but no system CUDA toolkit for
   the unpinned source builds.  Provisioning a toolkit is not useful until the
   sources themselves are immutably identified.

## Required authoritative evidence to reopen the gate

Obtain one of the following from the LaS-Comp authors or an explicitly
author-published/referenced immutable artifact:

1. A complete environment export containing versions and build strings for the
   sixteen unresolved packages (for example, Conda `--explicit` output plus
   pip direct-URL metadata); and
2. exact commit IDs, release tags, or original wheels plus SHA-256 for
   `nvdiffrast`, `diffoctreerast`, and the Mip-Splatting
   `diff-gaussian-rasterization` submodule.

Only then may the portable environment be assembled.  At that point the
already-recorded checkpoint repositories can be downloaded at their pinned
revisions, verified, and used for one pristine author-script invocation.  A
runtime trace, G suitability decision, read-only instrumentation, and any
real-scene freeze remain downstream of that successful pristine reproduction.
