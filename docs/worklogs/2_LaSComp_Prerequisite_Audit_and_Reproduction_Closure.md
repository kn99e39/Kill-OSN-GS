# 2 LaS-Comp Prerequisite Audit and Reproduction Closure

**Dates:** 2026-09-10 to 2026-09-11
**Scope:** Resolve the source, dependency, and checkpoint prerequisites required for a faithful LaS-Comp reproduction before any runtime tracing, model inference, suitability classification, or Experiment 1 execution.

## Objective

Determine whether the official LaS-Comp public release contains enough authoritative information to construct its declared environment and recover the required checkpoints faithfully on LabServer63.

## Evidence collected

- Audited all **21** `file://` requirement entries in the official lock-like file.
- Identified two exact public wheel candidates: Kaolin 0.18.0 and xformers 0.0.27.post2, each supported by the matching PyTorch CUDA 12.1 wheel index.
- Identified three source builds that must come from upstream repositories: `diff_gaussian_rasterization` (MIP-splatting), `diffoctreerast`, and `nvdiffrast`.
- Established that the remaining sixteen local Conda-build references have no versioned, portable public identity in the official requirements file.
- Checked the official repository history, branches, tags, releases, and issue tracker for a replacement lockfile or an author-provided revision record. None was available.
- Constructed a checkpoint provenance manifest from the official TRELLIS-linked runtime configuration, including immutable Hugging Face revisions, filenames, sizes, and SHA-256 values for the two text-flow checkpoints, four shared decoders, and the CLIP text-conditioning model.

## Verification

- The dependency-resolution and checkpoint-provenance manifests passed their schema and count validation: `dependency_entries=21; checkpoint_records=7; manifests_valid`.
- Static source inspection confirmed that the public text path creates a conditioned sparse structure and decoded mesh only after the conditioned flow. It did not yield a runtime-traced, formally classified raw structural artifact.

## Reproduction decision

**Status: `BLOCKED_PRE_INFERENCE`.**

The available public source establishes a compatible GPU runtime core, but it cannot faithfully recreate the authors' full declared environment: sixteen `file://` build artifacts are unversioned and unavailable, and three extension sources lack an author-pinned revision. Choosing substitute versions or arbitrary current source revisions would create a non-pristine environment. Under the project protocol, work must therefore stop before instrumentation, trace collection, checkpoint download, model import, inference, G-suitability classification, or Experiment 1 execution.

## What would close the gate

One of the following authoritative artifacts is required:

1. An author-exported portable lockfile or environment archive that identifies all local build artifacts.
2. Exact source revisions and build instructions for the three custom extensions, plus version identities for the sixteen unversioned packages.
3. An author-maintained released environment specification that supersedes the current `file://` entries.

## Boundaries preserved

No fallback environment was silently selected. No model weights were downloaded or used. No inference, trace, parameter sweep, optimization, segmentation, graph construction, NURBS processing, or experimental result occurred.

## Outcome and handoff

The prerequisite audit is complete and the reproduction gate is explicitly closed pending authoritative provenance. The experimental substrate and core GPU environment remain preserved for a future faithful retry.

**Related commits:** `3c27fe1a34557c63b2b273abd42fec392bb5207c`, `43414816beb0c4ae1f91a01f33a33e891c039884`
