# LaS-Comp reference and Structural-Association experimental substrate

Current readiness: **PARTIALLY READY**. The official LaS-Comp source is frozen, and the controlled frozen-artifact harness is implemented and tested with deterministic smoke fixtures. The official model is **NOT REPRODUCED** on this host, so no real LaS-Comp O/G artifact, instrumentation result, or Structural Association experiment has been produced.

## Reference-model status

The author-official source is isolated at `reference_models/lascomp/upstream` and pinned in [REFERENCE_LOCK.json](reference_models/lascomp/REFERENCE_LOCK.json). It is the clean commit `0da46f2569b484986d3737d97213428db962161f`; no upstream files were changed. The checkout has no submodules and no `LICENSE` file at that revision. The upstream `ckpt/` and `samples/` paths each only contain a two-byte placeholder, so checkpoints and official test material have not been acquired.

[ENVIRONMENT_ASSESSMENT.json](reference_models/lascomp/ENVIRONMENT_ASSESSMENT.json) records the pre-installation gate. The machine has a Windows 11 host, RTX 5080 with 16,303 MiB VRAM, CUDA driver support 13.2 and NVCC 13.3. The official project specifies Python 3.10, CUDA 12.1 and PyTorch 2.4.0/cu121, and explicitly warns about later CUDA versions. This host has Python 3.14.3 and no Conda or Python 3.10 on PATH. Additionally, the supplied requirements contain Linux-local package paths. Creating a different Windows environment would be a material, unvalidated deviation, so installation, checkpoint download, inference, and instrumentation are deliberately blocked.

## Pipeline mapping (source inspection only)

The text single-input entry point reads a partial PLY, conditionally performs `(x,y,z) -> (x,-z,y)`, writes `input_sample.ply`, and voxelizes the input into a 64-cube. It downsamples the observed occupancy mask and calls `TrellisTextTo3DPipeline.run_lascomp`.

Inside the official sampler, each sparse-structure denoising step decodes a predicted occupancy, writes the observed voxels into it, encodes that modified occupancy, and (with the documented default) optimizes the sample against the partial-voxel BCE loss. The returned sparse coordinates therefore already reflect interaction with O. The following structured-latent sampler decodes the final mesh. The public execution path does **not** expose a separate, unambiguous pre-interaction mesh-like `G_raw`; treating the final mesh as `G_raw` would be false. This finding must be resolved by a faithful, equivalence-checked extraction design before a real experiment begins.

## Experimental harness

The isolated `structural_experiment/` package has no import dependency on LaS-Comp. Its boundaries are:

| Module | Ownership |
| --- | --- |
| `contracts.py` | Immutable O, G, A, R, S*, ground truth/reference, stable IDs and coordinate convention |
| `artifacts.py`, `lascomp_adapter.py` | Frozen scene serialization, raw-file hashing, and external stage-file adapter |
| `reconcile.py` | Reference overlay only; it records injected A and never infers or edits it |
| `evaluation.py` | Per-region metric accounting and explicit unavailable metrics |
| `runner.py` | Deterministic run manifests, review exports, replay, and control-invariant verification |
| `fixtures.py` | Minimal continuation and unmatched smoke fixtures |

Surface identifiers belong to the frozen scene, are unique across O and G, and are retained in all artifacts. Coordinates are right-handed Cartesian `(x,y,z)`; units are declared per geometry and never converted. Normals, if present, are outward-facing under the right-hand convention. The experiment requires callers to supply A; an association entry is either `associated`, `unmatched_observation`, or `unmatched_generated`.

The smoke reconciliation is intentionally only a non-editing overlay. Its explicit fallback is `retain_both`; it is not LaS-Comp refinement and is not a Structural Attachment method.

Each frozen scene fingerprints O, G, preprocessing, sampling, evaluation reference, model provenance, and R configuration with canonical JSON SHA-256. A is separately fingerprinted. Each run writes a machine-readable manifest and a `review.json` containing O, G, A, S*, evaluation regions, and ownership of all stable surface IDs. The reusable metrics are symmetric nearest-point distance, normal discrepancy, association correctness, and unmatched correctness when an oracle is available. Seam and false-bridge metrics are reported as unavailable until a validated seam/topology representation exists; no aggregate score is made.

For real artifacts, `LaSCompStageAdapter` requires caller-designated exported stage files, independent evaluation reference, complete model provenance, and an explicit `generated_stage_semantics` label. It supports simple ASCII PLY without model dependencies, materializes raw source files, and records their hashes. Binary PLY and GLB require a format-specific reader before geometry evaluation; their raw files can still be preserved.

## Verification

Run focused contract tests:

```powershell
py -m unittest discover -s tests -v
```

Run the two-A dry run (not a quality comparison):

```powershell
py examples/run_smoke.py --output artifacts/smoke
```

The dry run uses exactly the same synthetic O, G, preprocessing, sampling, R, evaluation reference, regions, and model provenance for two different external A inputs. It automatically asserts that every controlled fingerprint is identical and produces deterministic replayable JSON manifests.

## Required next gate

Use a validated Linux/CUDA-12.1/Python-3.10 isolated environment (or obtain a documented official Windows-compatible environment) before proceeding. Then:

1. Acquire only project-linked TRELLIS/CLIP checkpoints and official samples.
2. Execute and preserve an unmodified official inference and bounded official evaluation.
3. Compare against an appropriate supplied/published reference and investigate material discrepancies.
4. Add only read-only stage exports and pass the exact/numerical instrumentation-equivalence gate.
5. Freeze independently meaningful O/G states and their coordinate transforms.

Until those gates pass: reference model **NOT REPRODUCED**; instrumentation **NOT EQUIVALENT (not introduced or tested)**; real-model experimental substrate **NOT READY**. The synthetic control substrate is ready for its limited implementation-validation role only.
