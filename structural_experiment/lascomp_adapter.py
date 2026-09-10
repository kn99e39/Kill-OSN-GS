"""Explicit, non-invasive conversion from validated LaS-Comp stage files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .artifacts import LaSCompArtifactExporter, load_ascii_ply_geometry
from .canonical import file_fingerprint
from .contracts import EvaluationReference, FrozenScene, GeneratedHypothesis, ModelProvenance, Observation


class LaSCompStageAdapter:
    """Freeze caller-designated O/G files without importing LaS-Comp internals.

    ``generated_stage_semantics`` is mandatory because the current official
    public pipeline does not expose a generic, unambiguous G_raw mesh state.
    """

    def freeze_ascii_ply_scene(
        self, destination: str | Path, *, scene_id: str, observation_ply: str | Path, generated_ply: str | Path,
        observation_surface_id: str, generated_surface_id: str, evaluation_reference: EvaluationReference,
        provenance: ModelProvenance, preprocessing: Mapping[str, Any], sampling: Mapping[str, Any],
        generated_stage_semantics: str, reference_result_path: str | Path | None = None,
    ) -> FrozenScene:
        if not generated_stage_semantics.strip():
            raise ValueError("generated_stage_semantics must state exactly what the exported LaS-Comp state represents.")
        observation_geometry = load_ascii_ply_geometry(
            observation_ply, geometry_id=f"{scene_id}:observation", surface_id=observation_surface_id,
            region_id="observed", tags=("full", "observed", "interface"),
        )
        generated_geometry = load_ascii_ply_geometry(
            generated_ply, geometry_id=f"{scene_id}:generated", surface_id=generated_surface_id,
            region_id="generated", tags=("full", "hidden", "interface"), units=observation_geometry.units,
        )
        observation = Observation(f"{scene_id}:O", observation_geometry, preprocessing, sampling,
            {"source_path": str(observation_ply), "sha256": file_fingerprint(observation_ply)})
        hypothesis = GeneratedHypothesis(f"{scene_id}:G", generated_geometry,
            {"stage_semantics": generated_stage_semantics, "materialized": True, "seed": provenance.seed},
            {"source_path": str(generated_ply), "sha256": file_fingerprint(generated_ply)})
        reference_artifact = {} if reference_result_path is None else {
            "source_path": str(reference_result_path), "sha256": file_fingerprint(reference_result_path)
        }
        scene = FrozenScene(scene_id, observation, hypothesis, evaluation_reference, provenance, reference_artifact)
        raw = {"observation": observation_ply, "generated": generated_ply}
        if reference_result_path is not None:
            raw["reference_result"] = reference_result_path
        LaSCompArtifactExporter().freeze(destination, scene, raw_artifacts=raw)
        return scene
