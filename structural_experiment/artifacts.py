"""Frozen-scene persistence and a non-invasive LaS-Comp adapter boundary."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import file_fingerprint, fingerprint, read_json, write_json
from .contracts import FrozenScene, Geometry, ModelProvenance, Observation, GeneratedHypothesis, Surface


SCENE_SCHEMA = "structural-experiment/frozen-scene/v1"


def write_frozen_scene(path: str | Path, scene: FrozenScene) -> None:
    payload = {"schema": SCENE_SCHEMA, "scene": scene.to_dict(), "scene_fingerprint": scene.fingerprint}
    write_json(path, payload)


def read_frozen_scene(path: str | Path) -> FrozenScene:
    payload = read_json(path)
    if payload.get("schema") != SCENE_SCHEMA:
        raise ValueError(f"Unsupported frozen scene schema: {payload.get('schema')!r}")
    scene = FrozenScene.from_dict(payload["scene"])
    if payload.get("scene_fingerprint") != scene.fingerprint:
        raise ValueError("Frozen scene fingerprint does not match its serialized contents.")
    return scene


def _copy_immutable(source: str | Path, destination: Path) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    copied_hash = file_fingerprint(destination)
    source_hash = file_fingerprint(source_path)
    if copied_hash != source_hash:
        raise RuntimeError("Raw artifact copy hash mismatch.")
    return {"path": destination.as_posix(), "sha256": copied_hash, "size_bytes": destination.stat().st_size}


def load_ascii_ply_geometry(
    path: str | Path, *, geometry_id: str, surface_id: str, region_id: str, tags: tuple[str, ...] = (), units: str = "unitless",
) -> Geometry:
    """Load a simple ASCII PLY point cloud or mesh without importing model code.

    Official LaS-Comp may emit binary PLY or GLB.  Those remain materialized as
    raw artifacts but need a format-specific adapter before geometry evaluation.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ply" or "format ascii" not in lines[:3]:
        raise ValueError("Only ASCII PLY is supported by this dependency-free adapter.")
    vertex_count = 0
    properties: list[str] = []
    header_end = None
    in_vertex = False
    for index, line in enumerate(lines[1:], start=1):
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count, in_vertex = int(parts[2]), True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            properties.append(parts[-1])
        elif line.strip() == "end_header":
            header_end = index
            break
    if header_end is None or vertex_count <= 0:
        raise ValueError("PLY lacks a valid vertex section.")
    required = {"x", "y", "z"}
    if not required.issubset(properties):
        raise ValueError("PLY vertices must include x, y, and z.")
    prop_index = {name: index for index, name in enumerate(properties)}
    rows = [line.split() for line in lines[header_end + 1:header_end + 1 + vertex_count]]
    vertices = tuple((float(row[prop_index["x"]]), float(row[prop_index["y"]]), float(row[prop_index["z"]])) for row in rows)
    normals = None
    if {"nx", "ny", "nz"}.issubset(properties):
        normals = tuple((float(row[prop_index["nx"]]), float(row[prop_index["ny"]]), float(row[prop_index["nz"]])) for row in rows)
    return Geometry(geometry_id, (Surface(surface_id, region_id, vertices, normals=normals, tags=tags),), units=units)


@dataclass(frozen=True)
class ArtifactEquivalence:
    equivalent: bool
    mode: str
    pristine_fingerprint: str
    instrumented_fingerprint: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"equivalent": self.equivalent, "mode": self.mode, "pristine_fingerprint": self.pristine_fingerprint,
                "instrumented_fingerprint": self.instrumented_fingerprint, "reason": self.reason}


def compare_serialized_outputs(pristine: str | Path, instrumented: str | Path) -> ArtifactEquivalence:
    """Strict byte-level output comparison for the instrumentation equivalence gate."""
    first, second = file_fingerprint(pristine), file_fingerprint(instrumented)
    return ArtifactEquivalence(first == second, "exact_sha256", first, second,
                               None if first == second else "Serialized outputs differ; no loose tolerance was applied.")


class LaSCompArtifactExporter:
    """Adapter whose only dependency on LaS-Comp is its exported files.

    A caller that has completed the pristine and instrumentation-equivalence
    gates supplies already-extracted O and G.  This class never imports or
    patches upstream code, and it requires the caller to state stage semantics.
    """

    def freeze(
        self, destination: str | Path, scene: FrozenScene, *, raw_artifacts: Mapping[str, str | Path] | None = None,
    ) -> Path:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        materialized: dict[str, Any] = {}
        for role, source in (raw_artifacts or {}).items():
            source_path = Path(source)
            materialized[role] = _copy_immutable(source_path, root / "raw" / source_path.name)
        write_frozen_scene(root / "scene.json", scene)
        write_json(root / "artifact_manifest.json", {
            "schema": "structural-experiment/artifact-package/v1", "scene_fingerprint": scene.fingerprint,
            "model_provenance_fingerprint": scene.model_provenance.fingerprint, "raw_artifacts": materialized,
            "note": "This package is immutable by convention; hash verification is required before use.",
        })
        return root

    def freeze_ascii_ply_stages(
        self, destination: str | Path, *, scene_id: str, observation_ply: str | Path, generated_ply: str | Path,
        observation_surface_id: str, generated_surface_id: str, provenance: ModelProvenance,
        observation_preprocessing: Mapping[str, Any], observation_sampling: Mapping[str, Any],
        generated_stage_semantics: str,
    ) -> FrozenScene:
        """Freeze explicitly chosen LaS-Comp stage files; never labels one G_raw implicitly."""
        observation_geometry = load_ascii_ply_geometry(observation_ply, geometry_id=f"{scene_id}:O",
            surface_id=observation_surface_id, region_id="observed", tags=("full", "observed", "interface"))
        generated_geometry = load_ascii_ply_geometry(generated_ply, geometry_id=f"{scene_id}:G",
            surface_id=generated_surface_id, region_id="generated", tags=("full", "hidden", "interface"), units=observation_geometry.units)
        observation = Observation(f"{scene_id}:O", observation_geometry, observation_preprocessing, observation_sampling,
                                  {"source_path": str(observation_ply), "sha256": file_fingerprint(observation_ply)})
        hypothesis = GeneratedHypothesis(f"{scene_id}:G", generated_geometry,
            {"stage_semantics": generated_stage_semantics, "materialized": True, "seed": provenance.seed},
            {"source_path": str(generated_ply), "sha256": file_fingerprint(generated_ply)})
        # Evaluation requires an independently supplied ground truth/reference, so no fake reference is created here.
        raise NotImplementedError(
            "LaS-Comp stage loading is available, but a frozen scene requires an explicit independent EvaluationReference. "
            "Construct it from official benchmark metadata rather than deriving it from an association."
        )
