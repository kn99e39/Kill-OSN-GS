"""Immutable contracts for the LaS-Comp Experiment 1 substrate.

This module deliberately models *provenance and selection*, not an attachment
solution.  The experiment may compare different association declarations only
after all geometry, region partitions, preprocessing, sampling and evaluation
choices have been frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


class ContractError(ValueError):
    """Raised when an Experiment 1 control is ill-defined or contradictory."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 over the semantic JSON representation of *value*."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash raw bytes without interpreting geometry or normalising line endings."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_nonempty(values: Iterable[int], label: str) -> tuple[int, ...]:
    result = tuple(sorted(set(values)))
    if not result:
        raise ContractError(f"{label} must not be empty")
    if any(not isinstance(index, int) or index < 0 for index in result):
        raise ContractError(f"{label} must contain non-negative integer indices")
    return result


@dataclass(frozen=True)
class RawArtifactLineage:
    """Identity of a raw surface before any partition, transform, or sampling."""

    artifact_id: str
    original_file_sha256: str
    representation: Literal["ply", "glb", "mesh", "point_cloud", "other"]
    vertex_count: int
    face_count: int
    source_locator: str = ""
    source_revision: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ContractError("raw artifact_id is required")
        if len(self.original_file_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.original_file_sha256.lower()
        ):
            raise ContractError("original_file_sha256 must be a lowercase/uppercase SHA-256")
        if self.vertex_count <= 0 or self.face_count < 0:
            raise ContractError("raw artifact counts are invalid")

    @property
    def identity_fingerprint(self) -> str:
        return fingerprint(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegionHandle:
    """Externally supplied ownership of raw vertices/faces for a named region.

    ``vertex_indices`` and ``face_indices`` always refer to the *raw* artifact
    described by ``RegionPartition.raw_artifact``.  They cannot be regenerated
    from, or inferred by, an association declaration.
    """

    region_id: str
    vertex_indices: tuple[int, ...]
    face_indices: tuple[int, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ContractError("region_id is required")
        object.__setattr__(self, "vertex_indices", _unique_nonempty(self.vertex_indices, "vertex_indices"))
        if self.face_indices:
            object.__setattr__(self, "face_indices", _unique_nonempty(self.face_indices, "face_indices"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "vertex_indices": list(self.vertex_indices),
            "face_indices": list(self.face_indices),
            "label": self.label,
        }


PartitionSource = Literal["manual_annotation", "benchmark_gt", "precomputed", "synthetic_gt"]


@dataclass(frozen=True)
class RegionPartition:
    """A frozen, independently created partition of one raw artifact."""

    partition_id: str
    raw_artifact: RawArtifactLineage
    source: PartitionSource
    regions: tuple[RegionHandle, ...]
    creation_protocol_id: str
    version: str = "1"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.partition_id or not self.creation_protocol_id:
            raise ContractError("partition_id and creation_protocol_id are required")
        if not self.regions:
            raise ContractError("a region partition requires at least one region")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ContractError("region IDs must be unique within a partition")
        claimed_vertices: set[int] = set()
        claimed_faces: set[int] = set()
        for region in self.regions:
            if any(index >= self.raw_artifact.vertex_count for index in region.vertex_indices):
                raise ContractError(f"region {region.region_id} claims a vertex outside raw artifact bounds")
            if any(index >= self.raw_artifact.face_count for index in region.face_indices):
                raise ContractError(f"region {region.region_id} claims a face outside raw artifact bounds")
            overlap_vertices = claimed_vertices.intersection(region.vertex_indices)
            overlap_faces = claimed_faces.intersection(region.face_indices)
            if overlap_vertices or overlap_faces:
                raise ContractError(
                    f"region ownership overlaps in partition {self.partition_id}; "
                    "declare a separate, explicitly overlapping annotation protocol instead"
                )
            claimed_vertices.update(region.vertex_indices)
            claimed_faces.update(region.face_indices)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "raw_artifact": self.raw_artifact.to_dict(),
            "source": self.source,
            "regions": [region.to_dict() for region in self.regions],
            "creation_protocol_id": self.creation_protocol_id,
            "version": self.version,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class EvaluationSelector:
    """A fixed selector for exactly one side of a result-vs-reference metric.

    The reference and result selectors deliberately have no shared-ID or
    one-to-one constraint.  This prevents the experiment harness from silently
    turning a geometric metric into an attachment solver.
    """

    selector_id: str
    artifact_id: str
    selection_kind: Literal["region_handle", "whole_artifact", "world_aabb", "explicit_indices"]
    region_id: str = ""
    vertex_indices: tuple[int, ...] = ()
    world_aabb: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.selector_id or not self.artifact_id:
            raise ContractError("selector_id and artifact_id are required")
        if self.selection_kind == "region_handle" and not self.region_id:
            raise ContractError("region_handle selector requires region_id")
        if self.selection_kind == "explicit_indices":
            object.__setattr__(self, "vertex_indices", _unique_nonempty(self.vertex_indices, "vertex_indices"))
        elif self.vertex_indices:
            raise ContractError("vertex_indices are allowed only for explicit_indices selectors")
        if self.selection_kind == "world_aabb":
            if self.world_aabb is None or len(self.world_aabb) != 6:
                raise ContractError("world_aabb selector requires six bounds")
            x0, y0, z0, x1, y1, z1 = self.world_aabb
            if not (x0 < x1 and y0 < y1 and z0 < z1):
                raise ContractError("world_aabb min bounds must be less than max bounds")
        elif self.world_aabb is not None:
            raise ContractError("world_aabb is allowed only for world_aabb selectors")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_id": self.selector_id,
            "artifact_id": self.artifact_id,
            "selection_kind": self.selection_kind,
            "region_id": self.region_id,
            "vertex_indices": list(self.vertex_indices),
            "world_aabb": list(self.world_aabb) if self.world_aabb is not None else None,
        }


@dataclass(frozen=True)
class FixedEvaluationDefinition:
    """Evaluation protocol frozen independently of the association A."""

    evaluation_id: str
    metric_id: str
    result: EvaluationSelector
    reference: EvaluationSelector
    protocol_revision: str
    preprocessing_fingerprint: str
    sampling_fingerprint: str

    def __post_init__(self) -> None:
        if not all((self.evaluation_id, self.metric_id, self.protocol_revision)):
            raise ContractError("evaluation_id, metric_id, and protocol_revision are required")
        for label, value in (
            ("preprocessing_fingerprint", self.preprocessing_fingerprint),
            ("sampling_fingerprint", self.sampling_fingerprint),
        ):
            if len(value) != 64:
                raise ContractError(f"{label} must be a SHA-256")

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "metric_id": self.metric_id,
            "result": self.result.to_dict(),
            "reference": self.reference.to_dict(),
            "protocol_revision": self.protocol_revision,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "sampling_fingerprint": self.sampling_fingerprint,
        }


@dataclass(frozen=True)
class AssociationDeclaration:
    """The sole permitted varying input to a controlled Experiment 1 run.

    A declaration is intentionally a labelled many-to-many relation.  It does
    not validate coverage, bijectivity, or semantic plausibility; such rules
    would be an attachment solver and must remain outside this substrate.
    """

    association_id: str
    links: tuple[tuple[str, str], ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.association_id:
            raise ContractError("association_id is required")
        normalized = tuple(sorted(set((str(left), str(right)) for left, right in self.links)))
        if any(not left or not right for left, right in normalized):
            raise ContractError("association links need two nonempty region IDs")
        object.__setattr__(self, "links", normalized)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"association_id": self.association_id, "links": [list(link) for link in self.links], "rationale": self.rationale}


@dataclass(frozen=True)
class Experiment1ControlPlane:
    """All frozen controls and their content-addressed fingerprints."""

    observed_partition: RegionPartition
    generated_partition: RegionPartition
    reference_partition: RegionPartition
    preprocessing_fingerprint: str
    sampling_fingerprint: str
    renderer_fingerprint: str
    evaluation: FixedEvaluationDefinition
    reference_model_fingerprint: str
    seed: int

    def __post_init__(self) -> None:
        for label, value in (
            ("preprocessing_fingerprint", self.preprocessing_fingerprint),
            ("sampling_fingerprint", self.sampling_fingerprint),
            ("renderer_fingerprint", self.renderer_fingerprint),
            ("reference_model_fingerprint", self.reference_model_fingerprint),
        ):
            if len(value) != 64:
                raise ContractError(f"{label} must be a SHA-256")
        if not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")

    def frozen_fingerprints(self) -> dict[str, str]:
        return {
            "O_raw": self.observed_partition.raw_artifact.identity_fingerprint,
            "G_raw": self.generated_partition.raw_artifact.identity_fingerprint,
            "R_raw": self.reference_partition.raw_artifact.identity_fingerprint,
            "O_partition": self.observed_partition.fingerprint,
            "G_partition": self.generated_partition.fingerprint,
            "R_partition": self.reference_partition.fingerprint,
            "preprocessing": self.preprocessing_fingerprint,
            "sampling": self.sampling_fingerprint,
            "renderer": self.renderer_fingerprint,
            "evaluation": self.evaluation.fingerprint,
            "reference_model": self.reference_model_fingerprint,
            "seed": fingerprint({"seed": self.seed}),
        }

    def manifest(self, association: AssociationDeclaration) -> dict[str, Any]:
        return {
            "schema": "experiment1-control-plane/v1",
            "frozen": self.frozen_fingerprints(),
            "association": association.to_dict(),
            "association_fingerprint": association.fingerprint,
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True)
class ControlInvariantReport:
    preserved: bool
    differences: Mapping[str, tuple[str | None, str | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"preserved": self.preserved, "differences": {key: list(value) for key, value in self.differences.items()}}


def verify_only_association_changed(
    baseline: Experiment1ControlPlane,
    candidate: Experiment1ControlPlane,
) -> ControlInvariantReport:
    """Reject every altered frozen control while allowing a different A."""
    before = baseline.frozen_fingerprints()
    after = candidate.frozen_fingerprints()
    keys = sorted(set(before).union(after))
    differences = {key: (before.get(key), after.get(key)) for key in keys if before.get(key) != after.get(key)}
    return ControlInvariantReport(preserved=not differences, differences=differences)
