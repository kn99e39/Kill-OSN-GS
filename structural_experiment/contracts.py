"""Model-independent, serializable contracts for controlled completion studies.

Coordinates are Cartesian right-handed ``(x, y, z)``.  Units are declared on
each geometry and never silently converted.  Surface IDs are owned by a frozen
scene and must be unique across its observation and hypothesis geometries.
Normals, when supplied, are outward-facing in the declared coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from .canonical import fingerprint, json_value


Vector3 = tuple[float, float, float]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _freeze({} if value is None else value)


def _vector(value: Any) -> Vector3:
    if len(value) != 3:
        raise ValueError("A coordinate or normal must have exactly three components.")
    return (float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class Surface:
    surface_id: str
    region_id: str
    vertices: tuple[Vector3, ...]
    faces: tuple[tuple[int, int, int], ...] = ()
    normals: Optional[tuple[Vector3, ...]] = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.surface_id or not self.region_id:
            raise ValueError("surface_id and region_id are required stable identifiers.")
        vertices = tuple(_vector(item) for item in self.vertices)
        if not vertices:
            raise ValueError(f"Surface {self.surface_id!r} has no vertices.")
        faces = tuple(tuple(int(index) for index in face) for face in self.faces)
        if any(len(face) != 3 or min(face) < 0 or max(face) >= len(vertices) for face in faces):
            raise ValueError(f"Surface {self.surface_id!r} contains an invalid triangle.")
        normals = None if self.normals is None else tuple(_vector(item) for item in self.normals)
        if normals is not None and len(normals) != len(vertices):
            raise ValueError("Normals must be one-per-vertex when present.")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "region_id": self.region_id,
            "vertices": [list(point) for point in self.vertices],
            "faces": [list(face) for face in self.faces],
            "normals": None if self.normals is None else [list(normal) for normal in self.normals],
            "tags": list(self.tags),
            "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Surface":
        return cls(
            surface_id=data["surface_id"], region_id=data["region_id"], vertices=tuple(data["vertices"]),
            faces=tuple(data.get("faces", ())), normals=None if data.get("normals") is None else tuple(data["normals"]),
            tags=tuple(data.get("tags", ())), metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    surfaces: tuple[Surface, ...]
    units: str = "unitless"
    coordinate_system: str = "right_handed_xyz"
    normal_orientation: str = "outward_right_hand_rule"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.geometry_id or not self.units:
            raise ValueError("geometry_id and units are required.")
        surfaces = tuple(self.surfaces)
        ids = [surface.surface_id for surface in surfaces]
        if len(ids) != len(set(ids)):
            raise ValueError("Surface IDs must be unique inside a geometry.")
        if self.coordinate_system != "right_handed_xyz":
            raise ValueError("This initial contract supports only right_handed_xyz coordinates.")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def surface_ids(self) -> tuple[str, ...]:
        return tuple(surface.surface_id for surface in self.surfaces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_id": self.geometry_id, "surfaces": [surface.to_dict() for surface in self.surfaces],
            "units": self.units, "coordinate_system": self.coordinate_system,
            "normal_orientation": self.normal_orientation, "metadata": json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Geometry":
        return cls(
            geometry_id=data["geometry_id"], surfaces=tuple(Surface.from_dict(item) for item in data["surfaces"]),
            units=data.get("units", "unitless"), coordinate_system=data.get("coordinate_system", "right_handed_xyz"),
            normal_orientation=data.get("normal_orientation", "outward_right_hand_rule"), metadata=data.get("metadata", {}),
        )

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class Observation:
    observation_id: str
    geometry: Geometry
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    sampling: Mapping[str, Any] = field(default_factory=dict)
    source_artifact: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "preprocessing", _mapping(self.preprocessing))
        object.__setattr__(self, "sampling", _mapping(self.sampling))
        object.__setattr__(self, "source_artifact", _mapping(self.source_artifact))

    def to_dict(self) -> dict[str, Any]:
        return {"observation_id": self.observation_id, "geometry": self.geometry.to_dict(),
                "preprocessing": json_value(self.preprocessing), "sampling": json_value(self.sampling),
                "source_artifact": json_value(self.source_artifact)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        return cls(data["observation_id"], Geometry.from_dict(data["geometry"]), data.get("preprocessing", {}),
                   data.get("sampling", {}), data.get("source_artifact", {}))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class GeneratedHypothesis:
    hypothesis_id: str
    geometry: Geometry
    generation: Mapping[str, Any] = field(default_factory=dict)
    source_artifact: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", _mapping(self.generation))
        object.__setattr__(self, "source_artifact", _mapping(self.source_artifact))

    def to_dict(self) -> dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "geometry": self.geometry.to_dict(),
                "generation": json_value(self.generation), "source_artifact": json_value(self.source_artifact)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratedHypothesis":
        return cls(data["hypothesis_id"], Geometry.from_dict(data["geometry"]), data.get("generation", {}),
                   data.get("source_artifact", {}))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class AssociationEntry:
    """An injected association; no inference or topology is implied."""
    state: str
    observation_surface_id: Optional[str]
    generated_surface_id: Optional[str]

    def __post_init__(self) -> None:
        allowed = {"associated", "unmatched_observation", "unmatched_generated"}
        if self.state not in allowed:
            raise ValueError(f"Unknown association state {self.state!r}.")
        if self.state == "associated" and (not self.observation_surface_id or not self.generated_surface_id):
            raise ValueError("An associated entry requires both stable surface IDs.")
        if self.state == "unmatched_observation" and (not self.observation_surface_id or self.generated_surface_id):
            raise ValueError("unmatched_observation requires only an observation surface ID.")
        if self.state == "unmatched_generated" and (not self.generated_surface_id or self.observation_surface_id):
            raise ValueError("unmatched_generated requires only a generated surface ID.")

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "observation_surface_id": self.observation_surface_id,
                "generated_surface_id": self.generated_surface_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssociationEntry":
        return cls(data["state"], data.get("observation_surface_id"), data.get("generated_surface_id"))


@dataclass(frozen=True)
class Association:
    association_id: str
    entries: tuple[AssociationEntry, ...]
    provider: str = "external"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not self.association_id:
            raise ValueError("association_id is required.")
        if len({tuple(entry.to_dict().items()) for entry in entries}) != len(entries):
            raise ValueError("Association contains duplicate entries.")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"association_id": self.association_id, "entries": [entry.to_dict() for entry in self.entries],
                "provider": self.provider, "metadata": json_value(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Association":
        return cls(data["association_id"], tuple(AssociationEntry.from_dict(item) for item in data["entries"]),
                   data.get("provider", "external"), data.get("metadata", {}))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class ReconciliationConfiguration:
    operator_id: str = "reference-overlay-v1"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    unmatched_policy: str = "retain_both"

    def __post_init__(self) -> None:
        if self.unmatched_policy != "retain_both":
            raise ValueError("The smoke-test reconciler only supports explicit retain_both fallback.")
        object.__setattr__(self, "parameters", _mapping(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {"operator_id": self.operator_id, "parameters": json_value(self.parameters),
                "unmatched_policy": self.unmatched_policy}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReconciliationConfiguration":
        return cls(data.get("operator_id", "reference-overlay-v1"), data.get("parameters", {}),
                   data.get("unmatched_policy", "retain_both"))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class EvaluationRegion:
    region_id: str
    surface_ids: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.region_id or not self.surface_ids:
            raise ValueError("An evaluation region needs an ID and at least one stable surface ID.")
        object.__setattr__(self, "surface_ids", tuple(self.surface_ids))

    def to_dict(self) -> dict[str, Any]:
        return {"region_id": self.region_id, "surface_ids": list(self.surface_ids), "description": self.description}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationRegion":
        return cls(data["region_id"], tuple(data["surface_ids"]), data["description"])


@dataclass(frozen=True)
class EvaluationReference:
    reference_id: str
    geometry: Geometry
    regions: tuple[EvaluationRegion, ...]
    oracle_association: Optional[Association] = None

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        known = set(self.geometry.surface_ids)
        if any(not set(region.surface_ids).issubset(known) for region in regions):
            raise ValueError("Evaluation regions can only select reference-owned stable surface IDs.")
        object.__setattr__(self, "regions", regions)

    def to_dict(self) -> dict[str, Any]:
        return {"reference_id": self.reference_id, "geometry": self.geometry.to_dict(),
                "regions": [region.to_dict() for region in self.regions],
                "oracle_association": None if self.oracle_association is None else self.oracle_association.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationReference":
        oracle = data.get("oracle_association")
        return cls(data["reference_id"], Geometry.from_dict(data["geometry"]),
                   tuple(EvaluationRegion.from_dict(item) for item in data["regions"]),
                   None if oracle is None else Association.from_dict(oracle))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class ModelProvenance:
    model_id: str
    repository_url: str
    commit_sha: str
    sample_id: str
    seed: Optional[int]
    checkpoints: tuple[Mapping[str, Any], ...] = ()
    coordinate_transform: Mapping[str, Any] = field(default_factory=dict)
    extraction: Mapping[str, Any] = field(default_factory=dict)
    instrumentation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints", tuple(_mapping(item) for item in self.checkpoints))
        object.__setattr__(self, "coordinate_transform", _mapping(self.coordinate_transform))
        object.__setattr__(self, "extraction", _mapping(self.extraction))
        object.__setattr__(self, "instrumentation", _mapping(self.instrumentation))

    def to_dict(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "repository_url": self.repository_url, "commit_sha": self.commit_sha,
                "sample_id": self.sample_id, "seed": self.seed, "checkpoints": json_value(self.checkpoints),
                "coordinate_transform": json_value(self.coordinate_transform), "extraction": json_value(self.extraction),
                "instrumentation": json_value(self.instrumentation)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelProvenance":
        return cls(data["model_id"], data["repository_url"], data["commit_sha"], data["sample_id"], data.get("seed"),
                   tuple(data.get("checkpoints", ())), data.get("coordinate_transform", {}),
                   data.get("extraction", {}), data.get("instrumentation", {}))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class FrozenScene:
    scene_id: str
    observation: Observation
    hypothesis: GeneratedHypothesis
    evaluation_reference: EvaluationReference
    model_provenance: ModelProvenance
    reference_result_artifact: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation_ids = set(self.observation.geometry.surface_ids)
        generated_ids = set(self.hypothesis.geometry.surface_ids)
        if observation_ids & generated_ids:
            raise ValueError("Surface IDs are scene-owned: O and G must not reuse an ID.")
        object.__setattr__(self, "reference_result_artifact", _mapping(self.reference_result_artifact))

    def to_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "observation": self.observation.to_dict(), "hypothesis": self.hypothesis.to_dict(),
                "evaluation_reference": self.evaluation_reference.to_dict(), "model_provenance": self.model_provenance.to_dict(),
                "reference_result_artifact": json_value(self.reference_result_artifact)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrozenScene":
        return cls(data["scene_id"], Observation.from_dict(data["observation"]),
                   GeneratedHypothesis.from_dict(data["hypothesis"]), EvaluationReference.from_dict(data["evaluation_reference"]),
                   ModelProvenance.from_dict(data["model_provenance"]), data.get("reference_result_artifact", {}))

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class ReconciledResult:
    result_id: str
    geometry: Geometry
    association: Association
    reconciliation_configuration: ReconciliationConfiguration
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", _mapping(self.notes))

    def to_dict(self) -> dict[str, Any]:
        return {"result_id": self.result_id, "geometry": self.geometry.to_dict(), "association": self.association.to_dict(),
                "reconciliation_configuration": self.reconciliation_configuration.to_dict(), "notes": json_value(self.notes)}

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())
