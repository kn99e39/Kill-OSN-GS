"""Read-only materialisation of externally partitioned raw geometry.

This is deliberately not a segmenter.  It accepts an externally supplied
``RegionPartition`` and retains raw vertex/face ownership in every surface
metadata record that it creates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Geometry, Surface
from .experiment1_contracts import ContractError, RawArtifactLineage, RegionHandle, RegionPartition, sha256_file
from .geometry_io import DecodedGeometry, decode_geometry


def _lineage_matches(expected: RawArtifactLineage, actual: RawArtifactLineage) -> bool:
    return (
        expected.artifact_id == actual.artifact_id
        and expected.original_file_sha256.lower() == actual.original_file_sha256.lower()
        and expected.representation == actual.representation
        and expected.vertex_count == actual.vertex_count
        and expected.face_count == actual.face_count
    )


@dataclass(frozen=True)
class PartitionedRawArtifact:
    """A raw file and a frozen external partition, verified against one another."""

    raw_path: Path
    lineage: RawArtifactLineage
    partition: RegionPartition
    geometry: DecodedGeometry

    def __post_init__(self) -> None:
        if self.lineage.artifact_id != self.partition.raw_artifact.artifact_id:
            raise ContractError("artifact and partition must use the same artifact_id")
        if self.lineage.vertex_count != len(self.geometry.positions):
            raise ContractError("decoded position count does not match raw lineage")
        if self.lineage.face_count != len(self.geometry.faces):
            raise ContractError("decoded face count does not match raw lineage")
        for face in self.geometry.faces:
            if len(face) < 3 or any(index < 0 or index >= self.lineage.vertex_count for index in face):
                raise ContractError("raw geometry contains a face with invalid vertex ownership")
        for region in self.partition.regions:
            owned_vertices = set(region.vertex_indices)
            for face_index in region.face_indices:
                if not set(self.geometry.faces[face_index]).issubset(owned_vertices):
                    raise ContractError(
                        f"region {region.region_id} owns face {face_index} but not every vertex of that face"
                    )

    @classmethod
    def from_file(cls, path: str | Path, partition: RegionPartition) -> "PartitionedRawArtifact":
        source = Path(path)
        decoded = decode_geometry(source)
        actual = RawArtifactLineage(
            artifact_id=partition.raw_artifact.artifact_id,
            original_file_sha256=sha256_file(source),
            representation="glb" if source.suffix.lower() == ".glb" else "ply",
            vertex_count=decoded.vertex_count,
            face_count=decoded.face_count,
            source_locator=str(source),
            source_revision=partition.raw_artifact.source_revision,
        )
        if not _lineage_matches(partition.raw_artifact, actual):
            raise ContractError(
                "raw file does not match the externally supplied partition lineage; "
                "do not reuse a partition after preprocessing or re-export"
            )
        return cls(source, actual, partition, decoded)

    def _surface(self, region: RegionHandle, *, surface_prefix: str, tags: tuple[str, ...]) -> Surface:
        original_indices = region.vertex_indices
        remap = {raw_index: local_index for local_index, raw_index in enumerate(original_indices)}
        owned_faces = tuple(
            tuple(remap[index] for index in self.geometry.faces[face_index])
            for face_index in region.face_indices
        )
        return Surface(
            surface_id=f"{surface_prefix}:{region.region_id}",
            region_id=region.region_id,
            vertices=tuple(self.geometry.positions[index] for index in original_indices),
            faces=owned_faces,
            tags=tags,
            metadata={
                "raw_artifact_id": self.lineage.artifact_id,
                "raw_file_sha256": self.lineage.original_file_sha256,
                "raw_vertex_indices": list(original_indices),
                "raw_face_indices": list(region.face_indices),
                "partition_id": self.partition.partition_id,
                "partition_fingerprint": self.partition.fingerprint,
                "partition_source": self.partition.source,
                "creation_protocol_id": self.partition.creation_protocol_id,
            },
        )

    def materialize_geometry(
        self,
        *,
        geometry_id: str,
        surface_prefix: str,
        units: str = "unitless",
        tags: tuple[str, ...] = (),
    ) -> Geometry:
        """Create region surfaces only from the supplied ownership indices."""
        return Geometry(
            geometry_id=geometry_id,
            surfaces=tuple(self._surface(region, surface_prefix=surface_prefix, tags=tags) for region in self.partition.regions),
            units=units,
            metadata={
                "raw_artifact": self.lineage.to_dict(),
                "partition": self.partition.to_dict(),
                "partition_fingerprint": self.partition.fingerprint,
                "materialisation": "external_partition_only",
            },
        )

    def raw_points(self, indices: tuple[int, ...]) -> tuple[tuple[float, float, float], ...]:
        if any(index >= self.lineage.vertex_count for index in indices):
            raise ContractError("selector index escapes raw artifact bounds")
        return tuple(self.geometry.positions[index] for index in indices)

    def region(self, region_id: str) -> RegionHandle:
        for region in self.partition.regions:
            if region.region_id == region_id:
                return region
        raise ContractError(f"unknown region {region_id!r} for artifact {self.lineage.artifact_id!r}")
