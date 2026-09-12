"""Lossless binding for the frozen GenPC NPZ fusion boundary.

The sidecars at the registration/fusion boundary are the authoritative O/G
inputs for Experiment 1.  This module intentionally contains no geometry
processing: it validates the file and exposes the arrays with their original
order and indices intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiment1_contracts import ContractError, RawArtifactLineage, sha256_file


def _array_content_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes of an array without casting it."""
    contiguous = np.ascontiguousarray(array)
    digest = __import__("hashlib").sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ExactNpzPointCloud:
    """A read-only, content-addressed view of one frozen GenPC sidecar."""

    path: Path
    file_sha256: str
    points: np.ndarray
    colors: np.ndarray

    def __post_init__(self) -> None:
        if len(self.file_sha256) != 64:
            raise ContractError("GenPC sidecar file_sha256 must be a SHA-256")
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ContractError("GenPC points must be an N x 3 array")
        if self.colors.ndim != 2 or self.colors.shape != self.points.shape:
            raise ContractError("GenPC colors must have the same N x 3 shape as points")
        if self.points.dtype != np.dtype("float64") or self.colors.dtype != np.dtype("float64"):
            raise ContractError("GenPC exact sidecars must retain float64 points and colors")
        if self.points.flags.writeable or self.colors.flags.writeable:
            raise ContractError("GenPC exact arrays must be read-only")

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def original_indices(self) -> np.ndarray:
        indices = np.arange(self.point_count, dtype=np.int64)
        indices.flags.writeable = False
        return indices

    @property
    def points_sha256(self) -> str:
        return _array_content_sha256(self.points)

    @property
    def colors_sha256(self) -> str:
        return _array_content_sha256(self.colors)

    @property
    def content_fingerprints(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "points_sha256": self.points_sha256,
            "colors_sha256": self.colors_sha256,
            "point_count": self.point_count,
            "points_dtype": self.points.dtype.str,
            "colors_dtype": self.colors.dtype.str,
            "array_order": "original NPZ order; no reorder/resample/normalization",
        }

    def raw_artifact_lineage(self, *, artifact_id: str, source_revision: str = "") -> RawArtifactLineage:
        """Create the existing ownership lineage for RegionPartition contracts."""
        return RawArtifactLineage(
            artifact_id=artifact_id,
            original_file_sha256=self.file_sha256,
            representation="point_cloud",
            vertex_count=self.point_count,
            face_count=0,
            source_locator=str(self.path),
            source_revision=source_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            **self.content_fingerprints,
        }


def load_exact_npz(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_point_count: int,
) -> ExactNpzPointCloud:
    """Load and validate a frozen NPZ without changing its numeric content."""
    source = Path(path)
    actual_sha256 = sha256_file(source)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ContractError(
            f"GenPC sidecar SHA-256 mismatch for {source}: expected {expected_sha256}, got {actual_sha256}"
        )
    with np.load(source, allow_pickle=False) as data:
        keys = tuple(sorted(data.files))
        if keys != ("colors", "points"):
            raise ContractError(f"unexpected GenPC NPZ keys {keys}; expected exactly ('colors', 'points')")
        # Copies are deliberate: the NPZ archive must not remain a mutable
        # backing store for the experiment's frozen arrays.
        points = np.array(data["points"], copy=True)
        colors = np.array(data["colors"], copy=True)
    points.flags.writeable = False
    colors.flags.writeable = False
    if len(points) != expected_point_count:
        raise ContractError(
            f"GenPC point-count mismatch for {source}: expected {expected_point_count}, got {len(points)}"
        )
    return ExactNpzPointCloud(source, actual_sha256, points, colors)
