"""Known-identity fixtures for testing A under the frozen GenPC R.

These are geometry constructors, not a reconciliation implementation or an
attachment solver.  Every dimension is a declared multiple of tau.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .experiment1_contracts import AssociationDeclaration, RawArtifactLineage, RegionHandle, RegionPartition


TAU = 0.01
GRID_STEP_RATIO = 0.25
OBSERVED_HALF_X_RATIO = 32
CONTINUATION_HALF_X_RATIO = 32
OVERLAP_X_RATIO = 8
HALF_Y_RATIO = 24
PARALLEL_OFFSET_RATIO = 0.5
JUNCTION_HALF_Z_RATIO = 24


def _grid(first: np.ndarray, second: np.ndarray, *, plane: str, fixed: float) -> np.ndarray:
    left, right = np.meshgrid(first, second, indexing="ij")
    if plane == "xy":
        return np.column_stack((left.ravel(), right.ravel(), np.full(left.size, fixed)))
    if plane == "yz":
        return np.column_stack((np.full(left.size, fixed), left.ravel(), right.ravel()))
    raise ValueError(f"unsupported plane {plane!r}")


def _axis(low_ratio: int, high_ratio: int) -> np.ndarray:
    return np.arange(low_ratio, high_ratio + GRID_STEP_RATIO / 2, GRID_STEP_RATIO, dtype=np.float64) * TAU


def _lineage(artifact_id: str, points: np.ndarray) -> RawArtifactLineage:
    digest = sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
    return RawArtifactLineage(artifact_id, digest, "point_cloud", len(points), 0, source_locator=f"controlled://{artifact_id}")


def _partition(artifact_id: str, points: np.ndarray, regions: tuple[tuple[str, np.ndarray], ...]) -> RegionPartition:
    return RegionPartition(
        partition_id=f"{artifact_id}-partition-v1",
        raw_artifact=_lineage(artifact_id, points),
        source="synthetic_gt",
        regions=tuple(RegionHandle(region_id, tuple(int(index) for index in indices), label=region_id) for region_id, indices in regions),
        creation_protocol_id="controlled-structural-fixtures-v1",
        notes="Known construction identity; dimensions derived solely from tau.",
    )


@dataclass(frozen=True)
class ControlledFixture:
    fixture_id: str
    observed_points: np.ndarray
    observed_colors: np.ndarray
    generated_points: np.ndarray
    generated_colors: np.ndarray
    gt_points: np.ndarray
    observed_partition: RegionPartition
    generated_partition: RegionPartition
    generated_regions: dict[str, np.ndarray]
    associations: dict[str, AssociationDeclaration]
    protocol: dict[str, float | int | str]


def _assemble(fixture_id: str, observed: np.ndarray, correct: np.ndarray, distinct: np.ndarray, gt: np.ndarray) -> ControlledFixture:
    generated = np.concatenate((correct, distinct), axis=0)
    correct_indices = np.arange(len(correct), dtype=np.int64)
    distinct_indices = np.arange(len(correct), len(generated), dtype=np.int64)
    observed_partition = _partition(f"{fixture_id}:O", observed, ((f"{fixture_id}:O_surface", np.arange(len(observed))),))
    generated_partition = _partition(
        f"{fixture_id}:G",
        generated,
        ((f"{fixture_id}:G_correct", correct_indices), (f"{fixture_id}:G_distinct", distinct_indices)),
    )
    observed_id = f"{fixture_id}:O_surface"
    correct_id = f"{fixture_id}:G_correct"
    distinct_id = f"{fixture_id}:G_distinct"
    return ControlledFixture(
        fixture_id=fixture_id,
        observed_points=observed,
        observed_colors=np.tile(np.array([[0.20, 0.45, 0.80]]), (len(observed), 1)),
        generated_points=generated,
        generated_colors=np.concatenate((
            np.tile(np.array([[0.20, 0.75, 0.35]]), (len(correct), 1)),
            np.tile(np.array([[0.95, 0.45, 0.15]]), (len(distinct), 1)),
        )),
        gt_points=gt,
        observed_partition=observed_partition,
        generated_partition=generated_partition,
        generated_regions={correct_id: correct_indices, distinct_id: distinct_indices},
        associations={
            "A_native": AssociationDeclaration("A_native", ((observed_id, correct_id), (observed_id, distinct_id))),
            "A_wrong": AssociationDeclaration("A_wrong", ((observed_id, distinct_id),)),
            "A_oracle": AssociationDeclaration("A_oracle", ((observed_id, correct_id),)),
        },
        protocol={
            "tau": TAU,
            "grid_step_ratio": GRID_STEP_RATIO,
            "observed_half_x_ratio": OBSERVED_HALF_X_RATIO,
            "continuation_half_x_ratio": CONTINUATION_HALF_X_RATIO,
            "overlap_x_ratio": OVERLAP_X_RATIO,
            "half_y_ratio": HALF_Y_RATIO,
            "parallel_offset_ratio": PARALLEL_OFFSET_RATIO,
            "junction_half_z_ratio": JUNCTION_HALF_Z_RATIO,
            "rule": "all dimensions are fixed tau multiples; no sweep",
        },
    )


def parallel_sheet_fixture() -> ControlledFixture:
    """Top continuation plus a valid nearby underside sheet within tau."""
    y = _axis(-HALF_Y_RATIO, HALF_Y_RATIO)
    observed = _grid(_axis(-OBSERVED_HALF_X_RATIO, 0), y, plane="xy", fixed=0.0)
    correct = _grid(_axis(-OVERLAP_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    distinct = _grid(_axis(-OVERLAP_X_RATIO, 0), y, plane="xy", fixed=-PARALLEL_OFFSET_RATIO * TAU)
    gt_top = _grid(_axis(-OBSERVED_HALF_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    return _assemble("parallel_sheet", observed, correct, distinct, np.concatenate((gt_top, distinct), axis=0))


def junction_fixture() -> ControlledFixture:
    """Horizontal continuation plus a legitimate near vertical junction sheet."""
    y = _axis(-HALF_Y_RATIO, HALF_Y_RATIO)
    observed = _grid(_axis(-OBSERVED_HALF_X_RATIO, 0), y, plane="xy", fixed=0.0)
    correct = _grid(_axis(-OVERLAP_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    distinct = _grid(y, _axis(-JUNCTION_HALF_Z_RATIO, JUNCTION_HALF_Z_RATIO), plane="yz", fixed=-PARALLEL_OFFSET_RATIO * TAU)
    gt_top = _grid(_axis(-OBSERVED_HALF_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    return _assemble("junction", observed, correct, distinct, np.concatenate((gt_top, distinct), axis=0))


def controlled_fixtures() -> tuple[ControlledFixture, ControlledFixture]:
    return parallel_sheet_fixture(), junction_fixture()
