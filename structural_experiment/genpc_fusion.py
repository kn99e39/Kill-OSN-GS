"""A-gated implementation of the frozen GenPC downstream fusion boundary.

The implementation is deliberately an operator, not an association solver.
The association declaration only determines which O/G region pairs are
eligible for the existing close-point removal query.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .experiment1_contracts import AssociationDeclaration, ContractError, RegionPartition
from .genpc_contracts import GenPCReconciliationSpec


ClosePointFilter = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray, np.ndarray]]
FpsSampler = Callable[[np.ndarray, int], np.ndarray]
NoiseFilter = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class GenPCFusionBackend:
    """Dependency boundary for the official Open3D/fpsample GenPC calls."""

    close_point_filter: ClosePointFilter
    fps_sampler: FpsSampler
    noise_filter: NoiseFilter


@dataclass(frozen=True)
class GenPCFusionResult:
    """All point accounting and provenance needed for an auditable condition."""

    points: np.ndarray
    colors: np.ndarray
    removed_g_mask: np.ndarray
    retained_g_indices: np.ndarray
    pre_fps_union_points: np.ndarray
    pre_fps_union_colors: np.ndarray
    pre_fps_union_provenance: tuple[tuple[str, int], ...]
    fps_selected_union_indices: np.ndarray
    final_selected_union_indices: np.ndarray
    removal_attribution: tuple[dict[str, Any], ...]
    eligible_generated_point_count: int
    close_point_query_count: int
    pair_eligibility_comparison_count: int

    @property
    def removed_g_indices(self) -> np.ndarray:
        return np.flatnonzero(self.removed_g_mask).astype(np.int64, copy=False)

    @property
    def pre_fps_union_count(self) -> int:
        return int(len(self.pre_fps_union_points))

    @property
    def fps_selected_count(self) -> int:
        return int(len(self.fps_selected_union_indices))

    @property
    def final_count(self) -> int:
        return int(len(self.points))

    def accounting(self, *, observed_count: int, generated_count: int, spec: GenPCReconciliationSpec) -> dict[str, Any]:
        return {
            "observed_count": int(observed_count),
            "generated_count": int(generated_count),
            "removed_generated_count": int(len(self.removed_g_indices)),
            "retained_generated_count": int(len(self.retained_g_indices)),
            "pre_fps_union_count": self.pre_fps_union_count,
            "fps_requested_count": spec.fps_requested_count,
            "fps_selected_count": self.fps_selected_count,
            "final_count": self.final_count,
            "eligible_generated_point_count": self.eligible_generated_point_count,
            "close_point_query_count": self.close_point_query_count,
            "pair_eligibility_comparison_count": self.pair_eligibility_comparison_count,
        }


def _validate_points(points: np.ndarray, colors: np.ndarray, label: str) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ContractError(f"{label} points must be N x 3")
    if colors.shape != points.shape:
        raise ContractError(f"{label} colors must have the same shape as points")


def _owner_by_index(partition: RegionPartition, point_count: int, label: str) -> dict[int, str]:
    if partition.raw_artifact.vertex_count != point_count:
        raise ContractError(f"{label} partition vertex count does not match exact point count")
    owner: dict[int, str] = {}
    for region in partition.regions:
        for index in region.vertex_indices:
            if index in owner:
                raise ContractError(f"{label} partition has overlapping ownership at point {index}")
            owner[index] = region.region_id
    if len(owner) != point_count:
        raise ContractError(f"{label} partition must cover every exact point exactly once for fusion")
    return owner


def _indices_for_region(partition: RegionPartition, region_id: str) -> np.ndarray:
    for region in partition.regions:
        if region.region_id == region_id:
            return np.asarray(region.vertex_indices, dtype=np.int64)
    raise ContractError(f"association references unknown region {region_id!r}")


def _validate_backend_indices(indices: np.ndarray, upper_bound: int, label: str) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64)
    if result.ndim != 1 or len(set(result.tolist())) != len(result) or np.any(result < 0) or np.any(result >= upper_bound):
        raise ContractError(f"{label} must be unique one-dimensional in-bounds indices")
    return result


class AGatedGenPCFusion:
    """Run one frozen O/G fusion under one predeclared association A."""

    def __init__(
        self,
        *,
        observed_points: np.ndarray,
        observed_colors: np.ndarray,
        generated_points: np.ndarray,
        generated_colors: np.ndarray,
        observed_partition: RegionPartition,
        generated_partition: RegionPartition,
        reconciliation: GenPCReconciliationSpec,
        backend: GenPCFusionBackend,
    ) -> None:
        _validate_points(observed_points, observed_colors, "observed")
        _validate_points(generated_points, generated_colors, "generated")
        self.observed_points = observed_points
        self.observed_colors = observed_colors
        self.generated_points = generated_points
        self.generated_colors = generated_colors
        self.observed_partition = observed_partition
        self.generated_partition = generated_partition
        self.reconciliation = reconciliation
        self.backend = backend
        self._observed_owner = _owner_by_index(observed_partition, len(observed_points), "observed")
        self._generated_owner = _owner_by_index(generated_partition, len(generated_points), "generated")

    def run(self, association: AssociationDeclaration, *, numpy_rng_state: tuple[Any, ...] | None = None) -> GenPCFusionResult:
        observed_region_ids = {region.region_id for region in self.observed_partition.regions}
        generated_region_ids = {region.region_id for region in self.generated_partition.regions}
        links = tuple(association.links)
        if any(left not in observed_region_ids or right not in generated_region_ids for left, right in links):
            raise ContractError("association contains a region ID outside the frozen O/G partitions")
        links_by_generated: dict[str, tuple[str, ...]] = {
            region.region_id: tuple(left for left, right in links if right == region.region_id)
            for region in self.generated_partition.regions
        }

        removed_mask = np.zeros(len(self.generated_points), dtype=bool)
        attribution: list[dict[str, Any]] = []
        eligible_generated = 0
        close_queries = 0
        pair_comparisons = 0
        for generated_region in self.generated_partition.regions:
            generated_indices = np.asarray(generated_region.vertex_indices, dtype=np.int64)
            allowed_observed_regions = links_by_generated[generated_region.region_id]
            pair_comparisons += len(generated_indices) * len(allowed_observed_regions)
            if not allowed_observed_regions:
                continue
            allowed_observed_indices = np.concatenate(
                [_indices_for_region(self.observed_partition, region_id) for region_id in allowed_observed_regions]
            )
            eligible_generated += len(generated_indices)
            close_queries += len(generated_indices)
            keep, nearest_local, nearest_squared = self.backend.close_point_filter(
                self.observed_points[allowed_observed_indices],
                self.generated_points[generated_indices],
                self.reconciliation.distance_threshold,
            )
            keep = np.asarray(keep, dtype=bool)
            nearest_local = np.asarray(nearest_local, dtype=np.int64)
            nearest_squared = np.asarray(nearest_squared, dtype=np.float64)
            if keep.shape != (len(generated_indices),) or nearest_local.shape != keep.shape or nearest_squared.shape != keep.shape:
                raise ContractError("close-point backend returned arrays with the wrong shape")
            if np.any(nearest_local[nearest_local >= 0] >= len(allowed_observed_indices)):
                raise ContractError("close-point backend returned an invalid nearest observed index")
            for local_index, should_keep in enumerate(keep):
                if should_keep:
                    continue
                generated_index = int(generated_indices[local_index])
                observed_index = int(allowed_observed_indices[nearest_local[local_index]])
                observed_region_id = self._observed_owner[observed_index]
                removed_mask[generated_index] = True
                attribution.append(
                    {
                        "observed_region_id": observed_region_id,
                        "generated_region_id": generated_region.region_id,
                        "observed_index": observed_index,
                        "generated_index": generated_index,
                        "squared_distance": float(nearest_squared[local_index]),
                    }
                )

        retained_g_indices = np.flatnonzero(~removed_mask).astype(np.int64, copy=False)
        pre_fps_union_points = np.concatenate((self.observed_points, self.generated_points[retained_g_indices]), axis=0)
        pre_fps_union_colors = np.concatenate((self.observed_colors, self.generated_colors[retained_g_indices]), axis=0)
        provenance = tuple(
            [("O", int(index)) for index in range(len(self.observed_points))]
            + [("G", int(index)) for index in retained_g_indices]
        )

        @contextmanager
        def restore_rng():
            previous = np.random.get_state() if numpy_rng_state is not None else None
            if numpy_rng_state is not None:
                np.random.set_state(numpy_rng_state)
            try:
                yield
            finally:
                if previous is not None:
                    np.random.set_state(previous)

        with restore_rng():
            fps_selected = _validate_backend_indices(
                self.backend.fps_sampler(pre_fps_union_points, self.reconciliation.fps_requested_count),
                len(pre_fps_union_points),
                "FPS-selected indices",
            )
            sampled_points = pre_fps_union_points[fps_selected]
            sampled_colors = pre_fps_union_colors[fps_selected]
            final_points, final_colors, final_local_indices = self.backend.noise_filter(
                sampled_points, sampled_colors, self.reconciliation.noise_std_ratio
            )
        final_local_indices = _validate_backend_indices(final_local_indices, len(fps_selected), "final filter indices")
        final_union_indices = fps_selected[final_local_indices]
        _validate_points(np.asarray(final_points), np.asarray(final_colors), "final")
        return GenPCFusionResult(
            points=np.asarray(final_points),
            colors=np.asarray(final_colors),
            removed_g_mask=removed_mask,
            retained_g_indices=retained_g_indices,
            pre_fps_union_points=pre_fps_union_points,
            pre_fps_union_colors=pre_fps_union_colors,
            pre_fps_union_provenance=provenance,
            fps_selected_union_indices=fps_selected,
            final_selected_union_indices=final_union_indices,
            removal_attribution=tuple(attribution),
            eligible_generated_point_count=eligible_generated,
            close_point_query_count=close_queries,
            pair_eligibility_comparison_count=pair_comparisons,
        )


def open3d_close_point_filter(
    observed_points: np.ndarray, generated_points: np.ndarray, distance_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the same squared-distance Open3D KDTree semantics as reg_xyz."""
    import open3d as o3d

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(observed_points)
    tree = o3d.geometry.KDTreeFlann(source)
    keep = np.ones(len(generated_points), dtype=bool)
    nearest = np.full(len(generated_points), -1, dtype=np.int64)
    squared = np.full(len(generated_points), np.inf, dtype=np.float64)
    for index, point in enumerate(generated_points):
        _, indices, distances = tree.search_knn_vector_3d(point, 1)
        if indices:
            nearest[index] = int(indices[0])
            squared[index] = float(distances[0])
            if distances[0] < distance_threshold:
                keep[index] = False
    return keep, nearest, squared


def native_fps_sampler(points: np.ndarray, requested_count: int) -> np.ndarray:
    from fpsample import fps_sampling

    return np.asarray(fps_sampling(points, requested_count), dtype=np.int64)


def native_noise_filter(
    points: np.ndarray, colors: np.ndarray, std_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d
    from utils.dataUtils import numpy2o3d, remove_noise_from_point_cloud

    cloud = numpy2o3d(points, colors)
    filtered = remove_noise_from_point_cloud(cloud, std_ratio=std_ratio)
    filtered_points = np.asarray(filtered.points).copy()
    filtered_colors = np.asarray(filtered.colors).copy()
    # Open3D's select_by_index preserves the returned index order.  Recreate
    # the same selected indices to retain final provenance in the report.
    _, indices = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
    local_indices = np.asarray(indices, dtype=np.int64)
    if not np.array_equal(filtered_points, points[local_indices]):
        raise ContractError("native statistical-filter provenance disagrees with Open3D output")
    return filtered_points, filtered_colors, local_indices


def native_genpc_backend() -> GenPCFusionBackend:
    return GenPCFusionBackend(
        close_point_filter=open3d_close_point_filter,
        fps_sampler=native_fps_sampler,
        noise_filter=native_noise_filter,
    )
