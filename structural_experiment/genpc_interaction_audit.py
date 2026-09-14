"""Scale-locked O/G interaction diagnostics for frozen GenPC boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TAU = 0.01


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def native_interaction_diagnostic(observed: np.ndarray, generated: np.ndarray, *, tau: float = TAU) -> dict:
    """Report raw native O-G contacts and tau-connected participating G patches."""
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError:  # dependency-free unit fixtures only
        cKDTree = None
    if cKDTree is None:
        nearest_observed = np.empty(len(generated), dtype=np.int64)
        distances = np.empty(len(generated), dtype=np.float64)
        for index, point in enumerate(generated):
            squared = np.sum((observed - point) ** 2, axis=1)
            nearest_observed[index] = int(np.argmin(squared))
            distances[index] = float(np.sqrt(squared[nearest_observed[index]]))
    else:
        observed_tree = cKDTree(observed)
        distances, nearest_observed = observed_tree.query(generated, k=1)
    interacting_indices = np.flatnonzero(distances < tau).astype(np.int64)
    interacting_points = generated[interacting_indices]
    component_by_generated = np.full(len(generated), -1, dtype=np.int64)
    components: list[dict] = []
    if len(interacting_indices):
        union = _UnionFind(len(interacting_indices))
        if cKDTree is None:
            pairs = (
                (left, right)
                for left in range(len(interacting_points))
                for right in range(left + 1, len(interacting_points))
                if np.linalg.norm(interacting_points[left] - interacting_points[right]) < tau
            )
        else:
            pairs = cKDTree(interacting_points).query_pairs(tau)
        for left, right in pairs:
            union.union(left, right)
        members: dict[int, list[int]] = {}
        for local_index in range(len(interacting_indices)):
            members.setdefault(union.find(local_index), []).append(local_index)
        ordered = sorted(members.values(), key=lambda values: (min(values), len(values)))
        for component_id, local_members in enumerate(ordered):
            global_members = interacting_indices[np.asarray(local_members, dtype=np.int64)]
            component_by_generated[global_members] = component_id
            support = np.unique(nearest_observed[global_members])
            components.append({
                "component_id": component_id,
                "generated_indices": global_members.tolist(),
                "generated_count": int(len(global_members)),
                "observed_support_indices": support.astype(np.int64).tolist(),
                "observed_support_count": int(len(support)),
                "squared_distance_min": float(np.min(distances[global_members] ** 2)),
                "squared_distance_max": float(np.max(distances[global_members] ** 2)),
            })
    component_support_by_observed: dict[int, list[int]] = {}
    for generated_index in interacting_indices:
        component_support_by_observed.setdefault(int(nearest_observed[generated_index]), []).append(int(component_by_generated[generated_index]))
    multiplicity = [
        {"observed_index": observed_index, "component_ids": sorted(set(component_ids)), "component_count": len(set(component_ids)), "interacting_generated_count": len(component_ids)}
        for observed_index, component_ids in sorted(component_support_by_observed.items())
    ]
    max_event = max(multiplicity, key=lambda item: (item["component_count"], item["interacting_generated_count"], -item["observed_index"]), default=None)
    return {
        "tau": tau,
        "tau_squared": tau * tau,
        "generated_count": int(len(generated)),
        "observed_count": int(len(observed)),
        "interacting_generated_indices": interacting_indices.tolist(),
        "nearest_observed_indices": nearest_observed[interacting_indices].astype(np.int64).tolist(),
        "squared_distances": (distances[interacting_indices] ** 2).tolist(),
        "interaction_density_over_observed": float(len(set(nearest_observed[interacting_indices])) / len(observed)) if len(observed) else 0.0,
        "components": components,
        "component_id_by_generated": component_by_generated.tolist(),
        "observed_component_multiplicity": multiplicity,
        "strongest_event": max_event,
        "multiple_component_opportunity": bool(max_event is not None and max_event["component_count"] > 1),
    }
