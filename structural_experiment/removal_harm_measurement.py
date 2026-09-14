"""Direct pre-FPS GT-coverage measurement for native GenPC removal."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def nearest_distances(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest distances and indices, using the frozen runtime when available."""
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError:
        distances = np.empty(len(query), dtype=np.float64)
        indices = np.empty(len(query), dtype=np.int64)
        for index, point in enumerate(query):
            squared = np.sum((reference - point) ** 2, axis=1)
            indices[index] = int(np.argmin(squared))
            distances[index] = float(np.sqrt(squared[indices[index]]))
        return distances, indices
    distances, indices = cKDTree(reference).query(query, k=1)
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Return each point's distance to a different point in the same cloud."""
    if len(points) < 2:
        return np.empty(0, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError:
        result = np.empty(len(points), dtype=np.float64)
        for index, point in enumerate(points):
            squared = np.sum((points - point) ** 2, axis=1)
            squared[index] = np.inf
            result[index] = float(np.sqrt(np.min(squared)))
        return result
    distances, _ = cKDTree(points).query(points, k=2)
    return np.asarray(distances[:, 1], dtype=np.float64)


def numerical_policy(points: np.ndarray) -> dict[str, float | str]:
    coordinate_scale = max(1.0, float(np.max(np.abs(points))) if points.size else 1.0)
    epsilon = float(np.finfo(np.float64).eps)
    tolerance = 64.0 * epsilon * coordinate_scale
    return {
        "dtype": "float64",
        "machine_epsilon": epsilon,
        "coordinate_scale": coordinate_scale,
        "tolerance": tolerance,
        "formula": "64 * finfo(float64).eps * max(1.0, max(abs(all scene coordinates)))",
        "meaningful_positive_delta": "delta > tolerance",
        "meaningful_negative_delta": "delta < -tolerance",
    }


def array_identity(points: np.ndarray, provenance: list[dict[str, int]]) -> dict[str, Any]:
    provenance_bytes = json.dumps(provenance, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "count": int(len(points)),
        "points_sha256": hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "ordering": "O original order followed by retained G original order",
    }


def coverage_harm(
    observed: np.ndarray,
    generated: np.ndarray,
    removed_g_mask: np.ndarray,
    gt: np.ndarray,
) -> dict[str, Any]:
    retained_indices = np.flatnonzero(~removed_g_mask).astype(np.int64)
    all_union = np.concatenate((observed, generated), axis=0)
    native_union = np.concatenate((observed, generated[retained_indices]), axis=0)
    all_provenance = [
        {"kind": "O", "index": index} for index in range(len(observed))
    ] + [
        {"kind": "G", "index": index} for index in range(len(generated))
    ]
    native_provenance = [
        {"kind": "O", "index": index} for index in range(len(observed))
    ] + [
        {"kind": "G", "index": int(index)} for index in retained_indices
    ]
    d_all, nearest_all = nearest_distances(all_union, gt)
    d_native, nearest_native = nearest_distances(native_union, gt)
    delta = d_native - d_all
    policy = numerical_policy(np.concatenate((observed, generated, gt), axis=0))
    tolerance = float(policy["tolerance"])
    positive_indices = np.flatnonzero(delta > tolerance).astype(np.int64)
    negative_beyond_tolerance = int(np.count_nonzero(delta < -tolerance))
    positive = delta[positive_indices]
    gt_nn = nearest_neighbor_distances(gt)
    gt_median_spacing = float(np.median(gt_nn)) if len(gt_nn) else 0.0
    removed_indices = np.flatnonzero(removed_g_mask).astype(np.int64)
    harmed_rows: list[dict[str, Any]] = []
    if len(removed_indices):
        removed_points = generated[removed_indices]
        removed_gt_distances, nearest_removed_local = nearest_distances(removed_points, gt)
        removed_o_distances, _ = nearest_distances(observed, removed_points)
    else:
        removed_gt_distances = np.empty(0, dtype=np.float64)
        nearest_removed_local = np.empty(0, dtype=np.int64)
        removed_o_distances = np.empty(0, dtype=np.float64)
    for gt_index in positive_indices:
        all_global_index = int(nearest_all[gt_index])
        all_source = all_provenance[all_global_index]
        removed_g_index = -1
        removed_gt_distance = None
        removed_to_o_distance = None
        all_is_removed = all_source["kind"] == "G" and bool(removed_g_mask[all_source["index"]])
        if all_is_removed:
            removed_g_index = int(all_source["index"])
            removed_gt_distance = float(np.linalg.norm(generated[removed_g_index] - gt[gt_index]))
            removed_to_o_distance = float(np.min(np.linalg.norm(observed - generated[removed_g_index], axis=1)))
        elif len(removed_indices):
            local = int(nearest_removed_local[gt_index])
            removed_g_index = int(removed_indices[local])
            removed_gt_distance = float(removed_gt_distances[gt_index])
            removed_to_o_distance = float(removed_o_distances[local])
        harmed_rows.append({
            "gt_index": int(gt_index),
            "d_all": float(d_all[gt_index]),
            "d_native": float(d_native[gt_index]),
            "delta": float(delta[gt_index]),
            "nearest_point_provenance_in_U_all": all_source,
            "nearest_point_is_removed_G": all_is_removed,
            "removed_G_index": removed_g_index,
            "removed_G_to_GT_distance": removed_gt_distance,
            "removed_G_to_nearest_O_distance": removed_to_o_distance,
        })
    quantile_levels = (0.5, 0.9, 0.95, 0.99, 1.0)
    positive_quantiles = {
        str(level): float(np.quantile(positive, level)) for level in quantile_levels
    } if len(positive) else {}
    removed_gt = nearest_distances(gt, generated[removed_indices])[0] if len(removed_indices) else np.empty(0)
    retained_gt = nearest_distances(gt, generated[retained_indices])[0] if len(retained_indices) else np.empty(0)
    return {
        "U_all": all_union,
        "U_native": native_union,
        "all_provenance": all_provenance,
        "native_provenance": native_provenance,
        "d_all": d_all,
        "d_native": d_native,
        "delta": delta,
        "nearest_all_indices": nearest_all,
        "nearest_native_indices": nearest_native,
        "harmed_gt_rows": harmed_rows,
        "numerical_policy": policy,
        "summary": {
            "gt_point_count": int(len(gt)),
            "removed_G_point_count": int(len(removed_indices)),
            "delta_sum": float(np.sum(delta)),
            "delta_mean": float(np.mean(delta)) if len(delta) else 0.0,
            "delta_median": float(np.median(delta)) if len(delta) else 0.0,
            "delta_max": float(np.max(delta)) if len(delta) else 0.0,
            "strictly_positive_delta_count": int(len(positive_indices)),
            "positive_delta_quantiles": positive_quantiles,
            "negative_delta_beyond_tolerance_count": negative_beyond_tolerance,
            "gt_median_1nn_spacing": gt_median_spacing,
            "mean_delta_over_gt_median_spacing": float(np.mean(delta) / gt_median_spacing) if gt_median_spacing else None,
            "max_delta_over_gt_median_spacing": float(np.max(delta) / gt_median_spacing) if len(delta) and gt_median_spacing else None,
            "mean_U_all_to_GT_distance": float(np.mean(nearest_distances(gt, all_union)[0])) if len(gt) else 0.0,
            "mean_U_native_to_GT_distance": float(np.mean(nearest_distances(gt, native_union)[0])) if len(gt) else 0.0,
            "removed_G_to_GT_distance_distribution": {
                "count": int(len(removed_gt)),
                "mean": float(np.mean(removed_gt)) if len(removed_gt) else None,
                "median": float(np.median(removed_gt)) if len(removed_gt) else None,
                "p95": float(np.quantile(removed_gt, 0.95)) if len(removed_gt) else None,
                "max": float(np.max(removed_gt)) if len(removed_gt) else None,
            },
            "retained_G_to_GT_distance_distribution": {
                "count": int(len(retained_gt)),
                "mean": float(np.mean(retained_gt)) if len(retained_gt) else None,
                "median": float(np.median(retained_gt)) if len(retained_gt) else None,
                "p95": float(np.quantile(retained_gt, 0.95)) if len(retained_gt) else None,
                "max": float(np.max(retained_gt)) if len(retained_gt) else None,
            },
        },
    }
