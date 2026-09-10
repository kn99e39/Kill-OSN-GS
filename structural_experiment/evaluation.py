"""Metric plumbing with explicit unavailable states, never aggregate scores."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable

from .contracts import Association, EvaluationReference, Geometry, ReconciledResult, Surface


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    available: bool
    value: dict[str, Any] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"metric_id": self.metric_id, "available": self.available, "value": self.value, "reason": self.reason}


def _select(geometry: Geometry, surface_ids: Iterable[str]) -> tuple[Surface, ...]:
    wanted = set(surface_ids)
    by_id = {surface.surface_id: surface for surface in geometry.surfaces}
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(f"Geometry lacks evaluation surface IDs: {sorted(missing)}")
    return tuple(by_id[key] for key in sorted(wanted))


def _points(surfaces: Iterable[Surface]) -> list[tuple[float, float, float]]:
    return [point for surface in surfaces for point in surface.vertices]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _nearest(point: tuple[float, float, float], candidates: list[tuple[float, float, float]]) -> tuple[int, float]:
    index, distance = min(enumerate(_distance(point, candidate) for candidate in candidates), key=lambda pair: pair[1])
    return index, distance


def geometry_distance(result: Geometry, reference: Geometry, surface_ids: Iterable[str]) -> MetricResult:
    candidate = _points(_select(result, surface_ids))
    target = _points(_select(reference, surface_ids))
    if not candidate or not target:
        return MetricResult("symmetric_mean_nearest_distance", False, reason="Region has no point samples.")
    forward = sum(_nearest(point, target)[1] for point in candidate) / len(candidate)
    backward = sum(_nearest(point, candidate)[1] for point in target) / len(target)
    return MetricResult("symmetric_mean_nearest_distance", True, {
        "result_to_reference": forward, "reference_to_result": backward,
        "symmetric_mean": (forward + backward) / 2, "units": result.units,
        "result_point_count": len(candidate), "reference_point_count": len(target),
    })


def normal_discrepancy(result: Geometry, reference: Geometry, surface_ids: Iterable[str]) -> MetricResult:
    result_surfaces, reference_surfaces = _select(result, surface_ids), _select(reference, surface_ids)
    if any(surface.normals is None for surface in result_surfaces + reference_surfaces):
        return MetricResult("normal_discrepancy", False, reason="Normals are not present on every selected surface.")
    result_points = _points(result_surfaces)
    result_normals = [normal for surface in result_surfaces for normal in surface.normals or ()]
    reference_points = _points(reference_surfaces)
    reference_normals = [normal for surface in reference_surfaces for normal in surface.normals or ()]
    discrepancies = []
    for point, normal in zip(result_points, result_normals):
        index, _ = _nearest(point, reference_points)
        other = reference_normals[index]
        magnitude_a = sqrt(sum(component * component for component in normal))
        magnitude_b = sqrt(sum(component * component for component in other))
        if magnitude_a == 0 or magnitude_b == 0:
            return MetricResult("normal_discrepancy", False, reason="A selected normal has zero magnitude.")
        cosine = sum(left * right for left, right in zip(normal, other)) / (magnitude_a * magnitude_b)
        discrepancies.append(1.0 - max(-1.0, min(1.0, cosine)))
    return MetricResult("normal_discrepancy", True, {"mean_one_minus_cosine": sum(discrepancies) / len(discrepancies)})


def association_correctness(candidate: Association, reference: EvaluationReference) -> MetricResult:
    if reference.oracle_association is None:
        return MetricResult("association_correctness", False, reason="No oracle association is available.")
    expected = {tuple(sorted(entry.to_dict().items())) for entry in reference.oracle_association.entries if entry.state == "associated"}
    actual = {tuple(sorted(entry.to_dict().items())) for entry in candidate.entries if entry.state == "associated"}
    true_positive = len(expected & actual)
    precision = true_positive / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = true_positive / len(expected) if expected else 1.0
    return MetricResult("association_correctness", True, {"exact_match": actual == expected, "precision": precision,
        "recall": recall, "true_positive": true_positive, "candidate_count": len(actual), "oracle_count": len(expected)})


def unmatched_correctness(candidate: Association, reference: EvaluationReference) -> MetricResult:
    if reference.oracle_association is None:
        return MetricResult("unmatched_correctness", False, reason="No oracle association is available.")
    states = {"unmatched_observation", "unmatched_generated"}
    expected = {tuple(sorted(entry.to_dict().items())) for entry in reference.oracle_association.entries if entry.state in states}
    actual = {tuple(sorted(entry.to_dict().items())) for entry in candidate.entries if entry.state in states}
    true_positive = len(expected & actual)
    precision = true_positive / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = true_positive / len(expected) if expected else 1.0
    return MetricResult("unmatched_correctness", True, {"exact_match": actual == expected, "precision": precision,
        "recall": recall, "true_positive": true_positive, "candidate_count": len(actual), "oracle_count": len(expected)})


def evaluate(result: ReconciledResult, reference: EvaluationReference) -> tuple[dict[str, list[MetricResult]], list[str]]:
    by_region: dict[str, list[MetricResult]] = {}
    warnings: list[str] = []
    for region in reference.regions:
        values = [geometry_distance(result.geometry, reference.geometry, region.surface_ids),
                  normal_discrepancy(result.geometry, reference.geometry, region.surface_ids)]
        # The base contract has no edge/curve or graph topology; making up a seam/bridge measure would bias later work.
        values += [MetricResult("seam_error", False, reason="No explicit interface curve or seam correspondence is available."),
                   MetricResult("false_bridge_accounting", False, reason="No validated topology graph is available in this contract.")]
        by_region[region.region_id] = values
        warnings.extend(f"{region.region_id}:{value.metric_id}:{value.reason}" for value in values if not value.available)
    by_region["association"] = [association_correctness(result.association, reference), unmatched_correctness(result.association, reference)]
    warnings.extend(f"association:{value.metric_id}:{value.reason}" for value in by_region["association"] if not value.available)
    return by_region, warnings
