"""Association-independent evaluation over independently selected artifacts."""

from __future__ import annotations

from math import sqrt

from .evaluation import MetricResult
from .experiment1_contracts import ContractError, FixedEvaluationDefinition, EvaluationSelector
from .real_data import PartitionedRawArtifact


def _select(artifact: PartitionedRawArtifact, selector: EvaluationSelector) -> tuple[tuple[float, float, float], ...]:
    if selector.artifact_id != artifact.lineage.artifact_id:
        raise ContractError(
            f"selector {selector.selector_id!r} targets {selector.artifact_id!r}, "
            f"not supplied artifact {artifact.lineage.artifact_id!r}"
        )
    if selector.selection_kind == "whole_artifact":
        return artifact.geometry.positions
    if selector.selection_kind == "region_handle":
        return artifact.raw_points(artifact.region(selector.region_id).vertex_indices)
    if selector.selection_kind == "explicit_indices":
        return artifact.raw_points(selector.vertex_indices)
    if selector.selection_kind == "world_aabb":
        assert selector.world_aabb is not None
        x0, y0, z0, x1, y1, z1 = selector.world_aabb
        return tuple(
            point for point in artifact.geometry.positions
            if x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 <= point[2] <= z1
        )
    raise ContractError(f"unsupported selector kind {selector.selection_kind!r}")


def _nearest(point: tuple[float, float, float], candidates: tuple[tuple[float, float, float], ...]) -> float:
    return min(sqrt(sum((left - right) ** 2 for left, right in zip(point, candidate))) for candidate in candidates)


def evaluate_fixed_definition(
    definition: FixedEvaluationDefinition,
    *,
    result_artifact: PartitionedRawArtifact,
    reference_artifact: PartitionedRawArtifact,
) -> MetricResult:
    """Evaluate fixed result/reference selectors without reading an association.

    The caller can therefore use a result artifact and a GT artifact with
    entirely unrelated surface and region IDs.
    """
    result_points = _select(result_artifact, definition.result)
    reference_points = _select(reference_artifact, definition.reference)
    if not result_points or not reference_points:
        return MetricResult(
            definition.metric_id,
            False,
            reason="A fixed selector produced no points; the frozen protocol is inapplicable to this artifact.",
        )
    if definition.metric_id != "symmetric_mean_nearest_distance":
        return MetricResult(definition.metric_id, False, reason="Metric implementation is not registered in the dependency-free substrate.")
    result_to_reference = sum(_nearest(point, reference_points) for point in result_points) / len(result_points)
    reference_to_result = sum(_nearest(point, result_points) for point in reference_points) / len(reference_points)
    return MetricResult(
        definition.metric_id,
        True,
        {
            "result_to_reference": result_to_reference,
            "reference_to_result": reference_to_result,
            "symmetric_mean": (result_to_reference + reference_to_result) / 2,
            "result_point_count": len(result_points),
            "reference_point_count": len(reference_points),
            "evaluation_fingerprint": definition.fingerprint,
            "result_selector_id": definition.result.selector_id,
            "reference_selector_id": definition.reference.selector_id,
        },
    )
