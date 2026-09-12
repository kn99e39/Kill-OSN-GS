from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from structural_experiment.experiment1_contracts import (
    AssociationDeclaration,
    RawArtifactLineage,
    RegionHandle,
    RegionPartition,
)
from structural_experiment.genpc_contracts import GenPCReconciliationSpec
from structural_experiment.genpc_fusion import AGatedGenPCFusion, GenPCFusionBackend


def partition(artifact_id: str, region_ids: tuple[str, ...], point_count: int) -> RegionPartition:
    regions = []
    for offset, region_id in enumerate(region_ids):
        indices = tuple(index for index in range(point_count) if index % len(region_ids) == offset)
        regions.append(RegionHandle(region_id, indices))
    return RegionPartition(
        partition_id=f"{artifact_id}-partition",
        raw_artifact=RawArtifactLineage(artifact_id, "a" * 64, "point_cloud", point_count, 0),
        source="manual_annotation",
        regions=tuple(regions),
        creation_protocol_id="test-fixed-v1",
    )


def fake_close(observed: np.ndarray, generated: np.ndarray, threshold: float):
    distances = ((generated[:, None, :] - observed[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1).astype(np.int64)
    nearest_distance = distances[np.arange(len(generated)), nearest]
    return nearest_distance >= threshold, nearest, nearest_distance


def fake_fps(points: np.ndarray, requested_count: int) -> np.ndarray:
    return np.arange(min(len(points), requested_count), dtype=np.int64)


def no_noise(points: np.ndarray, colors: np.ndarray, std_ratio: float):
    return points.copy(), colors.copy(), np.arange(len(points), dtype=np.int64)


def operator() -> AGatedGenPCFusion:
    observed = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64)
    generated = np.asarray([[0.005, 0.0, 0.0], [10.005, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=np.float64)
    colors_o = np.ones_like(observed)
    colors_g = np.ones_like(generated) * 0.5
    return AGatedGenPCFusion(
        observed_points=observed,
        observed_colors=colors_o,
        generated_points=generated,
        generated_colors=colors_g,
        observed_partition=partition("O", ("O:left", "O:right"), 3),
        generated_partition=partition("G", ("G:left", "G:right"), 3),
        reconciliation=GenPCReconciliationSpec(downstream_implementation_fingerprint="d" * 64),
        backend=GenPCFusionBackend(fake_close, fake_fps, no_noise),
    )


class GenPCFusionTests(unittest.TestCase):
    def test_association_only_changes_eligibility_and_unmatched_g_is_preserved(self) -> None:
        native = operator().run(
            AssociationDeclaration("native", (("O:left", "G:left"), ("O:right", "G:right")))
        )
        wrong = operator().run(AssociationDeclaration("wrong", (("O:left", "G:left"),)))
        np.testing.assert_array_equal(native.removed_g_indices, np.asarray([0, 1]))
        np.testing.assert_array_equal(native.retained_g_indices, np.asarray([2]))
        np.testing.assert_array_equal(wrong.removed_g_indices, np.asarray([0]))
        # G:right is not eligible under wrong A, so its close geometric point
        # remains.  The unassociated G point also remains.
        np.testing.assert_array_equal(wrong.retained_g_indices, np.asarray([1, 2]))
        self.assertEqual(native.pre_fps_union_count, 4)
        self.assertEqual(wrong.pre_fps_union_count, 5)
        self.assertEqual(native.final_count, 4)

    def test_many_to_many_is_accepted_and_native_all_pairs_is_reproducible(self) -> None:
        result = operator().run(
            AssociationDeclaration(
                "many-to-many",
                (("O:left", "G:left"), ("O:right", "G:left"), ("O:left", "G:right"), ("O:right", "G:right")),
            )
        )
        np.testing.assert_array_equal(result.removed_g_indices, np.asarray([0, 1]))
        self.assertEqual(result.eligible_generated_point_count, 3)
        self.assertEqual(result.pair_eligibility_comparison_count, 6)
        self.assertEqual(len(result.removal_attribution), 2)

    def test_invalid_association_region_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            operator().run(AssociationDeclaration("invalid", (("O:missing", "G:left"),)))


if __name__ == "__main__":
    unittest.main()
