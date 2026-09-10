from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from structural_experiment.experiment1_contracts import (
    ContractError,
    EvaluationSelector,
    FixedEvaluationDefinition,
    RegionHandle,
    RegionPartition,
)
from structural_experiment.fixed_evaluation import evaluate_fixed_definition
from structural_experiment.geometry_io import lineage_from_file
from structural_experiment.real_data import PartitionedRawArtifact


def write_triangle(path: Path, *, translated: bool = False) -> None:
    x = "2" if translated else "1"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        f"0 0 0\n{x} 0 0\n0 1 0\n3 0 1 2\n",
        encoding="ascii",
    )


def partition_for(path: Path, artifact_id: str, region_id: str) -> RegionPartition:
    return RegionPartition(
        partition_id=f"{artifact_id}-partition-v1",
        raw_artifact=lineage_from_file(path, artifact_id=artifact_id, source_locator=f"benchmark://{artifact_id}"),
        source="benchmark_gt",
        creation_protocol_id="human-annotated-v3",
        regions=(RegionHandle(region_id=region_id, vertex_indices=(0, 1, 2), face_indices=(0,)),),
    )


class RealDataSubstrateTests(unittest.TestCase):
    def test_materialisation_preserves_raw_lineage_and_ownership(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "observed.ply"
            write_triangle(source)
            partition = partition_for(source, "observed-raw", "O1")
            artifact = PartitionedRawArtifact.from_file(source, partition)
            geometry = artifact.materialize_geometry(
                geometry_id="scene:O", surface_prefix="O", units="metre", tags=("observed",)
            )
            surface = geometry.surfaces[0]
            self.assertEqual(surface.surface_id, "O:O1")
            self.assertEqual(surface.faces, ((0, 1, 2),))
            self.assertEqual(surface.metadata["raw_vertex_indices"], (0, 1, 2))
            self.assertEqual(surface.metadata["raw_face_indices"], (0,))
            self.assertEqual(surface.metadata["raw_file_sha256"], partition.raw_artifact.original_file_sha256)
            self.assertEqual(geometry.metadata["partition_fingerprint"], partition.fingerprint)

    def test_reexported_or_preprocessed_raw_file_cannot_reuse_partition(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "observed.ply"
            write_triangle(source)
            partition = partition_for(source, "observed-raw", "O1")
            write_triangle(source, translated=True)
            with self.assertRaises(ContractError):
                PartitionedRawArtifact.from_file(source, partition)

    def test_fixed_result_and_gt_selectors_can_use_unrelated_ids_without_association(self) -> None:
        with TemporaryDirectory() as directory:
            result_file, reference_file = Path(directory) / "result.ply", Path(directory) / "gt.ply"
            write_triangle(result_file)
            write_triangle(reference_file)
            result = PartitionedRawArtifact.from_file(result_file, partition_for(result_file, "completed-mesh", "result-side"))
            reference = PartitionedRawArtifact.from_file(reference_file, partition_for(reference_file, "benchmark-gt", "truth-side"))
            definition = FixedEvaluationDefinition(
                evaluation_id="fixed-geometry-v1",
                metric_id="symmetric_mean_nearest_distance",
                result=EvaluationSelector("result-window", "completed-mesh", "world_aabb", world_aabb=(-1, -1, -1, 2, 2, 2)),
                reference=EvaluationSelector("truth-region", "benchmark-gt", "region_handle", region_id="truth-side"),
                protocol_revision="metric-v1",
                preprocessing_fingerprint="a" * 64,
                sampling_fingerprint="b" * 64,
            )
            metric = evaluate_fixed_definition(definition, result_artifact=result, reference_artifact=reference)
            self.assertTrue(metric.available)
            self.assertEqual(metric.value["symmetric_mean"], 0.0)
            self.assertEqual(metric.value["result_selector_id"], "result-window")
            self.assertEqual(metric.value["reference_selector_id"], "truth-region")

    def test_face_owner_must_also_own_each_face_vertex(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "raw.ply"
            write_triangle(source)
            lineage = lineage_from_file(source, artifact_id="raw")
            partition = RegionPartition(
                partition_id="bad-face-ownership",
                raw_artifact=lineage,
                source="manual_annotation",
                creation_protocol_id="protocol",
                regions=(RegionHandle("cut", (0, 1), (0,)),),
            )
            with self.assertRaises(ContractError):
                PartitionedRawArtifact.from_file(source, partition)


if __name__ == "__main__":
    unittest.main()
