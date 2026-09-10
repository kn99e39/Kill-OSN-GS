from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

from structural_experiment.experiment1_contracts import (
    AssociationDeclaration,
    ContractError,
    EvaluationSelector,
    Experiment1ControlPlane,
    FixedEvaluationDefinition,
    RawArtifactLineage,
    RegionHandle,
    RegionPartition,
    verify_only_association_changed,
)
from structural_experiment.geometry_io import decode_geometry, lineage_from_file


HASH = "a" * 64


def raw(artifact_id: str, byte: str = "a") -> RawArtifactLineage:
    return RawArtifactLineage(
        artifact_id=artifact_id,
        original_file_sha256=byte * 64,
        representation="mesh",
        vertex_count=8,
        face_count=4,
        source_locator="benchmark://frozen-scene",
        source_revision="scene-v1",
    )


def partition(artifact_id: str, partition_id: str, *, label: str = "region") -> RegionPartition:
    return RegionPartition(
        partition_id=partition_id,
        raw_artifact=raw(artifact_id),
        source="benchmark_gt",
        creation_protocol_id="fixed-partition-v1",
        regions=(
            RegionHandle(region_id=f"{artifact_id}:one", vertex_indices=(0, 1, 2), face_indices=(0,), label=label),
            RegionHandle(region_id=f"{artifact_id}:two", vertex_indices=(3, 4, 5), face_indices=(1,)),
        ),
    )


def controls() -> Experiment1ControlPlane:
    observed = partition("O", "O-partition")
    generated = partition("G", "G-partition")
    reference = partition("R", "R-partition")
    evaluation = FixedEvaluationDefinition(
        evaluation_id="frozen-eval",
        metric_id="symmetric_chamfer",
        result=EvaluationSelector("result-region", "result_mesh", "world_aabb", world_aabb=(0, 0, 0, 1, 1, 1)),
        reference=EvaluationSelector("reference-region", "R", "region_handle", region_id="R:one"),
        protocol_revision="eval-v1",
        preprocessing_fingerprint=HASH,
        sampling_fingerprint="b" * 64,
    )
    return Experiment1ControlPlane(
        observed_partition=observed,
        generated_partition=generated,
        reference_partition=reference,
        preprocessing_fingerprint=HASH,
        sampling_fingerprint="b" * 64,
        renderer_fingerprint="c" * 64,
        evaluation=evaluation,
        reference_model_fingerprint="d" * 64,
        seed=17,
    )


class Experiment1ContractsTests(unittest.TestCase):
    def test_association_is_minimal_many_to_many_and_does_not_touch_controls(self) -> None:
        frozen = controls()
        baseline_a = AssociationDeclaration("A-baseline", (("O:one", "G:one"),))
        alternate_a = AssociationDeclaration(
            "A-alternate",
            (("O:one", "G:one"), ("O:one", "G:two"), ("O:two", "G:one")),
        )
        self.assertNotEqual(baseline_a.fingerprint, alternate_a.fingerprint)
        self.assertTrue(verify_only_association_changed(frozen, frozen).preserved)
        self.assertNotEqual(frozen.manifest(baseline_a)["association_fingerprint"], frozen.manifest(alternate_a)["association_fingerprint"])

    def test_every_frozen_control_change_is_rejected(self) -> None:
        baseline = controls()
        variants = {
            "O_raw": replace(baseline, observed_partition=replace(baseline.observed_partition, raw_artifact=raw("O", "e"))),
            "G_raw": replace(baseline, generated_partition=replace(baseline.generated_partition, raw_artifact=raw("G", "e"))),
            "R_raw": replace(baseline, reference_partition=replace(baseline.reference_partition, raw_artifact=raw("R", "e"))),
            "preprocessing": replace(baseline, preprocessing_fingerprint="e" * 64),
            "sampling": replace(baseline, sampling_fingerprint="e" * 64),
            "renderer": replace(baseline, renderer_fingerprint="e" * 64),
            "evaluation": replace(baseline, evaluation=replace(baseline.evaluation, metric_id="point_to_plane")),
            "O_partition": replace(baseline, observed_partition=replace(baseline.observed_partition, version="2")),
        }
        for changed_key, candidate in variants.items():
            with self.subTest(changed_key=changed_key):
                report = verify_only_association_changed(baseline, candidate)
                self.assertFalse(report.preserved)
                self.assertIn(changed_key, report.differences)

    def test_partitions_are_external_and_cannot_overlap_or_escape_raw_ownership(self) -> None:
        source = raw("O")
        with self.assertRaises(ContractError):
            RegionPartition(
                "invalid-overlap", source, "manual_annotation", (
                    RegionHandle("one", (0, 1)), RegionHandle("two", (1, 2)),
                ), "fixed",
            )
        with self.assertRaises(ContractError):
            RegionPartition(
                "invalid-bound", source, "manual_annotation", (RegionHandle("one", (8,)),), "fixed",
            )

    def test_evaluation_selectors_do_not_require_matching_region_ids(self) -> None:
        definition = controls().evaluation
        self.assertNotEqual(definition.result.artifact_id, definition.reference.artifact_id)
        self.assertNotEqual(definition.result.selector_id, definition.reference.selector_id)
        self.assertEqual(definition.reference.region_id, "R:one")

    def test_ascii_and_binary_ply_lineage_preserves_raw_hash_and_index_bounds(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ascii_ply = root / "triangle-ascii.ply"
            ascii_ply.write_text(
                "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
                encoding="ascii",
            )
            binary_ply = root / "point-binary.ply"
            binary_ply.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
                + struct.pack("<fff", 1.0, 2.0, 3.0)
            )
            self.assertEqual(decode_geometry(ascii_ply).face_count, 1)
            self.assertEqual(decode_geometry(binary_ply).positions, ((1.0, 2.0, 3.0),))
            lineage = lineage_from_file(ascii_ply, artifact_id="raw-triangle")
            self.assertEqual(lineage.vertex_count, 3)
            self.assertEqual(lineage.face_count, 1)
            self.assertEqual(len(lineage.original_file_sha256), 64)

    def test_glb_mesh_decoding_keeps_raw_vertex_and_face_ownership_addressable(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.glb"
            binary = struct.pack("<fffffffffHHH", 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2)
            document = {
                "asset": {"version": "2.0"},
                "buffers": [{"byteLength": len(binary)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}, {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
                "accessors": [
                    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                    {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
                ],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            }
            encoded_json = json.dumps(document, separators=(",", ":")).encode("utf-8")
            encoded_json += b" " * ((-len(encoded_json)) % 4)
            padded_binary = binary + b"\x00" * ((-len(binary)) % 4)
            payload = (
                struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded_json) + 8 + len(padded_binary))
                + struct.pack("<I4s", len(encoded_json), b"JSON") + encoded_json
                + struct.pack("<I4s", len(padded_binary), b"BIN\x00") + padded_binary
            )
            path.write_bytes(payload)
            geometry = decode_geometry(path)
            self.assertEqual((geometry.vertex_count, geometry.face_count, geometry.faces), (3, 1, ((0, 1, 2),)))


if __name__ == "__main__":
    unittest.main()
