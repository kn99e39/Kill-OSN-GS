from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from structural_experiment.artifacts import compare_serialized_outputs, read_frozen_scene, write_frozen_scene
from structural_experiment.contracts import Association, AssociationEntry, ReconciliationConfiguration
from structural_experiment.fixtures import continuation_fixture, unmatched_fixture
from structural_experiment.reconcile import ReferenceOverlayReconciler
from structural_experiment.runner import ExperimentHarness, verify_only_association_changes


class ExperimentalHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = continuation_fixture()
        self.harness = ExperimentHarness(ReferenceOverlayReconciler())
        self.configuration = ReconciliationConfiguration()
        self.oracle = Association("manual-associated", (AssociationEntry("associated", "obs-visible", "gen-continuation"),), "manual")
        self.unmatched = Association("manual-unmatched", (
            AssociationEntry("unmatched_observation", "obs-visible", None),
            AssociationEntry("unmatched_generated", None, "gen-continuation"),
        ), "manual")

    def test_domain_contract_serialization_and_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            write_frozen_scene(path, self.scene)
            loaded = read_frozen_scene(path)
        self.assertEqual(self.scene.fingerprint, loaded.fingerprint)
        self.assertEqual(self.scene.observation.geometry.coordinate_system, "right_handed_xyz")

    def test_fixtures_are_deterministic_with_stable_ids(self) -> None:
        self.assertEqual(continuation_fixture().fingerprint, continuation_fixture().fingerprint)
        self.assertEqual(unmatched_fixture().hypothesis.geometry.surface_ids, ("gen-detached",))
        self.assertEqual(self.scene.evaluation_reference.regions[1].surface_ids, ("gen-continuation",))

    def test_association_is_injected_and_not_inferred(self) -> None:
        result = self.harness.run(self.scene, self.unmatched, self.configuration).result
        self.assertEqual(result.association.fingerprint, self.unmatched.fingerprint)
        self.assertFalse(result.notes["association_inferred"])
        self.assertEqual(result.geometry.surfaces, self.scene.observation.geometry.surfaces + self.scene.hypothesis.geometry.surfaces)

    def test_evaluation_regions_and_metric_accounting(self) -> None:
        completed = self.harness.run(self.scene, self.oracle, self.configuration)
        hidden = {metric.metric_id: metric for metric in completed.metrics["hidden_region"]}
        self.assertEqual(hidden["symmetric_mean_nearest_distance"].value["symmetric_mean"], 0.0)
        self.assertFalse(hidden["seam_error"].available)
        association = {metric.metric_id: metric for metric in completed.metrics["association"]}
        self.assertTrue(association["association_correctness"].value["exact_match"])

    def test_manifest_review_and_replay_are_deterministic(self) -> None:
        first = self.harness.run(self.scene, self.oracle, self.configuration)
        replay = self.harness.run(self.scene, self.oracle, self.configuration)
        self.assertEqual(first.manifest(), replay.manifest())
        with tempfile.TemporaryDirectory() as directory:
            location = self.harness.persist(first, directory)
            self.assertTrue((location / "manifest.json").is_file())
            self.assertTrue((location / "review.json").is_file())
            equivalence = compare_serialized_outputs(location / "manifest.json", location / "manifest.json")
            self.assertTrue(equivalence.equivalent)

    def test_control_invariant_allows_only_association_to_change(self) -> None:
        first = self.harness.run(self.scene, self.oracle, self.configuration)
        second = self.harness.run(self.scene, self.unmatched, self.configuration)
        report = verify_only_association_changes((first, second))
        self.assertTrue(report.preserved, report.differences)
        self.assertNotEqual(first.association.fingerprint, second.association.fingerprint)


if __name__ == "__main__":
    unittest.main()
