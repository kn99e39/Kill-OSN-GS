"""Explicit negative controls required before any Experiment 1 data run."""

from __future__ import annotations

from dataclasses import replace
import unittest

from structural_experiment.experiment1_contracts import AssociationDeclaration, verify_only_association_changed
from test_experiment1_contracts import controls, partition, raw


class RequiredExperiment1InvariantTests(unittest.TestCase):
    def test_changing_a_is_the_only_permitted_variation(self) -> None:
        frozen = controls()
        a_one = AssociationDeclaration("A-one", (("O:one", "G:one"),))
        a_two = AssociationDeclaration("A-two", (("O:one", "G:one"), ("O:two", "G:two")))
        self.assertNotEqual(a_one.fingerprint, a_two.fingerprint)
        self.assertTrue(verify_only_association_changed(frozen, frozen).preserved)

    def test_each_required_frozen_input_is_rejected_when_changed(self) -> None:
        baseline = controls()
        changes = {
            "O_raw": replace(baseline, observed_partition=replace(baseline.observed_partition, raw_artifact=raw("O", "e"))),
            "G_raw": replace(baseline, generated_partition=replace(baseline.generated_partition, raw_artifact=raw("G", "e"))),
            "R_raw": replace(baseline, reference_partition=replace(baseline.reference_partition, raw_artifact=raw("R", "e"))),
            "preprocessing": replace(baseline, preprocessing_fingerprint="e" * 64),
            "sampling": replace(baseline, sampling_fingerprint="e" * 64),
            "renderer": replace(baseline, renderer_fingerprint="e" * 64),
            "evaluation": replace(baseline, evaluation=replace(baseline.evaluation, protocol_revision="eval-v2")),
            "O_partition": replace(baseline, observed_partition=partition("O", "O-partition-v2", label="updated")),
            "G_partition": replace(baseline, generated_partition=partition("G", "G-partition-v2", label="updated")),
            "R_partition": replace(baseline, reference_partition=partition("R", "R-partition-v2", label="updated")),
        }
        for name, candidate in changes.items():
            with self.subTest(name=name):
                report = verify_only_association_changed(baseline, candidate)
                self.assertFalse(report.preserved)
                self.assertIn(name, report.differences)


if __name__ == "__main__":
    unittest.main()
