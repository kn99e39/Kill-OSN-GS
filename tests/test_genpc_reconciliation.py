from __future__ import annotations

from dataclasses import replace
import unittest

from structural_experiment.experiment1_contracts import AssociationDeclaration, verify_only_association_changed
from structural_experiment.experiment1_artifacts import verify_experiment1_manifest, write_experiment1_manifest
from structural_experiment.genpc_contracts import GenPCExperiment1Control, GenPCReconciliationSpec
from test_experiment1_contracts import controls


class GenPCReconciliationContractTests(unittest.TestCase):
    def test_r_fingerprint_freezes_every_genpc_downstream_control(self) -> None:
        spec = GenPCReconciliationSpec(downstream_implementation_fingerprint="a" * 64)
        self.assertEqual(spec.distance_threshold, 0.0001)
        self.assertEqual(spec.fps_requested_count, 20000)
        self.assertEqual(spec.noise_std_ratio, 2.5)
        self.assertEqual(len(spec.fingerprint), 64)
        self.assertNotEqual(spec.fingerprint, replace(spec, distance_threshold=0.0002).fingerprint)
        self.assertNotEqual(spec.fingerprint, replace(spec, fps_requested_count=19999).fingerprint)
        self.assertNotEqual(spec.fingerprint, replace(spec, noise_std_ratio=3.0).fingerprint)

    def test_changing_r_is_rejected_by_the_shared_a_only_invariant(self) -> None:
        frozen = GenPCExperiment1Control(
            base=controls(),
            reconciliation=GenPCReconciliationSpec(downstream_implementation_fingerprint="b" * 64),
        )
        changed = replace(frozen, reconciliation=replace(frozen.reconciliation, distance_threshold=0.0002))
        report = verify_only_association_changed(frozen, changed)
        self.assertFalse(report.preserved)
        self.assertIn("reconciliation", report.differences)

    def test_genpc_wrapper_uses_existing_manifest_infrastructure(self) -> None:
        frozen = GenPCExperiment1Control(
            base=controls(),
            reconciliation=GenPCReconciliationSpec(downstream_implementation_fingerprint="c" * 64),
        )
        association = AssociationDeclaration("A-native", (("O:one", "G:one"),))
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = write_experiment1_manifest(path, controls=frozen, association=association)
            self.assertEqual(payload["schema"], "structural-experiment/genpc-experiment1-manifest/v1")
            verified = verify_experiment1_manifest(path, controls=frozen)
            self.assertEqual(verified["control_plane"]["frozen_fingerprints"], frozen.frozen_fingerprints())


if __name__ == "__main__":
    unittest.main()
