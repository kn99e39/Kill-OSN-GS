from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from structural_experiment.experiment1_artifacts import verify_experiment1_manifest, write_experiment1_manifest
from structural_experiment.experiment1_contracts import AssociationDeclaration
from test_experiment1_contracts import controls


class Experiment1ManifestTests(unittest.TestCase):
    def test_manifest_embeds_immutable_controls_and_varies_only_with_a(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment1.json"
            frozen = controls()
            first = AssociationDeclaration("A-one", (("O:one", "G:one"),))
            second = AssociationDeclaration("A-two", (("O:one", "G:two"),))
            one = write_experiment1_manifest(path, controls=frozen, association=first, artifact_locations={"O": "raw/O.ply"})
            verified = verify_experiment1_manifest(path, controls=frozen)
            two = write_experiment1_manifest(path, controls=frozen, association=second, artifact_locations={"O": "raw/O.ply"})
            self.assertEqual(verified["control_plane"]["frozen_fingerprints"], frozen.frozen_fingerprints())
            self.assertNotEqual(one["association_fingerprint"], two["association_fingerprint"])
            self.assertEqual(one["control_plane"]["frozen_fingerprints"], two["control_plane"]["frozen_fingerprints"])

    def test_manifest_rejects_a_different_control_plane(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment1.json"
            frozen = controls()
            write_experiment1_manifest(path, controls=frozen, association=AssociationDeclaration("A", ()))
            from dataclasses import replace
            changed = replace(frozen, renderer_fingerprint="f" * 64)
            with self.assertRaises(ValueError):
                verify_experiment1_manifest(path, controls=changed)


if __name__ == "__main__":
    unittest.main()
