from __future__ import annotations

import unittest

import numpy as np

from structural_experiment.detector_control_fixtures import detector_negative_controls
from structural_experiment.detector_validation import FROZEN_DETECTOR_SHA256, frozen_detector_definition, provenance_accounting


class DetectorValidationTests(unittest.TestCase):
    def test_frozen_detector_fingerprint_matches_part_b_source(self) -> None:
        definition = frozen_detector_definition()
        self.assertEqual(definition["source_sha256"], FROZEN_DETECTOR_SHA256)
        self.assertEqual(definition["tau"], 0.01)
        self.assertEqual(definition["candidate_criterion"], "strongest event component_count > 1")

    def test_negative_controls_keep_part_a_grid_conventions(self) -> None:
        single, noncompeting = detector_negative_controls()
        self.assertFalse(single.known_ambiguity)
        self.assertFalse(noncompeting.known_ambiguity)
        self.assertEqual(single.protocol["tau"], 0.01)
        self.assertEqual(noncompeting.protocol["grid_step_ratio"], 0.25)
        self.assertIn("G_distinct", noncompeting.generated_regions)

    def test_provenance_accounting_records_surface_collapse_after_detection(self) -> None:
        interaction = {
            "components": [{"component_id": 0, "generated_indices": [0, 1], "generated_count": 2}],
            "interacting_generated_indices": [0, 1],
            "nearest_observed_indices": [4, 4],
        }
        accounting = provenance_accounting(interaction, {
            "G_correct": np.asarray([0], dtype=np.int64),
            "G_distinct": np.asarray([1], dtype=np.int64),
        })
        self.assertTrue(accounting["component_collapse_across_known_structural_surfaces"])
        self.assertEqual(accounting["local_observed_neighborhoods_receiving_both_correct_and_distinct"], 1)
        self.assertEqual(accounting["components"][0]["known_structural_surface_count"], 2)


if __name__ == "__main__":
    unittest.main()
