from __future__ import annotations

import unittest

import numpy as np

from structural_experiment.removal_harm_measurement import coverage_harm, numerical_policy
from structural_experiment.removal_harm_fixtures import removal_harm_negative_controls


class RemovalHarmMeasurementTests(unittest.TestCase):
    def test_empty_union_preserves_every_generated_point(self) -> None:
        observed = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        generated = np.asarray([[0.001, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        result = coverage_harm(observed, generated, np.zeros(2, dtype=bool), generated.copy())
        self.assertTrue(np.array_equal(result["U_all"], np.concatenate((observed, generated))))
        self.assertTrue(np.array_equal(result["U_native"], result["U_all"]))

    def test_native_union_is_subset_and_delta_is_nonnegative(self) -> None:
        observed = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        generated = np.asarray([[0.001, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        result = coverage_harm(observed, generated, np.asarray([True, False]), np.asarray([[0.001, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        self.assertTrue(np.all(result["delta"] >= -result["numerical_policy"]["tolerance"]))
        self.assertEqual(len(result["U_native"]), len(result["U_all"]) - 1)
        for point in result["U_native"]:
            self.assertTrue(np.any(np.all(result["U_all"] == point, axis=1)))
        self.assertGreater(result["summary"]["strictly_positive_delta_count"], 0)

    def test_negative_controls_expose_empty_and_native_associations(self) -> None:
        single, noncompeting = removal_harm_negative_controls()
        for fixture in (single, noncompeting):
            self.assertEqual(fixture.associations["A_empty"].links, ())
            self.assertTrue(fixture.associations["A_native"].links)
            self.assertEqual(fixture.protocol["tau"], 0.01)
        self.assertIn("single_continuation:G_correct", single.generated_regions)
        self.assertIn("noncompeting_distinct_surface:G_distinct", noncompeting.generated_regions)

    def test_positive_control_provenance_contains_removed_generated_point(self) -> None:
        observed = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        generated = np.asarray([[0.001, 0.0, 0.0]], dtype=np.float64)
        result = coverage_harm(observed, generated, np.asarray([True]), np.asarray([[0.001, 0.0, 0.0]]))
        row = result["harmed_gt_rows"][0]
        self.assertEqual(row["nearest_point_provenance_in_U_all"], {"kind": "G", "index": 0})
        self.assertTrue(row["nearest_point_is_removed_G"])
        self.assertEqual(row["removed_G_index"], 0)

    def test_numerical_policy_is_machine_precision_and_scene_scale_only(self) -> None:
        policy = numerical_policy(np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64))
        expected = 64.0 * np.finfo(np.float64).eps * 2.0
        self.assertEqual(policy["tolerance"], expected)


if __name__ == "__main__":
    unittest.main()
