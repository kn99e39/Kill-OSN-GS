from __future__ import annotations

import unittest

import numpy as np

from structural_experiment.controlled_fixtures import TAU, controlled_fixtures


class ControlledFixtureTests(unittest.TestCase):
    def test_known_identity_fixtures_have_fixed_tau_derived_contracts(self) -> None:
        parallel, junction = controlled_fixtures()
        self.assertEqual(parallel.protocol["tau"], TAU)
        self.assertEqual(junction.protocol["tau"], TAU)
        for fixture in (parallel, junction):
            self.assertEqual(set(fixture.associations), {"A_native", "A_wrong", "A_oracle"})
            self.assertEqual(sum(len(region.vertex_indices) for region in fixture.observed_partition.regions), len(fixture.observed_points))
            self.assertEqual(sum(len(region.vertex_indices) for region in fixture.generated_partition.regions), len(fixture.generated_points))
            correct, distinct = fixture.generated_regions.values()
            self.assertFalse(set(correct).intersection(distinct))
            self.assertTrue(np.any(np.abs(fixture.generated_points[distinct, 2] - fixture.observed_points[0, 2]) < TAU))

    def test_wrong_and_oracle_are_distinct_only_by_association(self) -> None:
        for fixture in controlled_fixtures():
            self.assertNotEqual(fixture.associations["A_wrong"].fingerprint, fixture.associations["A_oracle"].fingerprint)
            self.assertEqual(len(fixture.associations["A_native"].links), 2)
            self.assertEqual(len(fixture.associations["A_wrong"].links), 1)
            self.assertEqual(len(fixture.associations["A_oracle"].links), 1)


if __name__ == "__main__":
    unittest.main()
