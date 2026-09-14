from __future__ import annotations

import unittest

import numpy as np

from structural_experiment.genpc_interaction_audit import native_interaction_diagnostic


class GenPCInteractionAuditTests(unittest.TestCase):
    def test_reports_two_tau_disconnected_generated_patches_at_one_observed_neighborhood(self) -> None:
        observed = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        generated = np.asarray([[0.009, 0.0, 0.0], [0.008, 0.0, 0.0], [-0.009, 0.0, 0.0]], dtype=np.float64)
        result = native_interaction_diagnostic(observed, generated, tau=0.01)
        self.assertEqual(len(result["interacting_generated_indices"]), 3)
        # The two positive-x points connect, while the third is tau-separated
        # from them and still contacts the same observed neighborhood.
        self.assertEqual(len(result["components"]), 2)
        self.assertTrue(result["multiple_component_opportunity"])
        self.assertEqual(result["strongest_event"]["component_count"], 2)


if __name__ == "__main__":
    unittest.main()
