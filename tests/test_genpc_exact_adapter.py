from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from structural_experiment.experiment1_contracts import ContractError, sha256_file
from structural_experiment.genpc_exact import load_exact_npz


class GenPCExactAdapterTests(unittest.TestCase):
    def test_adapter_preserves_float64_values_order_and_stable_indices(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.npz"
            points = np.asarray([[3.0, 2.0, 1.0], [0.5, 0.25, 0.125]], dtype=np.float64)
            colors = np.asarray([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=np.float64)
            np.savez_compressed(path, points=points, colors=colors)
            loaded = load_exact_npz(
                path,
                expected_sha256=sha256_file(path),
                expected_point_count=2,
            )
            np.testing.assert_array_equal(loaded.points, points)
            np.testing.assert_array_equal(loaded.colors, colors)
            np.testing.assert_array_equal(loaded.original_indices, np.asarray([0, 1], dtype=np.int64))
            self.assertEqual(loaded.point_count, 2)
            self.assertFalse(loaded.points.flags.writeable)
            self.assertFalse(loaded.colors.flags.writeable)
            self.assertEqual(loaded.raw_artifact_lineage(artifact_id="G").vertex_count, 2)
            self.assertEqual(loaded.content_fingerprints["array_order"], "original NPZ order; no reorder/resample/normalization")

    def test_adapter_rejects_wrong_hash_count_or_layout(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "frozen.npz"
            np.savez_compressed(path, points=np.zeros((2, 3)), colors=np.zeros((2, 3)))
            with self.assertRaises(ContractError):
                load_exact_npz(path, expected_sha256="0" * 64, expected_point_count=2)
            with self.assertRaises(ContractError):
                load_exact_npz(path, expected_sha256=sha256_file(path), expected_point_count=3)
            wrong_layout = root / "wrong.npz"
            np.savez_compressed(wrong_layout, points=np.zeros((2, 3)), colors=np.zeros((2, 3)), normals=np.zeros((2, 3)))
            with self.assertRaises(ContractError):
                load_exact_npz(wrong_layout, expected_sha256=sha256_file(wrong_layout), expected_point_count=2)


if __name__ == "__main__":
    unittest.main()
