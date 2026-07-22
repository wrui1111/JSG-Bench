from __future__ import annotations

import unittest

import numpy as np

from experiments.training_free_pipeline import (
    LOWLEVEL_FEATURE_NAMES,
    LOWLEVEL_FEATURE_VERSION,
    bbox_from_blocks,
    candidate_bboxes_from_scores,
    lowlevel_block_features,
)


class Pixel5dFeatureContractTest(unittest.TestCase):
    def test_feature_contract_has_five_unique_dimensions(self) -> None:
        self.assertEqual(LOWLEVEL_FEATURE_VERSION, "pixel5d-v2")
        self.assertEqual(len(LOWLEVEL_FEATURE_NAMES), 5)
        self.assertEqual(len(set(LOWLEVEL_FEATURE_NAMES)), 5)
        self.assertNotIn("rgb_mean", LOWLEVEL_FEATURE_NAMES)

    def test_block_feature_vector_matches_declared_order(self) -> None:
        patch = np.asarray(
            [
                [[0.0, 3.0, 6.0], [9.0, 12.0, 15.0]],
                [[18.0, 21.0, 24.0], [27.0, 30.0, 33.0]],
            ],
            dtype=np.float32,
        )
        gray = patch.mean(axis=2)
        grad = np.full_like(gray, 2.0)
        lap = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

        values = lowlevel_block_features(patch, gray, grad, lap)

        self.assertEqual(len(values), len(LOWLEVEL_FEATURE_NAMES))
        np.testing.assert_allclose(
            values,
            [patch.std(), gray.mean(), gray.std(), grad.mean(), lap.var()],
        )

    def test_pooled_rgb_mean_equals_unweighted_gray_mean(self) -> None:
        patch = np.arange(3 * 5 * 7, dtype=np.float32).reshape(5, 7, 3)
        self.assertTrue(np.allclose(patch.mean(), patch.mean(axis=2).mean(), atol=1e-5))


class BboxFromBlocksTest(unittest.TestCase):
    def test_last_row_and_column_absorb_remainder(self) -> None:
        bbox = bbox_from_blocks([(15, 15)], 31, 31, 501, 509, 1.0, 1.0)
        self.assertEqual(bbox, [465, 465, 509, 501])

    def test_only_last_column_absorbs_remainder(self) -> None:
        bbox = bbox_from_blocks([(14, 15)], 31, 31, 501, 509, 1.0, 1.0)
        self.assertEqual(bbox, [465, 434, 509, 465])

    def test_interior_block_keeps_nominal_extent(self) -> None:
        bbox = bbox_from_blocks([(14, 14)], 31, 31, 501, 509, 1.0, 1.0)
        self.assertEqual(bbox, [434, 434, 465, 465])

    def test_divisible_dimensions_are_unchanged(self) -> None:
        bbox = bbox_from_blocks([(15, 15)], 32, 32, 512, 512, 1.0, 1.0)
        self.assertEqual(bbox, [480, 480, 512, 512])

    def test_empty_selection_has_no_bbox(self) -> None:
        self.assertIsNone(bbox_from_blocks([], 31, 31, 501, 509, 1.0, 1.0))

    def test_remainder_edge_maps_back_to_original_coordinates(self) -> None:
        bbox = bbox_from_blocks([(15, 15)], 31, 31, 501, 509, 2.0, 3.0)
        self.assertEqual(bbox, [930, 1395, 1018, 1503])

    def test_candidate_neighborhood_includes_remainder_edges(self) -> None:
        indices = [(y, x) for y in range(16) for x in range(16)]
        scores = np.zeros(len(indices), dtype=np.float32)
        scores[indices.index((15, 15))] = 10.0

        candidates = candidate_bboxes_from_scores(
            scores,
            indices,
            31,
            31,
            501,
            509,
            1.0,
            1.0,
            top_k=1,
        )

        self.assertEqual(candidates[0]["bbox"], [434, 434, 509, 501])


if __name__ == "__main__":
    unittest.main()
