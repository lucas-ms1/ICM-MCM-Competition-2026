"""Unit tests for industry-based communities (small-world layer)."""

import unittest
import pandas as pd
from src.models.communities import (
    industry_communities_from_raw,
    community_weights_from_raw,
    INDUSTRY_COL,
    GENERAL_LABEL,
)


class TestIndustryCommunities(unittest.TestCase):
    """industry_communities_from_raw: G, industry_order, industry_to_g."""

    def test_no_industry_column_returns_general_only(self):
        raw = pd.DataFrame({"season": [1], "celebrity_name": ["x"]})
        G, order, ind_to_g = industry_communities_from_raw(raw, include_general=True)
        self.assertEqual(G, 1)
        self.assertEqual(order, [GENERAL_LABEL])
        self.assertEqual(ind_to_g[GENERAL_LABEL], 0)

    def test_with_industries_include_general(self):
        raw = pd.DataFrame({
            INDUSTRY_COL: ["Actor/Actress", "Athlete", "Actor/Actress", "Model"],
            "season": [1, 1, 2, 1],
        })
        G, order, ind_to_g = industry_communities_from_raw(raw, include_general=True)
        self.assertGreaterEqual(G, 2)
        self.assertIn(GENERAL_LABEL, order)
        self.assertEqual(ind_to_g[GENERAL_LABEL], G - 1)
        self.assertTrue(all(0 <= ind_to_g[k] < G for k in ind_to_g))

    def test_weights_uniform_sums_to_one(self):
        raw = pd.DataFrame({
            INDUSTRY_COL: ["Actor/Actress", "Athlete", "Model"],
            "season": [1, 1, 1],
        })
        G, w_g, _ = community_weights_from_raw(raw, include_general=True, mode="uniform")
        self.assertEqual(len(w_g), G)
        self.assertTrue(all(x >= 0 for x in w_g))
        self.assertAlmostEqual(w_g.sum(), 1.0)

    def test_weights_empirical_sums_to_one(self):
        raw = pd.DataFrame({
            INDUSTRY_COL: ["Actor/Actress", "Athlete", "Actor/Actress"],
            "season": [1, 1, 2],
        })
        G, w_g, _ = community_weights_from_raw(raw, include_general=True, mode="empirical")
        self.assertEqual(len(w_g), G)
        self.assertTrue(all(x >= 0 for x in w_g))
        self.assertAlmostEqual(w_g.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
