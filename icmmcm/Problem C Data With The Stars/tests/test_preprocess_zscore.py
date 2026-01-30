"""Unit tests for add_zscore_judge (S_t within week)."""

import unittest
import numpy as np
import pandas as pd
from src.preprocess import add_zscore_judge


class TestAddZscoreJudge(unittest.TestCase):
    def test_add_zscore_judge_single_week(self):
        """S is z-score among active; mean 0 and std 1 over active."""
        cw = pd.DataFrame({
            "season": [1, 1, 1],
            "week": [1, 1, 1],
            "score_week_total": [10.0, 20.0, 30.0],
            "active": [1, 1, 1],
        })
        out = add_zscore_judge(cw)
        self.assertIn("S", out.columns)
        S = out["S"].values
        mean_S_active = np.mean(S)
        std_S_active = np.std(S)
        self.assertTrue(np.isclose(mean_S_active, 0.0))
        self.assertTrue(np.isclose(std_S_active, 1.0))

    def test_add_zscore_judge_inactive_zero(self):
        """Inactive rows get S=0."""
        cw = pd.DataFrame({
            "season": [1, 1, 1],
            "week": [1, 1, 1],
            "score_week_total": [10.0, 20.0, 0.0],
            "active": [1, 1, 0],
        })
        out = add_zscore_judge(cw)
        S = out["S"].values
        self.assertEqual(S[2], 0.0)


if __name__ == "__main__":
    unittest.main()
