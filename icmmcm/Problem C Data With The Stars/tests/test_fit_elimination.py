"""Unit tests for elimination likelihood fit (G1–G4)."""

import unittest
import numpy as np
import pandas as pd
from src.fit.fit_elimination import (
    elimination_log_prob,
    plackett_luce_log_prob,
    build_elimination_events,
    build_finals_events,
    neg_log_likelihood,
)


class TestEliminationLogProb(unittest.TestCase):
    def test_percent_lowest_most_likely(self):
        c = np.array([0.5, 0.3, 0.6])
        tau = 1.0
        log_p_1 = elimination_log_prob("percent", c, 1, tau)
        log_p_0 = elimination_log_prob("percent", c, 0, tau)
        self.assertGreater(log_p_1, log_p_0)

    def test_percent_log_prob_sums_to_zero_in_log_space(self):
        c = np.array([0.4, 0.35, 0.25])
        tau = 1.0
        log_probs = [elimination_log_prob("percent", c, i, tau) for i in range(3)]
        log_sum = np.log(np.sum(np.exp(log_probs)))
        self.assertTrue(np.isclose(log_sum, 0.0))

    def test_rank_highest_most_likely(self):
        R = np.array([3.0, 5.0, 4.0])
        tau = 1.0
        log_p_1 = elimination_log_prob("rank", R, 1, tau)
        log_p_0 = elimination_log_prob("rank", R, 0, tau)
        self.assertGreater(log_p_1, log_p_0)

    def test_rank_log_prob_sums_to_zero(self):
        R = np.array([3.0, 5.0, 4.0])
        tau = 1.0
        log_probs = [elimination_log_prob("rank", R, i, tau) for i in range(3)]
        log_sum = np.log(np.sum(np.exp(log_probs)))
        self.assertTrue(np.isclose(log_sum, 0.0))


class TestPlackettLuce(unittest.TestCase):
    def test_plackett_luce_order(self):
        s = np.array([1.0, 0.5, 0.0])
        order = [0, 1, 2]
        log_p = plackett_luce_log_prob(s, order)
        self.assertLess(log_p, 0.0)
        self.assertTrue(np.isfinite(log_p))

    def test_plackett_luce_higher_strength_first_higher_prob(self):
        s = np.array([1.0, 0.0])
        order_correct = [0, 1]
        order_wrong = [1, 0]
        log_p_correct = plackett_luce_log_prob(s, order_correct)
        log_p_wrong = plackett_luce_log_prob(s, order_wrong)
        self.assertGreater(log_p_correct, log_p_wrong)


class TestBuildEvents(unittest.TestCase):
    def test_build_elimination_events_structure(self):
        cw = pd.DataFrame({
            "season": [1, 1, 1, 1],
            "week": [1, 1, 1, 1],
            "active": [1, 1, 1, 1],
            "elimination_week": [2, 2, 2, 2],
            "score_week_total": [10.0, 20.0, 30.0, 40.0],
            "z_J": [0.25, 0.5, 0.75, 1.0],
            "p_prev": [0, 0, 0, 0],
            "underdog": [0, 0, 0, 0],
            "finals_week": [np.nan] * 4,
        })
        cw.loc[0, "elimination_week"] = 1
        events = build_elimination_events(cw)
        self.assertGreater(len(events), 0)
        ev = events[0]
        self.assertIn("season", ev)
        self.assertIn("rule", ev)
        self.assertIn("J", ev)
        self.assertIn("observed_e_idx", ev)

    def test_build_finals_events_structure(self):
        cw = pd.DataFrame({
            "season": [1, 1, 1],
            "week": [5, 5, 5],
            "active": [1, 1, 1],
            "finals_week": [5.0, 5.0, 5.0],
            "placement": ["1", "2", "3"],
            "score_week_total": [25.0, 24.0, 23.0],
            "z_J": [1.0, 0.96, 0.92],
            "p_prev": [0, 0, 0],
            "underdog": [0, 0, 0],
        })
        events = build_finals_events(cw)
        self.assertGreater(len(events), 0)
        ev = events[0]
        self.assertIn("placement_order", ev)
        self.assertGreaterEqual(len(ev["placement_order"]), 2)


class TestNegLogLikelihood(unittest.TestCase):
    def test_neg_log_likelihood_finite(self):
        elim_events = [{
            "rule": "percent",
            "J": np.array([10.0, 20.0, 30.0]),
            "z_J": np.array([1/3, 2/3, 1.0]),
            "p_prev": np.zeros(3),
            "underdog": np.zeros(3),
            "X": np.ones((3, 1)),
            "observed_e_idx": 0,
        }]
        finals_events = []
        beta = {"beta0": 0.0, "beta_J": 1.0, "beta_P": 0.0, "beta_U": 0.0}
        tau = 1.0
        nll = neg_log_likelihood(beta, tau, elim_events, finals_events)
        self.assertTrue(np.isfinite(nll))
        self.assertGreater(nll, 0.0)


if __name__ == "__main__":
    unittest.main()
