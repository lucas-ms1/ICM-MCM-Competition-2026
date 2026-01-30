"""Unit tests for biased mean-field voter model (Layer 2)."""

import unittest
import numpy as np
from src.models.vote_latent import shares_from_preference
from src.models.meanfield import (
    underdog_score,
    switching_target,
    meanfield_update,
    watts_strogatz_adjacency,
    influence_matrix_from_adjacency,
    meanfield_update_networked,
    meanfield_update_smallworld,
    make_kernel,
    kernel_weighted_history,
    update_memory_state,
)


class TestSoftmax(unittest.TestCase):
    """shares_from_preference (softmax) output sums to 1."""

    def test_softmax_sums_to_one(self):
        u = np.array([0.0, 1.0, -0.5, 2.0])
        f = shares_from_preference(u)
        self.assertEqual(f.shape, u.shape)
        self.assertTrue(np.all(f >= 0))
        self.assertTrue(np.isclose(f.sum(), 1.0))


class TestMeanfieldUpdate(unittest.TestCase):
    """meanfield_update returns a distribution summing to 1."""

    def test_meanfield_update_preserves_sum_to_one(self):
        n = 4
        p_t = np.ones(n) / n
        S_t = np.array([0.5, -0.3, 0.0, 0.2])
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10}
        p_next = meanfield_update(p_t, S_t, None, params)
        self.assertEqual(p_next.shape, (n,))
        self.assertTrue(np.all(p_next >= 0))
        self.assertTrue(np.isclose(p_next.sum(), 1.0))

    def test_meanfield_update_with_underdog_sums_to_one(self):
        n = 3
        p_t = np.array([0.5, 0.3, 0.2])
        S_t = np.array([-0.5, 0.0, 0.5])
        underdog = underdog_score(S_t, p_t, mode="smooth", tau_S=0.0, tau_p=0.25)
        params = {"kappa": 0.3, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10}
        p_next = meanfield_update(p_t, S_t, None, params, underdog=underdog, beta_U=0.5)
        self.assertTrue(np.all(p_next >= 0))
        self.assertTrue(np.isclose(p_next.sum(), 1.0))

    def test_meanfield_update_with_X(self):
        n = 3
        p_t = np.ones(n) / n
        S_t = np.zeros(n)
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.0, "epsilon": 1e-10, "theta": np.array([0.0, 0.1])}
        p_next = meanfield_update(p_t, S_t, X, params)
        self.assertTrue(np.isclose(p_next.sum(), 1.0))


class TestUnderdog(unittest.TestCase):
    """Underdog scores in [0, 1]; smooth and indicator modes."""

    def test_underdog_smooth_in_zero_one(self):
        S_t = np.array([-1.0, 0.0, 1.0])
        p_t = np.array([0.2, 0.5, 0.3])
        out = underdog_score(S_t, p_t, mode="smooth", tau_S=0.0, tau_p=0.25)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 1))

    def test_underdog_indicator_zero_one(self):
        S_t = np.array([-1.0, 0.0, 1.0])
        p_t = np.array([0.1, 0.6, 0.3])
        out = underdog_score(S_t, p_t, mode="indicator", tau_S=0.0, tau_p=0.5)
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all((out == 0) | (out == 1)))

    def test_underdog_score_wrong_length_raises(self):
        with self.assertRaises(ValueError) as ctx:
            underdog_score(np.array([0.0, 1.0]), np.array([0.5, 0.5, 0.0]))
        self.assertIn("same length", str(ctx.exception))

    def test_underdog_score_invalid_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            underdog_score(np.array([0.0]), np.array([1.0]), mode="invalid")
        self.assertIn("smooth", str(ctx.exception))


class TestSwitchingTarget(unittest.TestCase):
    """switching_target returns a distribution summing to 1."""

    def test_switching_target_sums_to_one(self):
        p_t = np.array([0.4, 0.35, 0.25])
        S_t = np.array([0.5, -0.2, 0.0])
        q = switching_target(p_t, S_t, rho=0.6, eta=1.0)
        self.assertEqual(q.shape, p_t.shape)
        self.assertTrue(np.all(q >= 0))
        self.assertTrue(np.isclose(q.sum(), 1.0))


class TestWattsStrogatz(unittest.TestCase):
    """Watts–Strogatz adjacency: shape, row-stochastic, connectivity."""

    def test_watts_strogatz_shape(self):
        rng = np.random.default_rng(42)
        A = watts_strogatz_adjacency(G=10, K=2, beta=0.1, rng=rng)
        self.assertEqual(A.shape, (10, 10))

    def test_watts_strogatz_row_stochastic(self):
        rng = np.random.default_rng(42)
        A = watts_strogatz_adjacency(G=8, K=2, beta=0.2, row_stochastic=True, rng=rng)
        self.assertTrue(np.allclose(A.sum(axis=1), np.ones(8)))

    def test_watts_strogatz_single_node(self):
        A = watts_strogatz_adjacency(G=1, K=0, beta=0.0)
        self.assertEqual(A.shape, (1, 1))
        self.assertTrue(np.isclose(A[0, 0], 1.0))


class TestMeanfieldUpdateNetworked(unittest.TestCase):
    """meanfield_update_networked: simplex per community, aggregate sums to 1."""

    def test_networked_sums_to_one_per_community(self):
        G, n = 3, 4
        rng = np.random.default_rng(43)
        p_t = np.zeros((G, n))
        for g in range(G):
            x = rng.random(n)
            p_t[g] = x / x.sum()
        S_t = np.array([0.1, -0.2, 0.0, 0.3])
        A = watts_strogatz_adjacency(G=G, K=1, beta=0.0, rng=rng)
        w = np.ones(G) / G
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10}
        p_next, bar_p = meanfield_update_networked(
            p_t, S_t, None, params, A, w,
        )
        self.assertEqual(p_next.shape, (G, n))
        for g in range(G):
            self.assertTrue(np.all(p_next[g] >= 0))
            self.assertTrue(np.isclose(p_next[g].sum(), 1.0))

    def test_networked_aggregate_sums_to_one(self):
        G, n = 2, 3
        p_t = np.ones((G, n)) / n
        S_t = np.zeros(n)
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        w = np.array([0.6, 0.4])
        params = {"kappa": 0.3, "eta": 1.0, "gamma": 0.0, "epsilon": 1e-10}
        p_next, bar_p = meanfield_update_networked(p_t, S_t, None, params, A, w)
        self.assertEqual(bar_p.shape, (n,))
        self.assertTrue(np.all(bar_p >= 0))
        self.assertTrue(np.isclose(bar_p.sum(), 1.0))

    def test_networked_aggregate_weights(self):
        G, n = 2, 3
        p_t = np.ones((G, n)) / n
        p_t[0] = np.array([1.0, 0.0, 0.0])
        p_t[1] = np.array([0.0, 1.0, 0.0])
        S_t = np.zeros(n)
        A = np.eye(2)
        w = np.array([0.7, 0.3])
        params = {"kappa": 0.0, "eta": 0.0, "gamma": 0.0, "epsilon": 1e-10}
        p_next, bar_p = meanfield_update_networked(p_t, S_t, None, params, A, w)
        self.assertTrue(np.allclose(bar_p, w[0] * p_t[0] + w[1] * p_t[1]))


class TestKernel(unittest.TestCase):
    """make_kernel and kernel_weighted_history."""

    def test_kernel_exponential(self):
        k = make_kernel("exponential", decay=0.5)
        self.assertEqual(k(0), 1.0)
        self.assertEqual(k(1), 0.5)
        self.assertEqual(k(2), 0.25)

    def test_kernel_rectangular(self):
        k = make_kernel("rectangular", L=3)
        self.assertAlmostEqual(k(0), 1.0 / 3)
        self.assertAlmostEqual(k(1), 1.0 / 3)
        self.assertAlmostEqual(k(3), 1.0 / 3)
        self.assertEqual(k(4), 0.0)

    def test_kernel_power_law(self):
        k = make_kernel("power_law", d=0.5, L=5)
        self.assertGreaterEqual(k(0), 0.0)
        self.assertGreaterEqual(k(1), 0.0)
        self.assertGreaterEqual(k(5), 0.0)
        self.assertEqual(k(6), 0.0)
        self.assertEqual(k(-1), 0.0)
        total = sum(k(ell) for ell in range(6))
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_kernel_power_law_invalid_raises(self):
        with self.assertRaises(ValueError):
            make_kernel("power_law", d=0.0, L=5)
        with self.assertRaises(ValueError):
            make_kernel("power_law", d=0.5, L=-1)

    def test_kernel_weighted_history(self):
        # history: 3 time steps, 2 entries; current_t=2, current_n=2
        history = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        k = make_kernel("rectangular", L=10)
        out = kernel_weighted_history(history, current_t=2, kernel=k, current_n=2)
        self.assertEqual(out.shape, (2,))
        # weights 1/10 for delta_t 0,1,2
        expected_0 = (1.0 / 10) * (5.0 + 3.0 + 1.0)
        expected_1 = (1.0 / 10) * (6.0 + 4.0 + 2.0)
        self.assertAlmostEqual(out[0], expected_0)
        self.assertAlmostEqual(out[1], expected_1)


class TestMeanfieldUpdateWithMemory(unittest.TestCase):
    """meanfield_update with optional S_history, p_history, memory_params."""

    def test_meanfield_update_with_memory_sums_to_one(self):
        n = 3
        p_t = np.ones(n) / n
        S_t = np.array([0.1, -0.1, 0.0])
        # history: 2 time steps
        S_history = np.array([[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]])
        p_history = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.4, 0.35, 0.25]])
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10}
        memory_params = {"kernel": "exponential", "decay": 0.9, "eta_S": 0.2, "gamma_history": 0.1}
        p_next = meanfield_update(
            p_t, S_t, None, params,
            S_history=S_history, p_history=p_history,
            memory_params=memory_params, current_t=1,
        )
        self.assertTrue(np.all(p_next >= 0))
        self.assertTrue(np.isclose(p_next.sum(), 1.0))

    def test_meanfield_update_no_memory_unchanged(self):
        n = 3
        p_t = np.ones(n) / n
        S_t = np.array([0.5, -0.3, 0.0])
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10}
        p_next_no_mem = meanfield_update(p_t, S_t, None, params)
        p_next_mem = meanfield_update(
            p_t, S_t, None, params,
            memory_params={"eta_S": 0.0, "gamma_history": 0.0},
            current_t=0,
        )
        self.assertTrue(np.allclose(p_next_no_mem, p_next_mem))

    def test_meanfield_update_with_m_t_eta_m_sums_to_one(self):
        n = 3
        p_t = np.ones(n) / n
        S_t = np.array([0.2, -0.1, 0.0])
        m_t = np.array([0.1, 0.05, -0.05])
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10}
        p_next = meanfield_update(
            p_t, S_t, None, params,
            m_t=m_t, eta_m=0.3,
        )
        self.assertEqual(p_next.shape, (n,))
        self.assertTrue(np.all(p_next >= 0))
        self.assertTrue(np.isclose(p_next.sum(), 1.0))


class TestUpdateMemoryState(unittest.TestCase):
    """update_memory_state: m_next = (1 - lam)*m_prev + lam*S_t."""

    def test_update_memory_state_known(self):
        m_prev = np.array([0.0, 1.0, -0.5])
        S_t = np.array([1.0, 0.0, 0.0])
        lam = 0.2
        m_next = update_memory_state(m_prev, S_t, lam)
        expected = (1 - lam) * m_prev + lam * S_t
        self.assertTrue(np.allclose(m_next, expected))

    def test_update_memory_state_lam_one(self):
        m_prev = np.array([0.5, 0.0])
        S_t = np.array([1.0, -1.0])
        # lam must be in (0, 1), so 1.0 is invalid
        with self.assertRaises(ValueError):
            update_memory_state(m_prev, S_t, 1.0)

    def test_update_memory_state_lam_zero_raises(self):
        m_prev = np.array([0.0])
        S_t = np.array([1.0])
        with self.assertRaises(ValueError):
            update_memory_state(m_prev, S_t, 0.0)

    def test_update_memory_state_shape_mismatch_raises(self):
        m_prev = np.array([0.0, 1.0])
        S_t = np.array([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            update_memory_state(m_prev, S_t, 0.5)


class TestInfluenceMatrix(unittest.TestCase):
    """influence_matrix_from_adjacency: row-stochastic, self-weight."""

    def test_influence_row_stochastic(self):
        A = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        W = influence_matrix_from_adjacency(A, omega_self=0.5)
        self.assertEqual(W.shape, (3, 3))
        self.assertTrue(np.allclose(W.sum(axis=1), np.ones(3)))

    def test_influence_self_weight_positive(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        W = influence_matrix_from_adjacency(A, omega_self=1.0)
        self.assertGreater(W[0, 0], 0.0)
        self.assertGreater(W[1, 1], 0.0)
        self.assertTrue(np.allclose(W.sum(axis=1), np.ones(2)))

    def test_influence_omega_self_zero_allowed(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        W = influence_matrix_from_adjacency(A, omega_self=0.0)
        self.assertEqual(W.shape, (2, 2))
        self.assertTrue(np.allclose(W.sum(axis=1), np.ones(2)))

    def test_influence_omega_self_negative_raises(self):
        A = np.eye(2)
        with self.assertRaises(ValueError) as ctx:
            influence_matrix_from_adjacency(A, omega_self=-0.1)
        self.assertIn("omega_self", str(ctx.exception))


class TestMeanfieldUpdateSmallworld(unittest.TestCase):
    """meanfield_update_smallworld: simplex per community; delta=0 matches non-network; W=I consistency."""

    def test_smallworld_sums_to_one_per_community(self):
        G, n = 3, 4
        rng = np.random.default_rng(44)
        p_t = np.zeros((G, n))
        for g in range(G):
            x = rng.random(n)
            p_t[g] = x / x.sum()
        S_t = np.array([0.1, -0.2, 0.0, 0.3])
        A = watts_strogatz_adjacency(G=G, K=1, beta=0.0, row_stochastic=False, rng=rng)
        W = influence_matrix_from_adjacency(A, omega_self=0.5)
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10, "delta": 0.3}
        p_next, bar_p = meanfield_update_smallworld(
            p_t, S_t, None, params, W, weights_w=np.ones(G) / G,
        )
        self.assertEqual(p_next.shape, (G, n))
        for g in range(G):
            self.assertTrue(np.all(p_next[g] >= 0))
            self.assertTrue(np.isclose(p_next[g].sum(), 1.0))
        self.assertIsNotNone(bar_p)
        self.assertTrue(np.all(bar_p >= 0))
        self.assertTrue(np.isclose(bar_p.sum(), 1.0))

    def test_smallworld_delta_zero_matches_per_community_meanfield(self):
        G, n = 2, 3
        p_t = np.ones((G, n)) / n
        S_t = np.array([0.5, -0.3, 0.0])
        W = np.eye(G)  # no cross-community influence
        params = {"kappa": 0.2, "eta": 1.0, "gamma": 0.5, "epsilon": 1e-10, "delta": 0.0}
        p_next_sw, _ = meanfield_update_smallworld(p_t, S_t, None, params, W, weights_w=None)
        for g in range(G):
            p_next_g = meanfield_update(p_t[g], S_t, None, params)
            self.assertTrue(np.allclose(p_next_sw[g], p_next_g), f"community {g} should match single meanfield")

    def test_smallworld_aggregate_with_weights(self):
        G, n = 2, 3
        p_t = np.ones((G, n)) / n
        p_t[0] = np.array([1.0, 0.0, 0.0])
        p_t[1] = np.array([0.0, 1.0, 0.0])
        S_t = np.zeros(n)
        W = np.eye(G)
        params = {"kappa": 0.0, "eta": 0.0, "gamma": 0.0, "epsilon": 1e-10, "delta": 0.0}
        p_next, bar_p = meanfield_update_smallworld(
            p_t, S_t, None, params, W, weights_w=np.array([0.7, 0.3]),
        )
        self.assertTrue(np.allclose(p_next, p_t))
        self.assertTrue(np.allclose(bar_p, 0.7 * p_t[0] + 0.3 * p_t[1]))


if __name__ == "__main__":
    unittest.main()
