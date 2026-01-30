"""
Biased mean-field voter model (Layer 2).
Discrete-time week-to-week evolution: switching-rate microfoundation (F1),
mean-field update (F2), and underdog/rage-vote term (F3).
Networked mean-field: multiple communities coupled via Watts–Strogatz graph.
Hereditary/fading memory: kernel-weighted history of signals and opinions.
"""

import numpy as np
import pandas as pd
from src.models.vote_latent import shares_from_preference


def make_kernel(
    name: str,
    **kwargs: float,
) -> callable:
    """
    Return a kernel callable k(delta_t) for weighting history.

    Parameters
    ----------
    name : "exponential" | "rectangular" | "power_law"
    **kwargs
        exponential: decay (float in (0, 1], default 0.9). k(delta_t) = decay^delta_t.
        rectangular: L (int or float, window length). k(delta_t) = 1/L if delta_t <= L else 0.
        power_law: d (float > 0), L (int). K(ell) ∝ (ell+1)^{-d}, normalized over ell in [0, L].

    Returns
    -------
    k : callable
        k(delta_t) returns weight for lag delta_t (delta_t >= 0).
    """
    if name == "exponential":
        decay = float(kwargs.get("decay", 0.9))
        if not 0 < decay <= 1:
            raise ValueError("exponential kernel decay must be in (0, 1]")

        def k(delta_t: int | float) -> float:
            d = int(delta_t) if delta_t == int(delta_t) else float(delta_t)
            if d < 0:
                return 0.0
            return float(decay**d)

        return k
    if name == "rectangular":
        L = kwargs.get("L", 5)
        L = int(L) if L == int(L) else float(L)
        if L <= 0:
            raise ValueError("rectangular kernel L must be positive")

        def k(delta_t: int | float) -> float:
            d = int(delta_t) if delta_t == int(delta_t) else float(delta_t)
            if d < 0 or d > L:
                return 0.0
            return 1.0 / float(L)

        return k
    if name == "power_law":
        d = float(kwargs.get("d", 0.5))
        L = int(kwargs.get("L", 10))
        if d <= 0:
            raise ValueError("power_law kernel d must be > 0")
        if L < 0:
            raise ValueError("power_law kernel L must be non-negative")
        # K(ell) ∝ (ell + 1)^{-d} for ell = 0, 1, ..., L; normalize to sum to 1
        weights = np.array([(ell + 1) ** (-d) for ell in range(L + 1)], dtype=float)
        total = weights.sum()
        if total <= 0:
            raise ValueError("power_law kernel normalizer must be positive")
        weights = weights / total
        lookup = dict(enumerate(weights))

        def k(delta_t: int | float) -> float:
            ell = int(delta_t) if delta_t == int(delta_t) else int(float(delta_t))
            if ell < 0 or ell > L:
                return 0.0
            return float(lookup[ell])

        return k
    raise ValueError("kernel name must be 'exponential', 'rectangular', or 'power_law'")


def kernel_weighted_history(
    history_arrays: np.ndarray | list[np.ndarray],
    current_t: int,
    kernel: callable,
    current_n: int,
) -> np.ndarray:
    """
    Kernel-weighted sum over past: result[i] = sum_{tau=0}^{current_t} k(current_t - tau) * history[tau, i].
    Aligned to first current_n entries (current active set).

    Parameters
    ----------
    history_arrays : array (T+1, n_max) or list of length-(n_max) arrays
        history[tau] is the vector at time tau (e.g. S_tau or p_tau).
    current_t : int
        Current time index (0 <= current_t <= T).
    kernel : callable
        k(delta_t) returns weight for lag delta_t.
    current_n : int
        Number of active contestants; use first current_n entries per time.

    Returns
    -------
    out : array (current_n,)
        Kernel-weighted history for the current active set.
    """
    if isinstance(history_arrays, list):
        history_arrays = np.array([np.asarray(a).ravel() for a in history_arrays])
    history_arrays = np.asarray(history_arrays, dtype=float)
    if history_arrays.ndim == 1:
        history_arrays = history_arrays.reshape(1, -1)
    T_plus_1, n_max = history_arrays.shape
    if current_n > n_max:
        raise ValueError("current_n must be <= number of columns in history_arrays")
    if current_t < 0 or current_t >= T_plus_1:
        raise ValueError("current_t must be in [0, len(history_arrays)-1]")
    out = np.zeros(current_n)
    for tau in range(current_t + 1):
        w = kernel(current_t - tau)
        if w != 0:
            out += w * history_arrays[tau, :current_n]
    return out


def update_memory_state(
    m_prev: np.ndarray,
    S_t: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Exponential fading memory (Markovian state): m_next = (1 - lam) * m_prev + lam * S_t.
    Same across communities; caller maintains m per (season, week) and updates after each step.

    Parameters
    ----------
    m_prev : array (n,)
        Previous memory state per contestant.
    S_t : array (n,)
        Current judge signal (e.g. z-scores) per contestant.
    lam : float
        Memory update rate in (0, 1). Larger lambda gives more weight to current S_t.

    Returns
    -------
    m_next : array (n,)
        Updated memory state.
    """
    m_prev = np.asarray(m_prev, dtype=float).ravel()
    S_t = np.asarray(S_t, dtype=float).ravel()
    if m_prev.shape != S_t.shape:
        raise ValueError("m_prev and S_t must have same length")
    if not (0 < lam < 1):
        raise ValueError("lam must be in (0, 1)")
    return (1.0 - lam) * m_prev + lam * S_t


def watts_strogatz_adjacency(
    G: int,
    K: int,
    beta: float,
    *,
    row_stochastic: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Watts–Strogatz small-world graph: ring lattice with K nearest neighbors per node,
    then each edge rewired with probability beta.

    Parameters
    ----------
    G : int
        Number of nodes (communities).
    K : int
        Each node connected to K nearest neighbors (K must be < G).
    beta : float
        Rewiring probability per edge, in [0, 1].
    row_stochastic : bool
        If True, normalize each row to sum to 1 (for neighbor averaging).
    rng : np.random.Generator or None
        Random state; if None, use default.

    Returns
    -------
    A : array (G, G)
        Adjacency (optionally row-stochastic).
    """
    if G < 2:
        return np.ones((G, G)) / max(1, G)
    if K >= G:
        K = G - 1
    rng = rng or np.random.default_rng()
    # Ring lattice: node i connected to (i+1)%G, ..., (i+K)%G
    A = np.zeros((G, G))
    for i in range(G):
        for j in range(1, K + 1):
            neighbor = (i + j) % G
            A[i, neighbor] = 1.0
    # Rewire each edge with probability beta
    for i in range(G):
        for j in range(1, K + 1):
            if rng.random() >= beta:
                continue
            neighbor = (i + j) % G
            A[i, neighbor] = 0.0
            candidates = [k for k in range(G) if k != i and A[i, k] == 0]
            if not candidates:
                A[i, neighbor] = 1.0
                continue
            new_j = rng.choice(candidates)
            A[i, new_j] = 1.0
    if row_stochastic:
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        A = A / row_sums
    return A


def influence_matrix_from_adjacency(A: np.ndarray, omega_self: float) -> np.ndarray:
    """
    Turn adjacency A into a row-stochastic influence matrix W (DeGroot weights).
    W_gh ∝ A_gh + 1[g=h]*omega_self; each row normalized to sum to 1.

    Build small-world graph as:
      A = watts_strogatz_adjacency(G, K, p_rewire, row_stochastic=False)
      W = influence_matrix_from_adjacency(A, omega_self)

    Parameters
    ----------
    A : array (G, G)
        Adjacency (not necessarily row-stochastic).
    omega_self : float
        Self-weight; must be >= 0.

    Returns
    -------
    W : array (G, G)
        Row-stochastic influence matrix.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if omega_self < 0:
        raise ValueError("omega_self must be >= 0")
    G = A.shape[0]
    W = A + omega_self * np.eye(G)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    W = W / row_sums
    return W


def underdog_score(
    S_t: np.ndarray,
    p_t: np.ndarray,
    mode: str = "smooth",
    *,
    a: float = 1.0,
    b: float = 1.0,
    tau_p: float | None = None,
    tau_S: float | None = None,
    **kwargs: float,
) -> np.ndarray:
    """
    Underdog / rage-vote term: low judge score (S) but high popularity (p).

    Indicator: underdog_i = 1[S_i low] * 1[p_i high] with thresholds tau_S, tau_p.
    Smooth:   underdog_i = sigma(a(p_i - tau_p)) * sigma(b(tau_S - S_i)), sigma(x)=1/(1+exp(-x)).

    Parameters
    ----------
    S_t : array (n,)
        Judge z-scores within week.
    p_t : array (n,)
        Fan shares (popularity).
    mode : "smooth" | "indicator"
    a, b : float
        Smoothness for smooth mode.
    tau_p, tau_S : float
        Thresholds: low S means S_i <= tau_S, high p means p_i >= tau_p.
        If None in indicator mode, use median of S_t and p_t respectively.
    **kwargs : optional tau_p, tau_S, a, b overrides.

    Returns
    -------
    array (n,) of underdog scores in [0, 1].
    """
    S_t = np.asarray(S_t, dtype=float)
    p_t = np.asarray(p_t, dtype=float)
    n = len(S_t)
    if len(p_t) != n:
        raise ValueError("S_t and p_t must have same length")

    tau_p = kwargs.get("tau_p", tau_p)
    tau_S = kwargs.get("tau_S", tau_S)
    a = kwargs.get("a", a)
    b = kwargs.get("b", b)

    if mode == "indicator":
        if tau_S is None:
            tau_S = np.median(S_t)
        if tau_p is None:
            tau_p = np.median(p_t)
        low_S = (S_t <= tau_S).astype(float)
        high_p = (p_t >= tau_p).astype(float)
        return low_S * high_p

    if mode == "smooth":
        if tau_S is None:
            tau_S = np.median(S_t)
        if tau_p is None:
            tau_p = np.median(p_t)
        # sigma(x) = 1 / (1 + exp(-x))
        def sig(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -500, 500)
            return 1.0 / (1.0 + np.exp(-x))
        high_p_term = sig(a * (p_t - tau_p))
        low_S_term = sig(b * (tau_S - S_t))
        return high_p_term * low_S_term

    raise ValueError("mode must be 'smooth' or 'indicator'")


def switching_target(
    p_t: np.ndarray,
    S_t: np.ndarray,
    rho: float,
    eta: float,
) -> np.ndarray:
    """
    F1 microfoundation: target distribution for switching rates.
    q_t = rho * p_t + (1 - rho) * softmax(eta * S_t).
    """
    p_t = np.asarray(p_t, dtype=float)
    S_t = np.asarray(S_t, dtype=float)
    n = len(p_t)
    if len(S_t) != n:
        raise ValueError("p_t and S_t must have same length")
    broadcast_signal = shares_from_preference(eta * S_t)
    q_t = rho * p_t + (1.0 - rho) * broadcast_signal
    return q_t / q_t.sum()


def _utility_with_memory(
    S_t: np.ndarray,
    p_t: np.ndarray,
    X: np.ndarray | None,
    params: dict,
    *,
    underdog: np.ndarray | None = None,
    beta_U: float = 0.0,
    S_history: np.ndarray | list[np.ndarray] | None = None,
    p_history: np.ndarray | list[np.ndarray] | None = None,
    memory_params: dict | None = None,
    current_t: int | None = None,
    m_t: np.ndarray | None = None,
    eta_m: float = 0.0,
) -> np.ndarray:
    """
    Build utility u_t; optionally with kernel-weighted S_bar, p_bar and Markovian m_t.

    Memory channels (same across communities):
    - Judge memory: S_history + memory_params["eta_S"] -> kernel-weighted S_bar (M_{i,t}).
    - Social buzz memory: p_history + memory_params["gamma_history"] -> kernel-weighted log(p_bar).
    - Markovian fading: m_t + eta_m -> eta_m * m_t (m_t = (1-lam)*m_{t-1} + lam*S_t, maintained by caller).
    """
    n = len(p_t)
    eps = float(params.get("epsilon", 1e-10))
    eta = float(params["eta"])
    gamma = float(params["gamma"])
    eta_S = 0.0
    gamma_history = 0.0
    kernel = None
    if memory_params is not None and (S_history is not None or p_history is not None) and current_t is not None:
        kernel_name = memory_params.get("kernel", "exponential")
        kernel_kw = {
            k: v for k, v in memory_params.items()
            if k not in ("kernel", "eta_S", "gamma_history", "eta_m", "memory_lambda")
        }
        kernel = make_kernel(kernel_name, **kernel_kw)
        eta_S = float(memory_params.get("eta_S", 0.0))
        gamma_history = float(memory_params.get("gamma_history", 0.0))

    if kernel is not None and (eta_S != 0 or gamma_history != 0):
        S_bar = kernel_weighted_history(S_history, current_t, kernel, n) if S_history is not None and eta_S != 0 else np.zeros(n)
        p_bar = kernel_weighted_history(p_history, current_t, kernel, n) if p_history is not None and gamma_history != 0 else np.ones(n) / n
        u = eta_S * S_bar + eta * S_t + gamma_history * np.log(np.clip(p_bar, 1e-15, None) + eps) + gamma * np.log(p_t + eps)
    else:
        u = eta * S_t + gamma * np.log(p_t + eps)

    if m_t is not None and eta_m != 0:
        m_t = np.asarray(m_t, dtype=float).ravel()
        if len(m_t) != n:
            raise ValueError("m_t must have same length as p_t")
        u = u + eta_m * m_t

    if X is not None and "theta" in params:
        X = np.asarray(X)
        theta = np.atleast_1d(params["theta"])
        u = u + np.dot(X, theta)
    if underdog is not None and beta_U != 0:
        u = u + beta_U * np.asarray(underdog, dtype=float)
    return u


def _utility_smallworld(
    S_t: np.ndarray,
    p_t_per_community: np.ndarray,
    X: np.ndarray | None,
    params: dict,
    W: np.ndarray,
    delta: float,
    *,
    underdog: np.ndarray | None = None,
    beta_U: float = 0.0,
) -> np.ndarray:
    """
    Build per-community utility u^(g) with small-world social influence.
    u_i^(g) = eta*S_i + theta'X_i + beta_U*underdog_i + gamma*log(p_i^(g)+eps)
              + delta * sum_h W_gh * log(p_i^(h)+eps).

    Parameters
    ----------
    S_t : array (n,)
    p_t_per_community : array (G, n); each row sums to 1.
    X : (n, k) or None
    params : dict with eta, gamma, epsilon, theta (length k).
    W : array (G, G), row-stochastic influence matrix.
    delta : float, social-influence weight.
    underdog : (n,) or None
    beta_U : float

    Returns
    -------
    u_per_community : array (G, n)
    """
    p_t_per_community = np.asarray(p_t_per_community, dtype=float)
    W = np.asarray(W, dtype=float)
    G, n = p_t_per_community.shape
    if len(S_t) != n:
        raise ValueError("p_t_per_community columns and S_t must have same length")
    if W.shape != (G, G):
        raise ValueError("W must be (G, G)")
    eps = float(params.get("epsilon", 1e-10))
    eta = float(params["eta"])
    gamma = float(params["gamma"])
    log_p = np.log(np.clip(p_t_per_community, 1e-15, None) + eps)  # (G, n)
    social = W @ log_p  # (G, n)
    u_per_community = np.zeros((G, n))
    for g in range(G):
        u = eta * S_t + gamma * log_p[g]
        if delta != 0:
            u = u + delta * social[g]
        if X is not None and "theta" in params:
            X_arr = np.asarray(X)
            theta = np.atleast_1d(params["theta"])
            u = u + np.dot(X_arr, theta)
        if underdog is not None and beta_U != 0:
            u = u + beta_U * np.asarray(underdog, dtype=float)
        u_per_community[g] = u
    return u_per_community


def meanfield_update(
    p_t: np.ndarray,
    S_t: np.ndarray,
    X: np.ndarray | None,
    params: dict,
    *,
    underdog: np.ndarray | None = None,
    beta_U: float = 0.0,
    S_history: np.ndarray | list[np.ndarray] | None = None,
    p_history: np.ndarray | list[np.ndarray] | None = None,
    memory_params: dict | None = None,
    current_t: int | None = None,
    m_t: np.ndarray | None = None,
    eta_m: float = 0.0,
    memory_lambda: float | None = None,
) -> np.ndarray:
    """
    F2 + F3: one-week mean-field update.
    p_{t+1} = (1 - kappa) * p_t + kappa * softmax(u_t).
    u_t = eta*S_t + gamma*log(p_t+eps) + ...; with optional memory:
    u_t = eta_S*S_bar_t + eta*S_t + gamma_history*log(p_bar_t+eps) + gamma*log(p_t+eps) + eta_m*m_t + ...

    Caller is responsible for updating m via update_memory_state(m_prev, S_t, memory_lambda) after each step.

    Parameters
    ----------
    p_t, S_t : arrays (n,)
    X : (n, k) or None; if None, theta'X is omitted.
    params : dict with kappa, eta, gamma, epsilon, theta (length k).
    underdog : (n,) or None; if None, beta_U term is omitted.
    beta_U : float
    S_history, p_history : optional (T+1, n) or list of arrays; for kernel-weighted S_bar, p_bar.
    memory_params : optional dict with kernel ("exponential"|"rectangular"|"power_law"), decay or L/d, eta_S, gamma_history.
    current_t : optional int; required if memory is used.
    m_t : optional (n,) Markovian memory state; use with eta_m.
    eta_m : float; weight for m_t in utility (same across communities).
    memory_lambda : float in (0,1); used by caller to update m (not used inside this function).

    Returns
    -------
    p_next : (n,) summing to 1.
    """
    p_t = np.asarray(p_t, dtype=float)
    S_t = np.asarray(S_t, dtype=float)
    n = len(p_t)
    if len(S_t) != n:
        raise ValueError("p_t and S_t must have same length")

    kappa = float(params["kappa"])
    eps = float(params.get("epsilon", 1e-10))
    eta_m_val = float(eta_m) if eta_m is not None else 0.0

    u = _utility_with_memory(
        S_t, p_t, X, params,
        underdog=underdog, beta_U=beta_U,
        S_history=S_history, p_history=p_history,
        memory_params=memory_params, current_t=current_t,
        m_t=m_t, eta_m=eta_m_val,
    )

    soft = shares_from_preference(u)
    p_next = (1.0 - kappa) * p_t + kappa * soft
    return p_next / p_next.sum()


def meanfield_update_networked(
    p_t_per_community: np.ndarray,
    S_t: np.ndarray,
    X: np.ndarray | None,
    params: dict,
    graph_adjacency: np.ndarray,
    weights_w: np.ndarray,
    *,
    underdog: np.ndarray | None = None,
    beta_U: float = 0.0,
    coupling_alpha: float = 0.0,
    S_history: np.ndarray | list[np.ndarray] | None = None,
    p_history: np.ndarray | None = None,
    memory_params: dict | None = None,
    current_t: int | None = None,
    m_t: np.ndarray | None = None,
    eta_m: float = 0.0,
    memory_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Networked mean-field update: G communities with share vectors p_t^{(g)},
    coupled via graph_adjacency (option A: neighbor-mixed opinion in utility).
    Returns per-community p_{t+1}^{(g)} and aggregate bar_p_{t+1} = sum_g w_g p_{t+1}^{(g)}.
    Optional memory: S_history (T+1, n), p_history (T+1, G, n) for kernel-weighted S_bar, p_bar per community.
    Optional Markovian m_t (same across communities); caller updates m via update_memory_state.

    Parameters
    ----------
    p_t_per_community : array (G, n)
        Share vector per community; each row sums to 1.
    S_t : array (n,)
        Judge z-scores (same across communities).
    X : (n, k) or None
    params : dict with kappa, eta, gamma, epsilon, theta (length k).
    graph_adjacency : array (G, G)
        Row-stochastic adjacency for neighbor averaging.
    weights_w : array (G,)
        Reporting weights; must sum to 1, w_g >= 0.
    underdog : (n,) or None
    beta_U : float
    coupling_alpha : float
        Mix: tilde_p^{(g)} = (1 - alpha)*p^{(g)} + alpha * A @ p (row-wise).
    S_history, p_history : optional; p_history shape (T+1, G, n) for per-community history.
    memory_params : optional dict with kernel, eta_S, gamma_history, decay or L/d.
    current_t : optional int; required if memory is used.
    m_t : optional (n,) Markovian memory state; use with eta_m.
    eta_m : float; weight for m_t in utility (same across communities).
    memory_lambda : float in (0,1); used by caller to update m (not used inside this function).

    Returns
    -------
    p_next_per_community : (G, n), each row sums to 1.
    bar_p_next : (n,) aggregate sum_g w_g p_{t+1}^{(g)}.
    """
    p_t_per_community = np.asarray(p_t_per_community, dtype=float)
    S_t = np.asarray(S_t, dtype=float)
    graph_adjacency = np.asarray(graph_adjacency, dtype=float)
    weights_w = np.asarray(weights_w, dtype=float)
    G, n = p_t_per_community.shape
    if len(S_t) != n:
        raise ValueError("p_t_per_community columns and S_t must have same length")
    if graph_adjacency.shape != (G, G):
        raise ValueError("graph_adjacency must be (G, G)")
    if len(weights_w) != G:
        raise ValueError("weights_w must have length G")
    if not np.isclose(weights_w.sum(), 1.0) or np.any(weights_w < 0):
        raise ValueError("weights_w must be non-negative and sum to 1")

    kappa = float(params["kappa"])
    eps = float(params.get("epsilon", 1e-10))

    # Option A: tilde_p^{(g)} = (1 - alpha)*p^{(g)} + alpha * (A @ p)[g]
    neighbor_p = graph_adjacency @ p_t_per_community  # (G, n)
    tilde_p = (1.0 - coupling_alpha) * p_t_per_community + coupling_alpha * neighbor_p
    tilde_p = np.clip(tilde_p, 1e-15, None)
    tilde_p = tilde_p / tilde_p.sum(axis=1, keepdims=True)

    p_next_per_community = np.zeros((G, n))
    for g in range(G):
        p_t_g = p_t_per_community[g]
        p_tilde_g = tilde_p[g]
        # Per-community p_history slice: (T+1, n) for community g
        p_history_g = None
        if p_history is not None and memory_params is not None and current_t is not None:
            p_hist = np.asarray(p_history, dtype=float)
            if p_hist.ndim == 3 and p_hist.shape[1] == G:
                p_history_g = p_hist[:, g, :]
        eta_m_val = float(eta_m) if eta_m is not None else 0.0
        u = _utility_with_memory(
            S_t, p_tilde_g, X, params,
            underdog=underdog, beta_U=beta_U,
            S_history=S_history, p_history=p_history_g,
            memory_params=memory_params, current_t=current_t,
            m_t=m_t, eta_m=eta_m_val,
        )
        soft = shares_from_preference(u)
        p_next_g = (1.0 - kappa) * p_t_g + kappa * soft
        p_next_per_community[g] = p_next_g / p_next_g.sum()

    bar_p_next = np.dot(weights_w, p_next_per_community)
    bar_p_next = bar_p_next / bar_p_next.sum()
    return p_next_per_community, bar_p_next


def meanfield_update_smallworld(
    p_t_per_community: np.ndarray,
    S_t: np.ndarray,
    X: np.ndarray | None,
    params: dict,
    W: np.ndarray,
    *,
    underdog: np.ndarray | None = None,
    beta_U: float = 0.0,
    weights_w: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    S_history: np.ndarray | list[np.ndarray] | None = None,
    p_history: np.ndarray | None = None,
    memory_params: dict | None = None,
    current_t: int | None = None,
    m_t: np.ndarray | None = None,
    eta_m: float = 0.0,
    memory_lambda: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Small-world mean-field update: per-community utility with delta*(W@log(p)) term.
    p_{t+1}^{(g)} = (1 - kappa)*p_t^{(g)} + kappa*softmax(u_t^{(g)}).
    u^{(g)} = eta*S + theta'X + beta_U*underdog + gamma*log(p^{(g)}+eps)
              + delta * (W @ log(p+eps))_g.
    Optional memory (same eta_m and kernel across communities): S_history, p_history (T+1, G, n),
    memory_params, current_t, m_t, eta_m; caller updates m via update_memory_state.
    Optional: add sigma_shock * N(0,I) to u before softmax (logit-normal shock).

    Parameters
    ----------
    p_t_per_community : array (G, n), share vector per community; each row sums to 1.
    S_t : array (n,), judge z-scores (same across communities).
    X : (n, k) or None
    params : dict with kappa, eta, gamma, epsilon, theta (length k), delta; optionally sigma_shock.
    W : array (G, G), row-stochastic influence matrix (from influence_matrix_from_adjacency).
    underdog : (n,) or None
    beta_U : float
    weights_w : array (G,) or None; if provided, return aggregate bar_p_next = sum_g w_g p_{t+1}^{(g)}.
    rng : np.random.Generator or None; used when sigma_shock > 0.
    S_history, p_history : optional; p_history shape (T+1, G, n) for per-community kernel-weighted p_bar.
    memory_params : optional dict with kernel, eta_S, gamma_history, decay or L/d.
    current_t : optional int; required if memory is used.
    m_t : optional (n,) Markovian memory state; use with eta_m.
    eta_m : float; weight for m_t in utility (same across communities).
    memory_lambda : float in (0,1); used by caller to update m (not used inside this function).

    Returns
    -------
    p_next_per_community : (G, n), each row sums to 1.
    bar_p_next : (n,) or None; aggregate if weights_w provided, else None.
    """
    p_t_per_community = np.asarray(p_t_per_community, dtype=float)
    S_t = np.asarray(S_t, dtype=float)
    W = np.asarray(W, dtype=float)
    G, n = p_t_per_community.shape
    if len(S_t) != n:
        raise ValueError("p_t_per_community columns and S_t must have same length")
    if W.shape != (G, G):
        raise ValueError("W must be (G, G)")
    kappa = float(params["kappa"])
    delta = float(params.get("delta", 0.0))
    sigma_shock = float(params.get("sigma_shock", 0.0))
    rng = rng or np.random.default_rng()
    eps = float(params.get("epsilon", 1e-10))
    log_p = np.log(np.clip(p_t_per_community, 1e-15, None) + eps)
    social = W @ log_p  # (G, n)

    use_memory = (
        (S_history is not None or p_history is not None)
        and memory_params is not None
        and current_t is not None
    ) or (m_t is not None and eta_m != 0)

    if use_memory:
        eta_m_val = float(eta_m) if eta_m is not None else 0.0
        u_per_community = np.zeros((G, n))
        for g in range(G):
            p_history_g = None
            if p_history is not None and memory_params is not None and current_t is not None:
                p_hist = np.asarray(p_history, dtype=float)
                if p_hist.ndim == 3 and p_hist.shape[1] == G:
                    p_history_g = p_hist[:, g, :]
            u_g = _utility_with_memory(
                S_t, p_t_per_community[g], X, params,
                underdog=underdog, beta_U=beta_U,
                S_history=S_history, p_history=p_history_g,
                memory_params=memory_params, current_t=current_t,
                m_t=m_t, eta_m=eta_m_val,
            )
            u_g = u_g + delta * social[g]
            u_per_community[g] = u_g
    else:
        u_per_community = _utility_smallworld(
            S_t, p_t_per_community, X, params, W, delta,
            underdog=underdog, beta_U=beta_U,
        )
    if sigma_shock > 0:
        u_per_community = u_per_community + sigma_shock * rng.standard_normal((G, n))

    p_next_per_community = np.zeros((G, n))
    for g in range(G):
        soft = shares_from_preference(u_per_community[g])
        p_next_g = (1.0 - kappa) * p_t_per_community[g] + kappa * soft
        p_next_per_community[g] = p_next_g / p_next_g.sum()

    bar_p_next = None
    if weights_w is not None:
        weights_w = np.asarray(weights_w, dtype=float)
        if len(weights_w) != G:
            raise ValueError("weights_w must have length G")
        if not np.isclose(weights_w.sum(), 1.0) or np.any(weights_w < 0):
            raise ValueError("weights_w must be non-negative and sum to 1")
        bar_p_next = np.dot(weights_w, p_next_per_community)
        bar_p_next = bar_p_next / bar_p_next.sum()
    return p_next_per_community, bar_p_next


def design_matrix_active(cw: pd.DataFrame, season: int, week: int, columns: list[str] | None = None) -> np.ndarray:
    """
    Build design matrix X for active contestants in (season, week).
    cw must already have covariate columns (e.g. from build_contestant_week_covariates).
    columns: list of column names to use; default ["age", "industry_dummy"] if present.
    """
    grp = cw[(cw["season"] == season) & (cw["week"] == week) & (cw["active"] == 1)]
    if grp.empty:
        return np.zeros((0, 0))
    if columns is None:
        candidates = ["age", "industry_dummy"]
        columns = [c for c in candidates if c in grp.columns]
    if not columns:
        return np.ones((len(grp), 1))
    return grp[columns].astype(float).values
