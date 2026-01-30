"""
Margin-to-flip (H1): minimal perturbation to fan shares that would change
the eliminated contestant. Yields a robustness radius per week; small radius => uncertain week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from src.scoring import percent_combined, rank_combined


def _softmax_log_param(x: np.ndarray) -> np.ndarray:
    """x is unconstrained; return f = exp(x) / sum(exp(x)) on simplex."""
    x = np.asarray(x, dtype=float)
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def _f_prime_from_z(f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """f' = softmax(log(f) + z). Keeps simplex; z=0 => f'=f."""
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=float)
    np.clip(f, 1e-15, None, out=f)
    log_f = np.log(f)
    return _softmax_log_param(log_f + z)


def _eliminated_percent(J: np.ndarray, f: np.ndarray) -> int:
    """Eliminated index under percent rule (combined j + f)."""
    j = J / J.sum() if J.sum() > 0 else np.ones(len(J)) / len(J)
    c = j + f
    return int(np.argmin(c))


def _eliminated_rank(J: np.ndarray, f: np.ndarray) -> int:
    """Eliminated index under rank rule (V proportional to f)."""
    V = f * (1.0 / f.sum()) if f.sum() > 0 else np.ones(len(f)) / len(f)
    _, elim = rank_combined(J, V)
    return elim


def _smooth_min_approx(x: np.ndarray, k: float = 20.0) -> float:
    """Smooth approximation to min(x): -log(sum(exp(-k*x)))/k."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    mx = x.max()
    return mx - np.log(np.exp(k * (x - mx)).sum()) / k


def margin_to_flip_radius(
    J: np.ndarray,
    f: np.ndarray,
    rule: str,
    current_eliminated_index: int,
    *,
    tol: float = 1e-6,
    k_smooth: float = 20.0,
) -> tuple[float, np.ndarray | None]:
    """
    Find minimal L2 perturbation (in log-param space) to f such that the
    eliminated contestant changes. f must be on simplex (non-negative, sum 1).

    Parameters
    ----------
    J : (n,) judge totals
    f : (n,) fan shares (simplex)
    rule : "percent" | "rank"
    current_eliminated_index : index in [0, n) of who is currently eliminated
    tol : constraint tolerance for flip
    k_smooth : smooth-min parameter for constraint

    Returns
    -------
    radius : L2 norm of optimal z (or np.inf if no flip possible / already tied)
    f_opt : optimal f' on simplex that achieves flip, or None
    """
    J = np.asarray(J, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(J)
    if n < 2 or current_eliminated_index < 0 or current_eliminated_index >= n:
        return np.inf, None
    if not np.isclose(f.sum(), 1.0) or np.any(f < 0):
        f = np.clip(f, 1e-15, None)
        f = f / f.sum()

    def f_prime(z: np.ndarray) -> np.ndarray:
        return _f_prime_from_z(f, z)

    if rule == "percent":
        j_norm = J / J.sum() if J.sum() > 0 else np.ones(n) / n

        def constraint_flip(z: np.ndarray) -> float:
            fp = f_prime(z)
            c = j_norm + fp
            c_elim = c[current_eliminated_index]
            others = np.concatenate([c[:current_eliminated_index], c[current_eliminated_index + 1:]])
            if len(others) == 0:
                return 0.0
            m_other = _smooth_min_approx(others, k_smooth)
            return m_other - c_elim + tol  # want <= 0 so m_other < c_elim - tol
    else:
        # rank: eliminated = argmax R. We need argmax R' != current_eliminated_index.
        # R_i = r_j_i + r_f_i. So we need some other i to have R'_i > R'_{current_elim}.
        # Smooth constraint: smooth_max(R'_i for i != current_elim) - R'_{current_elim} >= tol
        def constraint_flip(z: np.ndarray) -> float:
            fp = f_prime(z)
            V = fp / fp.sum() if fp.sum() > 0 else np.ones(n) / n
            r_j = rankdata(-J, method="average")
            r_f = rankdata(-V, method="average")
            R = r_j + r_f
            R_elim = R[current_eliminated_index]
            others = np.concatenate([R[:current_eliminated_index], R[current_eliminated_index + 1:]])
            if len(others) == 0:
                return 0.0
            # We need max(others) > R_elim to flip. Smooth max = log(sum(exp(k*x)))/k
            m_other = np.log(np.exp(k_smooth * others).sum()) / k_smooth
            return R_elim - m_other + tol  # want <= 0 so m_other > R_elim + tol
    # Objective: 0.5 * ||z||^2
    def objective(z: np.ndarray) -> float:
        return 0.5 * (z ** 2).sum()

    # Constraint: constraint_flip(z) <= 0
    constraints = {"type": "ineq", "fun": lambda z: -constraint_flip(z)}
    z0 = np.zeros(n)
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=[(-10.0, 10.0)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not result.success:
        return np.inf, None
    z_opt = result.x
    radius = float(np.sqrt(2 * result.fun))
    f_opt = f_prime(z_opt)
    return radius, f_opt


def margin_to_flip_radius_l2_f(
    J: np.ndarray,
    f: np.ndarray,
    rule: str,
    current_eliminated_index: int,
    *,
    tol: float = 1e-6,
    k_smooth: float = 20.0,
) -> tuple[float, np.ndarray | None]:
    """
    Same as margin_to_flip_radius but returns the L2 norm of (f' - f) in share space
    as the radius, for interpretability. Also returns f_opt.
    """
    rad_z, f_opt = margin_to_flip_radius(
        J, f, rule, current_eliminated_index, tol=tol, k_smooth=k_smooth
    )
    if f_opt is None:
        return np.inf, None
    rad_f = float(np.linalg.norm(f_opt - f))
    return rad_f, f_opt


def margin_to_flip_radius_combined(
    J: np.ndarray,
    f: np.ndarray,
    current_eliminated_index: int,
    combined_fn: callable,
    *,
    tol: float = 1e-6,
    k_smooth: float = 20.0,
) -> tuple[float, np.ndarray | None]:
    """
    Minimal L2 perturbation (in log-param space) to f such that the eliminated
    contestant changes, for rules that use a combined score c = combined_fn(f').
    Eliminated = argmin(c). combined_fn(f_prime) must return (n,) array of combined
    scores given fan shares f_prime on simplex.
    """
    J = np.asarray(J, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(J)
    if n < 2 or current_eliminated_index < 0 or current_eliminated_index >= n:
        return np.inf, None
    if not np.isclose(f.sum(), 1.0) or np.any(f < 0):
        f = np.clip(f, 1e-15, None)
        f = f / f.sum()

    def f_prime(z: np.ndarray) -> np.ndarray:
        return _f_prime_from_z(f, z)

    def constraint_flip(z: np.ndarray) -> float:
        fp = f_prime(z)
        c = combined_fn(fp)
        c_elim = c[current_eliminated_index]
        others = np.concatenate([c[:current_eliminated_index], c[current_eliminated_index + 1:]])
        if len(others) == 0:
            return 0.0
        m_other = _smooth_min_approx(others, k_smooth)
        return m_other - c_elim + tol

    def objective(z: np.ndarray) -> float:
        return 0.5 * (z ** 2).sum()

    constraints = {"type": "ineq", "fun": lambda z: -constraint_flip(z)}
    z0 = np.zeros(n)
    result = minimize(
        objective,
        z0,
        method="SLSQP",
        bounds=[(-10.0, 10.0)] * n,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not result.success:
        return np.inf, None
    z_opt = result.x
    radius = float(np.sqrt(2 * result.fun))
    f_opt = f_prime(z_opt)
    return radius, f_opt


def robustness_radii_for_forward_events(
    forward_events: list[dict],
    *,
    use_l2_f: bool = True,
) -> pd.DataFrame:
    """
    Compute robustness radius for each (season, week) in forward_events.
    Returns DataFrame with season, week, rule, radius, observed_eliminated_index.
    """
    rows = []
    for ev in forward_events:
        J = ev["J"]
        f = ev["f"]
        rule = ev["rule"]
        obs = ev.get("observed_eliminated_index")
        if obs is None or len(J) < 2:
            continue
        if use_l2_f:
            radius, _ = margin_to_flip_radius_l2_f(J, f, rule, obs)
        else:
            radius, _ = margin_to_flip_radius(J, f, rule, obs)
        rows.append({
            "season": ev["season"],
            "week": ev["week"],
            "rule": rule,
            "robustness_radius": radius,
            "observed_eliminated_index": obs,
        })
    return pd.DataFrame(rows)
