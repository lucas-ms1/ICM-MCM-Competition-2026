"""
Proposed better system (K): axioms (K1), candidate systems (K2), evaluation (K3).
Axioms are formal and testable; candidate rules live in scoring.py; evaluation in proposed_system_eval.py.
"""

from __future__ import annotations

import numpy as np

from src.scoring import percent_combined, rank_combined
from src.fit.margin_to_flip import margin_to_flip_radius


# ---- K1: Axiom checks (testable predicates / metrics) ----


def monotonicity_holds(J: np.ndarray, eliminated_index: int) -> bool:
    """
    Monotonicity: higher judge score should not hurt survival.
    Check: eliminated contestant has J no greater than all others (so no one with
    strictly lower J survived). Equivalently: J[eliminated] <= J[i] for all i.
    """
    J = np.asarray(J, dtype=float)
    n = len(J)
    if n == 0 or eliminated_index < 0 or eliminated_index >= n:
        return True
    j_e = J[eliminated_index]
    for i in range(n):
        if i != eliminated_index and J[i] < j_e:
            return False
    return True


def fan_relevance_bounds(w_J: float, w_F: float) -> bool:
    """
    Fan relevance (weight bounds): judges and fans both matter.
    Check: 0 < w_J <= 1, w_F = 1 - w_J (or non-negative and w_J + w_F = 1).
    """
    if w_J <= 0 or w_J > 1:
        return False
    if not np.isclose(w_J + w_F, 1.0):
        return False
    return True


def fan_dominance_check(
    J: np.ndarray,
    f: np.ndarray,
    eliminated_index: int,
) -> bool:
    """
    Fan dominance check: the rule did not eliminate the unique max-J contestant
    when they do not have min f. (If eliminated has max J and min f, that is
    consistent with fans dominating; we flag when eliminated has max J and
    not min f as "judges' favorite eliminated by fan vote alone".)
    Returns True if the axiom holds (no such dominance), False if the unique
    max-J contestant was eliminated despite not having min f.
    """
    J = np.asarray(J, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(J)
    if n == 0 or eliminated_index < 0 or eliminated_index >= n:
        return True
    j_max = np.max(J)
    f_min = np.min(f)
    # Unique argmax J
    max_J_indices = np.where(J >= j_max - 1e-12)[0]
    if len(max_J_indices) != 1:
        return True
    i_star = int(max_J_indices[0])
    if i_star != eliminated_index:
        return True
    # Eliminated is the unique max-J. Axiom: they must have min f (so fan vote alone explains it).
    if f[eliminated_index] <= f_min + 1e-12:
        return True
    return False


def robustness_radius(
    J: np.ndarray,
    f: np.ndarray,
    rule: str,
    eliminated_index: int,
) -> float:
    """
    Robustness: radius of smallest perturbation to f that flips elimination.
    Uses margin_to_flip_radius for rule "percent" or "rank". For other rules
    (e.g. proposed), returns np.nan unless a custom callable is supported later.
    """
    J = np.asarray(J, dtype=float)
    f = np.asarray(f, dtype=float)
    if rule not in ("percent", "rank"):
        return np.nan
    radius, _ = margin_to_flip_radius(J, f, rule, eliminated_index)
    return float(radius)


def robustness_axiom(radius: float, threshold: float) -> bool:
    """Robustness axiom: radius >= threshold (small fluctuations do not cause chaos)."""
    if np.isnan(radius):
        return False
    return radius >= threshold


def transparency_rule_description(rule_name: str, params: dict) -> str:
    """Short human-readable description of the rule for transparency."""
    desc = {
        "percent": "Combined score = judge share + fan share; lowest leaves.",
        "rank": "Rank sum of judge and fan ranks; worst sum leaves.",
        "weighted_saturation": "Combined = w * judge share + (1-w) * softcap(fan share); lowest leaves.",
        "percent_trigger": "Percent picks bottom two; judges save only if judge gap > threshold.",
    }
    base = desc.get(rule_name, rule_name)
    if params:
        base += f" Params: {params}."
    return base


def transparency_axiom(rule_name: str, params: dict, max_params: int = 5) -> bool:
    """Transparency: rule has at most max_params parameters."""
    return len(params) <= max_params


def score_axioms_one_week(
    J: np.ndarray,
    f: np.ndarray,
    eliminated_index: int,
    rule: str,
    *,
    w_J: float | None = None,
    w_F: float | None = None,
    robustness_threshold: float = 0.0,
) -> dict[str, bool | float]:
    """
    Score all axioms for one week. Returns dict with keys monotonicity, fan_relevance,
    fan_dominance, robustness_radius, robustness, transparency (if params provided).
    """
    out = {}
    out["monotonicity"] = monotonicity_holds(J, eliminated_index)
    out["fan_dominance"] = fan_dominance_check(J, f, eliminated_index)
    rad = robustness_radius(J, f, rule, eliminated_index)
    out["robustness_radius"] = rad
    out["robustness"] = robustness_axiom(rad, robustness_threshold)
    if w_J is not None and w_F is not None:
        out["fan_relevance"] = fan_relevance_bounds(w_J, w_F)
    return out
