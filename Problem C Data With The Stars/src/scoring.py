"""
DWTS combination rules as deterministic scoring functions.
Core utilities for fitting and counterfactuals; operate on one week with active set A.
Inputs: J (judge total), V (latent fan vote); V supplied by caller.
"""

import numpy as np
from scipy.stats import rankdata

# Season-rule mapping (tunable)
PERCENT_SEASONS = range(3, 28)   # 3--27 inclusive
RANK_SEASONS = [1, 2] + list(range(28, 35))  # 1, 2, 28--34
JUDGES_SAVE_FROM_SEASON = 28


def get_rule_for_season(season: int) -> str:
    """Return 'percent' or 'rank' for the given season."""
    s = int(season)
    if s in PERCENT_SEASONS:
        return "percent"
    if s in RANK_SEASONS:
        return "rank"
    # Fallback: treat unknown seasons as rank (e.g. future seasons)
    return "rank"


def use_judges_save(season: int) -> bool:
    """True if season uses judges save (bottom two, judges pick)."""
    return int(season) >= JUDGES_SAVE_FROM_SEASON


def percent_combined(J: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Percent method: j_i = J_i / sum(J), f_i = V_i / sum(V), c_i = j_i + f_i.
    Eliminated = argmin c_i (lowest combined percent leaves).
    J, V are 1d arrays for active set only (same length).
    Edge: if sum(J)==0 use uniform j_i=1/n; if sum(V)==0 use uniform f_i=1/n.
    Tie: first index with minimum c.
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(J)
    if n == 0:
        return np.array([]), -1

    sum_j = J.sum()
    sum_v = V.sum()
    if sum_j == 0:
        j = np.ones(n) / n
    else:
        j = J / sum_j
    if sum_v == 0:
        f = np.ones(n) / n
    else:
        f = V / sum_v
    c = j + f
    eliminated_index = int(np.argmin(c))
    return c, eliminated_index


def rank_combined(J: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Rank method: r^J by J (best=1), r^F by V (best=1), R_i = r^J_i + r^F_i.
    Eliminated = argmax R_i (worst rank sum leaves).
    Ties: average ranks (scipy.stats.rankdata method='average').
    Higher J / V = better = rank 1, so we rank by -J and -V (ascending rank = smallest gets 1).
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(J)
    if n == 0:
        return np.array([]), -1

    # Higher score = better = rank 1: rank ascending so smallest gets 1 -> use -J, -V
    r_j = rankdata(-J, method="average")
    r_f = rankdata(-V, method="average")
    R = r_j + r_f
    eliminated_index = int(np.argmax(R))
    return R, eliminated_index


def bottom_two_indices(combined_scores: np.ndarray) -> tuple[int, int]:
    """
    Indices of the two contestants with smallest combined score (bottom two).
    For rank method, combined_scores is R (higher = worse), so bottom two = two with largest R.
    Tie: first two by order (argsort is stable).
    """
    arr = np.asarray(combined_scores, dtype=float)
    if len(arr) < 2:
        raise ValueError("Need at least 2 contestants for bottom two")
    # Largest two: argsort ascending, so last two indices are max and second-max
    idx = np.argsort(arr)[-2:]
    return int(idx[0]), int(idx[1])


def judges_save_deterministic(i: int, k: int, J_i: float, J_k: float) -> int:
    """
    Among bottom two (i, k), return the index of the contestant eliminated (lower J).
    Judges save the higher-scoring contestant.
    """
    return i if J_i < J_k else k


def judges_save_probability_eliminate_i(J_i: float, J_k: float, alpha: float) -> float:
    """
    P(eliminate i | i, k bottom 2) = sigma(alpha * (J_k - J_i)).
    sigma(x) = 1 / (1 + exp(-x)).
    """
    x = alpha * (float(J_k) - float(J_i))
    return 1.0 / (1.0 + np.exp(-x))


def week_elimination_counterfactuals(
    J: np.ndarray,
    V: np.ndarray,
    *,
    use_judges_save: bool = False,
) -> dict:
    """
    Compute elimination under rank, percent, and (optionally) bottom-two + judges save
    for the same (J, V). Used to compare rules on the same vote state.
    Returns dict with elim_rank, elim_percent, elim_judges_save (if use_judges_save),
    bottom_two_indices (if use_judges_save).
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(J)
    if n == 0:
        return {
            "elim_rank": -1,
            "elim_percent": -1,
            "elim_judges_save": -1,
            "bottom_two_indices": None,
        }
    _, elim_percent = percent_combined(J, V)
    R, elim_rank = rank_combined(J, V)
    out = {
        "elim_rank": elim_rank,
        "elim_percent": elim_percent,
        "elim_judges_save": -1,
        "bottom_two_indices": None,
    }
    if use_judges_save and n >= 2:
        i, k = bottom_two_indices(R)
        elim_js = judges_save_deterministic(i, k, J[i], J[k])
        out["elim_judges_save"] = elim_js
        out["bottom_two_indices"] = (i, k)
    return out


def weighted_percent_combined(
    J: np.ndarray,
    V: np.ndarray,
    w_J: float,
    w_F: float,
) -> tuple[np.ndarray, int]:
    """
    Generalized weighted percent: c_i = w_J * j_i + w_F * f_i with j_i = J_i/sum(J), f_i = V_i/sum(V).
    Eliminated = argmin c_i. Used for sensitivity plots (vary w_J, w_F).
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(J)
    if n == 0:
        return np.array([]), -1
    sum_j = J.sum()
    sum_v = V.sum()
    if sum_j == 0:
        j = np.ones(n) / n
    else:
        j = J / sum_j
    if sum_v == 0:
        f = np.ones(n) / n
    else:
        f = V / sum_v
    c = w_J * j + w_F * f
    eliminated_index = int(np.argmin(c))
    return c, eliminated_index


# ---- Proposed systems (K2): weighted percent with saturation, percent + judges-save trigger ----


def softcap_fan_share(
    f: np.ndarray,
    method: str = "power",
    *,
    alpha: float = 0.8,
    scale: float = 1.0,
    center: float = 0.5,
) -> np.ndarray:
    """
    Softcap fan shares so extreme blocs do not fully dominate.
    Returns a probability vector (sum 1). Methods:
    - "power": f_capped = f^alpha then renormalize (alpha in (0, 1]).
    - "logit_cap": logit(f_capped) = scale * (logit(f) - center), then renormalize.
    """
    f = np.asarray(f, dtype=float)
    n = len(f)
    if n == 0:
        return f
    f = np.clip(f, 1e-15, 1.0)
    if method == "power":
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1:
            alpha = 0.8
        out = np.power(f, alpha)
    elif method == "logit_cap":
        # logit(p) = log(p/(1-p)); inverse: p = 1/(1+exp(-x))
        logit_f = np.log(f / (1.0 - f + 1e-15))
        x = scale * (logit_f - center)
        x = np.clip(x, -500, 500)
        out = 1.0 / (1.0 + np.exp(-x))
    else:
        out = f.copy()
    out = np.clip(out, 1e-15, None)
    return out / out.sum()


def weighted_percent_with_saturation(
    J: np.ndarray,
    V: np.ndarray,
    w: float,
    softcap_method: str = "power",
    **kwargs,
) -> tuple[np.ndarray, int]:
    """
    Proposed rule (Option A): c_i = w * (J_i/sum J) + (1-w) * softcap(f_i).
    Eliminated = argmin c_i. w is judge weight; softcap reduces fan domination.
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(J)
    if n == 0:
        return np.array([]), -1
    sum_j = J.sum()
    sum_v = V.sum()
    if sum_j == 0:
        j = np.ones(n) / n
    else:
        j = J / sum_j
    if sum_v == 0:
        f = np.ones(n) / n
    else:
        f = V / sum_v
    f_cap = softcap_fan_share(f, method=softcap_method, **kwargs)
    c = w * j + (1.0 - w) * f_cap
    eliminated_index = int(np.argmin(c))
    return c, eliminated_index


def percent_bottom_two(J: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Percent rule combined score and indices of the two contestants with
    smallest combined score (bottom two). c_i = j_i + f_i; bottom two = two smallest c.
    Returns (c, idx1, idx2) with idx1, idx2 the two indices (order by c: c[idx1] <= c[idx2]).
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    n = len(J)
    if n < 2:
        raise ValueError("Need at least 2 contestants for bottom two")
    sum_j = J.sum()
    sum_v = V.sum()
    if sum_j == 0:
        j = np.ones(n) / n
    else:
        j = J / sum_j
    if sum_v == 0:
        f = np.ones(n) / n
    else:
        f = V / sum_v
    c = j + f
    order = np.argsort(c)
    idx1, idx2 = int(order[0]), int(order[1])
    return c, idx1, idx2


def judges_save_trigger(
    J: np.ndarray,
    c: np.ndarray,
    bottom_two: tuple[int, int],
    threshold: float,
) -> int:
    """
    Proposed rule (Option B): among bottom two (i, k), eliminate the one with lower J
    only if J_high - J_low > threshold (judges save); else eliminate the one with
    lower combined score c (keeps fans relevant).
    """
    i, k = bottom_two
    J_i = float(J[i])
    J_k = float(J[k])
    gap = abs(J_i - J_k)
    if gap > threshold:
        return i if J_i < J_k else k
    # Else: eliminate the one with higher c (worse combined score)
    return i if c[i] >= c[k] else k


def percent_with_judges_save_trigger(
    J: np.ndarray,
    V: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    """
    Full Option B: percent picks bottom two; judges save only if judge gap > threshold.
    Returns (c, eliminated_index).
    """
    c, idx1, idx2 = percent_bottom_two(J, V)
    elim = judges_save_trigger(J, c, (idx1, idx2), threshold)
    return c, elim


def week_elimination(
    season: int,
    J: np.ndarray,
    V: np.ndarray,
    *,
    rule_override: str | None = None,
    use_judges_save_override: bool | None = None,
    alpha: float | None = None,
) -> dict:
    """
    Single entry point for one-week elimination given (season, J, V) for active set.
    J, V: 1d arrays, same length and order.
    rule_override: if not None, use "percent" or "rank" instead of get_rule_for_season(season).
    use_judges_save_override: if None, use use_judges_save(season); else override.
    alpha: for probabilistic judges save; not used here (deterministic elimination returned).
    Returns dict: rule, combined, eliminated_index, bottom_two_indices (or None).
    """
    J = np.asarray(J, dtype=float)
    V = np.asarray(V, dtype=float)
    rule = (
        rule_override
        if rule_override is not None
        else get_rule_for_season(season)
    )
    use_js = use_judges_save_override if use_judges_save_override is not None else use_judges_save(season)

    if rule == "percent":
        c, elim = percent_combined(J, V)
        return {
            "rule": "percent",
            "combined": c,
            "eliminated_index": elim,
            "bottom_two_indices": None,
        }

    # rule == "rank"
    R, elim_by_rank = rank_combined(J, V)
    if not use_js:
        return {
            "rule": "rank",
            "combined": R,
            "eliminated_index": elim_by_rank,
            "bottom_two_indices": None,
        }

    # Judges save: bottom two by R (worst two), then deterministic pick by lower J
    i, k = bottom_two_indices(R)  # R is rank sum, higher = worse, so bottom two = two largest R
    elim = judges_save_deterministic(i, k, J[i], J[k])
    return {
        "rule": "rank",
        "combined": R,
        "eliminated_index": elim,
        "bottom_two_indices": (i, k),
    }
