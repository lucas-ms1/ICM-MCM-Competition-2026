"""
Latent voting model: preference scores u, softmax fan shares f, and index-scaled totals.
Layer 1: forward map β, covariates -> f (and optionally V for reporting).
"""

import numpy as np
import pandas as pd

# Default index scale for "vote totals" (reporting only; label as index-scaled)
DEFAULT_INDEX_TOTAL = 10_000_000


def shares_to_index_totals(
    f: np.ndarray,
    total: float = DEFAULT_INDEX_TOTAL,
) -> np.ndarray:
    """
    Convert fan vote shares to index-scaled totals for reporting only.
    Returns V such that V.sum() == total and V / V.sum() == f (i.e. V = f * total).
    Label any such output as "index-scaled total" in reports; not true vote count.
    """
    f = np.asarray(f, dtype=float)
    if np.any(f < 0) or not np.isclose(f.sum(), 1.0):
        raise ValueError("f must be non-negative and sum to 1")
    return f * total


def shares_from_preference(u: np.ndarray) -> np.ndarray:
    """
    Softmax over active set: f_i = exp(u_i) / sum_k exp(u_k).
    Numerically stable: subtract max(u) before exp.
    """
    u = np.asarray(u, dtype=float)
    u = u - u.max()
    exp_u = np.exp(u)
    return exp_u / exp_u.sum()


def preference_scores(
    z_J: np.ndarray,
    p_prev: np.ndarray,
    underdog: np.ndarray,
    X: np.ndarray | None,
    beta: dict,
) -> np.ndarray:
    """
    u_{i,t} = beta0 + beta_J*z_J + beta_X'*X + beta_P*p_prev + beta_U*underdog.
    z_J, p_prev, underdog: 1d arrays (length n = active set size).
    X: optional n x k matrix; if None, no X term.
    beta: dict with keys beta0, beta_J, beta_P, beta_U, and optionally beta_X (length k).
    """
    n = len(z_J)
    u = np.full(n, float(beta["beta0"]))
    u += float(beta["beta_J"]) * np.asarray(z_J, dtype=float)
    u += float(beta["beta_P"]) * np.asarray(p_prev, dtype=float)
    u += float(beta["beta_U"]) * np.asarray(underdog, dtype=float)
    if X is not None and "beta_X" in beta:
        X = np.asarray(X)
        beta_X = np.atleast_1d(beta["beta_X"])
        u += np.dot(X, beta_X)
    return u


def fan_shares_from_beta(
    J: np.ndarray,
    z_J: np.ndarray,
    p_prev: np.ndarray,
    underdog: np.ndarray,
    X: np.ndarray | None,
    beta: dict,
) -> np.ndarray:
    """
    Combined forward map: u = preference_scores(...), f = shares_from_preference(u).
    J is unused in u but can be used by caller to compute z_J; we take z_J precomputed.
    Returns f (fan shares) for the active set.
    """
    u = preference_scores(z_J, p_prev, underdog, X, beta)
    return shares_from_preference(u)


def build_contestant_week_covariates(cw: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build covariates for the preference model: z_J, age, industry_dummy, p_prev, underdog.
    Adds columns to a copy of cw; align by (season, celebrity_name, ballroom_partner).
    - z_J: J / max(J) within (season, week) among active; 0 if max(J)==0.
    - age: celebrity_age_during_season from raw (fill missing with 0).
    - industry_dummy: 1 if celebrity_industry == "Actor/Actress", else 0.
    - p_prev: rank by J in week t-1 (1 = best); week 1 = 0.
    - underdog: 1 if J <= median J in that (season, week) among active, else 0.
    """
    cw = cw.copy()
    # Merge raw demographics (minimal: age, industry_dummy)
    demo_cols = ["season", "celebrity_name", "ballroom_partner"]
    if "celebrity_age_during_season" in raw.columns:
        demo_cols.append("celebrity_age_during_season")
    if "celebrity_industry" in raw.columns:
        demo_cols.append("celebrity_industry")
    demo = raw[demo_cols].drop_duplicates()
    cw = cw.merge(demo, on=["season", "celebrity_name", "ballroom_partner"], how="left")
    cw["age"] = pd.to_numeric(cw["celebrity_age_during_season"], errors="coerce").fillna(0).values if "celebrity_age_during_season" in cw.columns else 0.0
    cw["industry_dummy"] = (cw["celebrity_industry"].astype(str).str.strip().str.lower() == "actor/actress").astype(int).values if "celebrity_industry" in cw.columns else 0
    cw = cw.drop(columns=[c for c in ["celebrity_age_during_season", "celebrity_industry"] if c in cw.columns], errors="ignore")

    # z_J: within (season, week), z = J / max(J); 0 if max(J)==0
    cw["z_J"] = np.nan
    for (season, week), grp in cw.groupby(["season", "week"]):
        j = grp["score_week_total"].astype(float).values
        m = grp["active"].astype(bool).values
        j_active = np.where(m, j, np.nan)
        mx = np.nanmax(j_active) if np.any(m) else 1.0
        if mx == 0:
            mx = 1.0
        cw.loc[grp.index, "z_J"] = j / mx

    # p_prev: rank by J in previous week (1 = best); week 1 = 0
    cw["p_prev"] = 0.0
    for (season, week), grp in cw.groupby(["season", "week"]):
        if week == 1:
            continue
        prev = cw[(cw["season"] == season) & (cw["week"] == week - 1)]
        if prev.empty:
            continue
        j_prev = prev.set_index(["celebrity_name", "ballroom_partner"])["score_week_total"]
        rank_prev = j_prev.rank(ascending=False, method="average")
        for idx in grp.index:
            nm = cw.loc[idx, "celebrity_name"]
            bp = cw.loc[idx, "ballroom_partner"]
            if (nm, bp) in rank_prev.index:
                cw.loc[idx, "p_prev"] = rank_prev.loc[(nm, bp)]

    # underdog: 1 if J <= median J in (season, week) among active
    cw["underdog"] = 0.0
    for (season, week), grp in cw.groupby(["season", "week"]):
        j = grp["score_week_total"].values.astype(float)
        m = grp["active"].values.astype(bool)
        if not np.any(m):
            continue
        med = np.median(j[m])
        cw.loc[grp.index, "underdog"] = (j <= med).astype(float)

    return cw
