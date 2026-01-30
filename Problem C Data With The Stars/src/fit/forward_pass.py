"""
Week-by-week forward pass: given fitted beta and cw with covariates,
compute fan shares f_{i,t} and V for every (season, week) with active set.
Used by bootstrap (H2), margin-to-flip (H1), counterfactual engine (I), controversy cases (I3).
Optional meanfield+memory path: forward_pass_meanfield_week_by_week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import get_rule_for_season
from src.models.vote_latent import (
    fan_shares_from_beta,
    shares_to_index_totals,
)
from src.models.meanfield import (
    meanfield_update,
    update_memory_state,
)


def forward_pass_week_by_week(
    cw: pd.DataFrame,
    beta: dict,
) -> list[dict]:
    """
    Compute fan shares f and V for every (season, week) with at least 2 active contestants.
    cw must have covariates: z_J, p_prev, underdog, and optionally age, industry_dummy (for X).
    p_prev in cw is assumed to come from build_contestant_week_covariates (rank by J in week t-1).

    Returns list of dicts, one per (season, week) elimination-relevant week:
      - season, week: int
      - rule: "percent" | "rank"
      - contestant_keys: list of (celebrity_name, ballroom_partner) in active order
      - J: (n,) judge totals
      - f: (n,) fan shares
      - V: (n,) index-scaled vote totals (f * default total)
      - observed_eliminated_index: int index in active set of who was eliminated this week, or None
    """
    cw = cw.sort_values(["season", "week"])
    out = []
    for (season, week), grp in cw.groupby(["season", "week"], sort=True):
        active = grp[grp["active"] == 1]
        if len(active) < 2:
            continue
        J = active["score_week_total"].astype(float).values
        z_J = (
            active["z_J"].astype(float).values
            if "z_J" in active.columns
            else np.ones(len(active)) / len(active)
        )
        p_prev = (
            active["p_prev"].astype(float).values
            if "p_prev" in active.columns
            else np.zeros(len(active))
        )
        underdog = (
            active["underdog"].astype(float).values
            if "underdog" in active.columns
            else np.zeros(len(active))
        )
        X_cols = [c for c in ["age", "industry_dummy"] if c in active.columns]
        X = (
            active[X_cols].astype(float).values
            if X_cols
            else np.ones((len(active), 1))
        )
        f = fan_shares_from_beta(J, z_J, p_prev, underdog, X, beta)
        V = shares_to_index_totals(f)
        contestant_keys = list(
            zip(
                active["celebrity_name"].values,
                active["ballroom_partner"].values,
            )
        )
        eliminated_mask = active["elimination_week"] == week
        if eliminated_mask.any():
            e_row = active.loc[eliminated_mask].iloc[0]
            observed_eliminated_index = int(active.index.get_loc(e_row.name))
        else:
            observed_eliminated_index = None
        rule = get_rule_for_season(int(season))
        out.append({
            "season": int(season),
            "week": int(week),
            "rule": rule,
            "contestant_keys": contestant_keys,
            "J": J,
            "f": f,
            "V": V,
            "observed_eliminated_index": observed_eliminated_index,
        })
    return out


def _align_history_to_keys(
    history_list: list[tuple[list[tuple], np.ndarray, np.ndarray]],
    current_keys: list[tuple],
    current_n: int,
    fill: float = 0.0,
    value_index: int = 1,
) -> np.ndarray:
    """
    Build (T+1, current_n) array from list of (keys_w, S_w, p_w) so that
    out[tau, i] = value at week tau for contestant current_keys[i] (fill if not present).
    value_index: 1 = use S_w, 2 = use p_w.
    """
    T_plus_1 = len(history_list)
    out = np.full((T_plus_1, current_n), fill, dtype=float)
    for tau in range(T_plus_1):
        keys_w, S_w, p_w = history_list[tau]
        arr_w = (S_w, p_w)[value_index - 1]
        key_to_idx = {k: j for j, k in enumerate(keys_w)}
        for i, k in enumerate(current_keys):
            if k in key_to_idx:
                out[tau, i] = arr_w[key_to_idx[k]]
    return out


def forward_pass_meanfield_week_by_week(
    cw: pd.DataFrame,
    params: dict,
    *,
    memory_params: dict | None = None,
    underdog_beta_U: float = 0.0,
) -> list[dict]:
    """
    Meanfield+memory forward pass: week-by-week meanfield_update with optional
    S_history, p_history, and Markovian m_t. Maintains history and m per season;
    returns same event structure as forward_pass_week_by_week (contestant_keys, J, f, V, etc.).

    Parameters
    ----------
    cw : DataFrame
        Must have season, week, active, score_week_total, z_J (or computed), and optionally
        underdog, age, industry_dummy. Can use build_contestant_week_covariates for z_J, p_prev, underdog.
    params : dict
        Meanfield params: kappa, eta, gamma, epsilon; optionally theta (length k for X).
    memory_params : dict or None
        If provided: kernel ("exponential"|"rectangular"|"power_law"), eta_S, gamma_history,
        eta_m, memory_lambda (lam in (0,1)); and decay or L/d for kernel. If None, no memory.
    underdog_beta_U : float
        beta_U for underdog term; underdog vector from cw if column present.

    Returns
    -------
    events : list[dict]
        Same structure as forward_pass_week_by_week: season, week, rule, contestant_keys,
        J, f, V, observed_eliminated_index.
    """
    cw = cw.sort_values(["season", "week"])
    out = []
    # Per-season state: list of (keys_w, S_w, p_w), and m_prev dict (key -> float)
    season_history: list[tuple[list, np.ndarray, np.ndarray]] = []
    m_prev: dict[tuple, float] = {}
    last_season: int | None = None

    lam = None
    eta_m = 0.0
    if memory_params is not None:
        lam = memory_params.get("memory_lambda")
        eta_m = float(memory_params.get("eta_m", 0.0))

    for (season, week), grp in cw.groupby(["season", "week"], sort=True):
        active = grp[grp["active"] == 1]
        if len(active) < 2:
            continue

        # New season: reset history and m
        if last_season is not None and season != last_season:
            season_history = []
            m_prev = {}
        last_season = season

        J = active["score_week_total"].astype(float).values
        z_J = (
            active["z_J"].astype(float).values
            if "z_J" in active.columns
            else np.ones(len(active)) / len(active)
        )
        S_t = z_J  # judge signal
        underdog = (
            active["underdog"].astype(float).values
            if "underdog" in active.columns
            else np.zeros(len(active))
        )
        X_cols = [c for c in ["age", "industry_dummy"] if c in active.columns]
        X = (
            active[X_cols].astype(float).values
            if X_cols
            else None
        )
        if X is not None and "theta" in params:
            pass
        else:
            X = None
        contestant_keys = list(
            zip(
                active["celebrity_name"].values,
                active["ballroom_partner"].values,
            )
        )
        n = len(contestant_keys)
        t = len(season_history)  # 0-indexed week within season

        # p_t: from previous week or uniform
        if t == 0:
            p_t = np.ones(n) / n
        else:
            _, _, p_last = season_history[-1]
            # Align: current keys vs last week keys
            p_t = np.ones(n) / n
            last_keys, _, p_last_arr = season_history[-1]
            key_to_idx_last = {k: j for j, k in enumerate(last_keys)}
            for i, k in enumerate(contestant_keys):
                if k in key_to_idx_last:
                    p_t[i] = p_last_arr[key_to_idx_last[k]]
            p_t = p_t / p_t.sum()

        # Build S_history, p_history (t+1, n) aligned to current keys
        S_history = None
        p_history = None
        current_t = None
        m_t = None
        if memory_params is not None and (eta_m != 0 or memory_params.get("eta_S", 0) != 0 or memory_params.get("gamma_history", 0) != 0):
            current_t = t
            if season_history:
                S_history = _align_history_to_keys(
                    season_history, contestant_keys, n, fill=0.0, value_index=1,
                )
                p_history = _align_history_to_keys(
                    season_history, contestant_keys, n, fill=1.0 / max(1, n), value_index=2,
                )
                # Append current (S_t, p_t) so history has t+1 rows
                S_history = np.vstack([S_history, S_t.reshape(1, -1)])
                p_history = np.vstack([p_history, p_t.reshape(1, -1)])
            else:
                S_history = S_t.reshape(1, -1)
                p_history = p_t.reshape(1, -1)

        if eta_m != 0 and lam is not None and 0 < lam < 1:
            if t == 0:
                m_prev = {k: float(S_t[i]) for i, k in enumerate(contestant_keys)}
            m_t = np.array([m_prev.get(k, 0.0) for k in contestant_keys], dtype=float)
        else:
            m_t = None

        p_next = meanfield_update(
            p_t, S_t, X, params,
            underdog=underdog, beta_U=underdog_beta_U,
            S_history=S_history, p_history=p_history,
            memory_params=memory_params, current_t=current_t,
            m_t=m_t, eta_m=eta_m, memory_lambda=lam,
        )
        f = p_next
        V = shares_to_index_totals(f)

        if eta_m != 0 and lam is not None and 0 < lam < 1:
            for i, k in enumerate(contestant_keys):
                m_prev[k] = update_memory_state(
                    np.array([m_prev[k]], dtype=float),
                    np.array([S_t[i]], dtype=float),
                    lam,
                )[0]

        season_history.append((contestant_keys, S_t.copy(), p_next.copy()))

        eliminated_mask = active["elimination_week"] == week
        if eliminated_mask.any():
            e_row = active.loc[eliminated_mask].iloc[0]
            observed_eliminated_index = int(active.index.get_loc(e_row.name))
        else:
            observed_eliminated_index = None
        rule = get_rule_for_season(int(season))
        out.append({
            "season": int(season),
            "week": int(week),
            "rule": rule,
            "contestant_keys": contestant_keys,
            "J": J,
            "f": f,
            "V": V,
            "observed_eliminated_index": observed_eliminated_index,
        })
    return out


def forward_pass_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """
    Convert forward_pass_week_by_week output to a long DataFrame with one row per
    (season, week, contestant): season, week, celebrity_name, ballroom_partner,
    J, f, V, is_eliminated_this_week.
    """
    rows = []
    for ev in events:
        for i, (name, partner) in enumerate(ev["contestant_keys"]):
            rows.append({
                "season": ev["season"],
                "week": ev["week"],
                "celebrity_name": name,
                "ballroom_partner": partner,
                "J": ev["J"][i],
                "f": ev["f"][i],
                "V": ev["V"][i],
                "is_eliminated_this_week": ev["observed_eliminated_index"] == i,
            })
    return pd.DataFrame(rows)
