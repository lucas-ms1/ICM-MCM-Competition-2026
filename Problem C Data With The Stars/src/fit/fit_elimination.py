"""
Elimination-order fitting: treat eliminations and finals placement as a likelihood.
G1/G2: elimination_log_prob (percent and rank). G3: Plackett-Luce finals. G4: L-BFGS-B fit.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.scoring import (
    get_rule_for_season,
    percent_combined,
    rank_combined,
)
from src.models.vote_latent import (
    fan_shares_from_beta,
    shares_to_index_totals,
    build_contestant_week_covariates,
)


def elimination_log_prob(
    rule: str,
    combined_scores: np.ndarray,
    e_idx: int,
    tau: float,
) -> float:
    """
    Log probability that contestant e_idx is eliminated.
    Percent: Pr(e_t=i) propto exp(-tau * c_i). Rank: Pr(e_t=i) propto exp(+tau * R_i).
    """
    combined_scores = np.asarray(combined_scores, dtype=float)
    n = len(combined_scores)
    if n == 0 or e_idx < 0 or e_idx >= n:
        return -np.inf
    if tau <= 0:
        tau = 1e-10
    if rule == "percent":
        # lower c => more likely eliminated: log Pr(i) = -tau*c_i - log(sum_k exp(-tau*c_k))
        x = -tau * combined_scores
    else:
        # rank: higher R => more likely eliminated: log Pr(i) = +tau*R_i - log(sum_k exp(tau*R_k))
        x = tau * combined_scores
    x_max = x.max()
    log_z = x_max + np.log(np.exp(x - x_max).sum())
    return float(x[e_idx] - log_z)


def plackett_luce_log_prob(strengths: np.ndarray, order: list[int]) -> float:
    """
    Log probability of observed placement ordering under Plackett-Luce.
    order[0] = 1st, order[1] = 2nd, ...; strengths higher = better.
    """
    strengths = np.asarray(strengths, dtype=float)
    if len(order) == 0:
        return 0.0
    # Numerically stable: subtract max before exp
    s = strengths - strengths.max()
    exp_s = np.exp(s)
    remaining = exp_s.copy()
    log_p = 0.0
    for j, idx in enumerate(order):
        if idx < 0 or idx >= len(strengths):
            return -np.inf
        z = remaining.sum()
        if z <= 0:
            return -np.inf
        log_p += s[idx] - np.log(z)
        remaining[idx] = 0.0
    return float(log_p)


def build_elimination_events(cw: pd.DataFrame) -> list[dict]:
    """
    Build list of elimination events from cw (must have covariates: z_J, p_prev, underdog, etc.).
    Each event: season, week, rule, J, z_J, p_prev, underdog, X, observed_e_idx.
    """
    events = []
    for (season, week), grp in cw.groupby(["season", "week"]):
        active = grp[grp["active"] == 1]
        if len(active) < 2:
            continue
        # Who was eliminated this week? (contestant with elimination_week == week)
        eliminated_mask = active["elimination_week"] == week
        if not eliminated_mask.any():
            continue
        # Exactly one (or take first) eliminated
        e_row = active.loc[eliminated_mask].iloc[0]
        observed_e_idx = int(active.index.get_loc(e_row.name))
        contestant_keys = list(
            zip(
                active["celebrity_name"].values,
                active["ballroom_partner"].values,
            )
        )
        # Covariates for active set (same row order as active)
        J = active["score_week_total"].astype(float).values
        z_J = active["z_J"].astype(float).values if "z_J" in active.columns else np.ones(len(active)) / len(active)
        p_prev = active["p_prev"].astype(float).values if "p_prev" in active.columns else np.zeros(len(active))
        underdog = active["underdog"].astype(float).values if "underdog" in active.columns else np.zeros(len(active))
        X_cols = [c for c in ["age", "industry_dummy"] if c in active.columns]
        X = active[X_cols].astype(float).values if X_cols else np.ones((len(active), 1))
        contestant_id = active["contestant_id"].astype(int).values if "contestant_id" in active.columns else None
        rule = get_rule_for_season(int(season))
        events.append({
            "season": int(season),
            "week": int(week),
            "rule": rule,
            "contestant_keys": contestant_keys,
            "J": J,
            "z_J": z_J,
            "p_prev": p_prev,
            "underdog": underdog,
            "X": X,
            "observed_e_idx": observed_e_idx,
            "observed_eliminated_name": str(e_row.get("celebrity_name", "")),
            "observed_eliminated_partner": str(e_row.get("ballroom_partner", "")),
            "contestant_id": contestant_id,
        })
    return events


def build_finals_events(cw: pd.DataFrame) -> list[dict]:
    """
    Build list of finals events. Each: season, finals_week, rule, J, covariates, placement_order (indices).
    placement_order[0] = index of 1st place, etc.
    """
    events = []
    if "finals_week" not in cw.columns:
        return events
    cw_season = cw["season"].astype(int)
    cw_week = cw["week"].astype(int)
    for season in cw["season"].dropna().unique():
        s = int(season)
        fin = cw[(cw_season == s) & (cw["finals_week"].notna())]
        if fin.empty:
            continue
        fw = int(fin["finals_week"].iloc[0])
        active = cw[(cw_season == s) & (cw_week == fw) & (cw["active"] == 1)]
        if len(active) < 2:
            continue
        # Placement: 1, 2, 3 (numeric from placement column)
        if "placement" not in active.columns:
            continue
        # placement can be numeric 1,2,3 or string "1st Place", "2nd Place"
        placement_series = active["placement"].astype(str).str.extract(r"(\d+)", expand=False)
        placement_vals = pd.to_numeric(placement_series, errors="coerce")
        valid = placement_vals.notna() & (placement_vals >= 1)
        if valid.sum() < 2:
            continue
        place_to_idx = {}
        for i in range(len(active)):
            idx = active.index[i]
            p = placement_vals.loc[idx] if idx in placement_vals.index else np.nan
            if pd.notna(p) and 1 <= p <= 5:
                place_to_idx[int(p)] = i
        places_sorted = sorted(k for k in place_to_idx if 1 <= k <= 3)
        placement_order = [place_to_idx[p] for p in places_sorted]
        if len(placement_order) < 2:
            continue
        J = active["score_week_total"].astype(float).values
        z_J = active["z_J"].astype(float).values if "z_J" in active.columns else np.ones(len(active)) / len(active)
        p_prev = active["p_prev"].astype(float).values if "p_prev" in active.columns else np.zeros(len(active))
        underdog = active["underdog"].astype(float).values if "underdog" in active.columns else np.zeros(len(active))
        X_cols = [c for c in ["age", "industry_dummy"] if c in active.columns]
        X = active[X_cols].astype(float).values if X_cols else np.ones((len(active), 1))
        contestant_id = active["contestant_id"].astype(int).values if "contestant_id" in active.columns else None
        rule = get_rule_for_season(s)
        events.append({
            "season": s,
            "finals_week": fw,
            "rule": rule,
            "J": J,
            "z_J": z_J,
            "p_prev": p_prev,
            "underdog": underdog,
            "X": X,
            "placement_order": placement_order,
            "contestant_id": contestant_id,
        })
    return events


def compute_fit_diagnostics(
    beta: dict,
    tau: float,
    elimination_events: list[dict],
) -> pd.DataFrame:
    """
    Per-(season, week) fit diagnostics: rule_regime, n_active, observed_eliminated,
    model_elim_prob_observed, rank_of_observed_eliminated (1 = most likely), map_matches_observed,
    plus baselines: random (prob=1/n) and judges-only (percent-era: c=j; rank-era: R=r^J).
    """
    rows = []
    for ev in elimination_events:
        n = len(ev["J"])
        J = np.asarray(ev["J"], dtype=float)
        cid = ev.get("contestant_id")
        f = fan_shares_from_beta(
            ev["J"], ev["z_J"], ev["p_prev"], ev["underdog"], ev["X"], beta,
            contestant_id=cid,
        )
        V = shares_to_index_totals(f)
        rule = ev["rule"]
        if rule == "percent":
            combined_scores, _ = percent_combined(ev["J"], V)
        else:
            combined_scores, _ = rank_combined(ev["J"], V)
        # Elimination probability for each contestant (model)
        log_p = np.array(
            [elimination_log_prob(rule, combined_scores, i, tau) for i in range(n)]
        )
        p = np.exp(log_p)
        obs_idx = ev["observed_e_idx"]
        model_log_prob_observed = float(log_p[obs_idx])
        model_elim_prob_observed = float(p[obs_idx])
        # Rank: 1 = most likely eliminated (strictly higher p => better rank)
        rank_of_observed = 1 + int((p > p[obs_idx]).sum())
        map_elim = int(np.argmax(p))
        map_matches_observed = map_elim == obs_idx

        # Random baseline: Pr(elim=i) = 1/n
        random_log_prob_observed = -np.log(n)
        random_elim_prob_observed = 1.0 / n
        random_match_expected = 1.0 / n

        # Judges-only baseline: percent-era c_i = j_i (fan uniform); rank-era R_i = r^J_i (fan uniform)
        V_uniform = np.ones(n)
        if rule == "percent":
            combined_judges_only, _ = percent_combined(J, V_uniform)
        else:
            combined_judges_only, _ = rank_combined(J, V_uniform)
        log_p_jo = np.array(
            [
                elimination_log_prob(rule, combined_judges_only, i, tau)
                for i in range(n)
            ]
        )
        p_jo = np.exp(log_p_jo)
        judges_only_log_prob_observed = float(log_p_jo[obs_idx])
        judges_only_elim_prob_observed = float(p_jo[obs_idx])
        judges_only_map_elim = int(np.argmax(p_jo))
        judges_only_map_matches_observed = judges_only_map_elim == obs_idx

        rows.append({
            "season": ev["season"],
            "week": ev["week"],
            "rule_regime": rule,
            "n_active": n,
            "observed_eliminated": obs_idx,
            "observed_eliminated_name": ev.get("observed_eliminated_name", ""),
            "observed_eliminated_partner": ev.get("observed_eliminated_partner", ""),
            "model_map_eliminated": map_elim,
            "model_elim_prob_observed": model_elim_prob_observed,
            "model_log_prob_observed": model_log_prob_observed,
            "rank_of_observed_eliminated": rank_of_observed,
            "map_matches_observed": map_matches_observed,
            "random_log_prob_observed": float(random_log_prob_observed),
            "random_elim_prob_observed": float(random_elim_prob_observed),
            "random_match_expected": float(random_match_expected),
            "judges_only_log_prob_observed": judges_only_log_prob_observed,
            "judges_only_elim_prob_observed": judges_only_elim_prob_observed,
            "judges_only_map_eliminated": judges_only_map_elim,
            "judges_only_map_matches_observed": judges_only_map_matches_observed,
        })
    return pd.DataFrame(rows)


def neg_log_likelihood(
    beta: dict,
    tau: float,
    elimination_events: list[dict],
    finals_events: list[dict],
    *,
    lambda_ridge: float = 0.0,
) -> float:
    """
    Negative total log-likelihood: eliminations + finals (Plackett-Luce).
    If beta contains "theta", add ridge penalty lambda_ridge * sum(theta^2).
    """
    nll = 0.0
    for ev in elimination_events:
        cid = ev.get("contestant_id")
        f = fan_shares_from_beta(
            ev["J"], ev["z_J"], ev["p_prev"], ev["underdog"], ev["X"], beta,
            contestant_id=cid,
        )
        V = shares_to_index_totals(f)
        if ev["rule"] == "percent":
            c, _ = percent_combined(ev["J"], V)
            log_p = elimination_log_prob("percent", c, ev["observed_e_idx"], tau)
        else:
            R, _ = rank_combined(ev["J"], V)
            log_p = elimination_log_prob("rank", R, ev["observed_e_idx"], tau)
        nll -= log_p
    for ev in finals_events:
        cid = ev.get("contestant_id")
        f = fan_shares_from_beta(
            ev["J"], ev["z_J"], ev["p_prev"], ev["underdog"], ev["X"], beta,
            contestant_id=cid,
        )
        V = shares_to_index_totals(f)
        if ev["rule"] == "percent":
            c, _ = percent_combined(ev["J"], V)
            strengths = c
        else:
            R, _ = rank_combined(ev["J"], V)
            strengths = -R
        log_p = plackett_luce_log_prob(strengths, ev["placement_order"])
        nll -= log_p
    theta = beta.get("theta")
    if theta is not None and lambda_ridge > 0:
        theta = np.asarray(theta, dtype=float)
        nll += lambda_ridge * float(np.sum(theta**2))
    return nll


def _pack_beta(beta: dict, beta_keys: list[str]) -> np.ndarray:
    x = []
    for k in beta_keys:
        v = beta[k]
        if np.shape(v) == ():
            x.append(float(v))
        else:
            x.extend(np.atleast_1d(v).tolist())
    return np.array(x)


def _unpack_beta(x: np.ndarray, beta_keys: list[str], beta_template: dict) -> dict:
    beta = {}
    i = 0
    for k in beta_keys:
        v = beta_template[k]
        if np.shape(v) == ():
            beta[k] = float(x[i])
            i += 1
        else:
            n = len(np.atleast_1d(v))
            beta[k] = np.array(x[i : i + n])
            i += n
    return beta


def fit(
    beta_init: dict,
    tau_init: float,
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    bounds: dict | None = None,
    elimination_events: list[dict] | None = None,
    finals_events: list[dict] | None = None,
    lambda_ridge: float = 0.5,
    n_contestants: int | None = None,
):
    """
    Maximize log-likelihood over beta, theta (contestant baselines), and tau using L-BFGS-B.
    Ridge penalty: lambda_ridge * sum(theta^2). cw must have covariates and contestant_id
    (run build_contestant_week_covariates(cw, raw) first).
    n_contestants: if provided (e.g. from full cw for bootstrap), theta has this length; else from cw.
    Returns beta_opt (with key "theta"), tau_opt, scipy result.
    """
    if elimination_events is None:
        elimination_events = build_elimination_events(cw)
    if finals_events is None:
        finals_events = build_finals_events(cw)
    beta_keys = ["beta0", "beta_J", "beta_P", "beta_U"]
    if "beta_X" in beta_init:
        beta_keys.append("beta_X")
    n_theta = n_contestants
    if n_theta is None and "contestant_id" in cw.columns:
        n_theta = int(cw["contestant_id"].max()) + 1
    if n_theta is None or n_theta <= 0:
        n_theta = 0
    n_beta = len(_pack_beta(beta_init, beta_keys))
    x0 = np.concatenate([
        _pack_beta(beta_init, beta_keys),
        np.zeros(n_theta),
        [tau_init],
    ])

    def objective(x: np.ndarray) -> float:
        beta = _unpack_beta(x[:n_beta], beta_keys, beta_init)
        theta = x[n_beta : n_beta + n_theta].copy() if n_theta > 0 else np.array([])
        tau = float(x[-1])
        if tau <= 0:
            tau = 1e-10
        beta_with_theta = {**beta, "theta": theta} if n_theta > 0 else beta
        return neg_log_likelihood(
            beta_with_theta, tau, elimination_events, finals_events,
            lambda_ridge=lambda_ridge,
        )

    # Bounds: beta and theta unbounded, tau > 0
    n_x = len(x0)
    if bounds is None:
        bounds_list = [(None, None)] * (n_x - 1) + [(1e-10, None)]
    else:
        beta_lb = bounds.get("beta_lb", None)
        beta_ub = bounds.get("beta_ub", None)
        tau_lb = bounds.get("tau_lb", 1e-10)
        tau_ub = bounds.get("tau_ub", None)
        bounds_list = [(beta_lb, beta_ub)] * (n_x - 1) + [(tau_lb, tau_ub)]
    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds_list,
        options={"maxfun": 60000},
    )
    x_opt = result.x
    beta_opt = _unpack_beta(x_opt[:n_beta], beta_keys, beta_init)
    if n_theta > 0:
        beta_opt["theta"] = x_opt[n_beta : n_beta + n_theta].copy()
    tau_opt = float(x_opt[-1])
    return beta_opt, tau_opt, result


def fit_multistart(
    beta_init: dict,
    tau_inits: list[float],
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    bounds: dict | None = None,
    elimination_events: list[dict] | None = None,
    finals_events: list[dict] | None = None,
    lambda_ridge: float = 0.5,
    n_contestants: int | None = None,
):
    """
    Try multiple tau initializations and keep the best successful fit.
    Returns (beta_opt, tau_opt, result). If no successful fit, returns best by objective.
    """
    best = None
    best_success = None
    for tau_init in tau_inits:
        try:
            beta_opt, tau_opt, result = fit(
                beta_init,
                tau_init,
                cw,
                raw,
                bounds=bounds,
                elimination_events=elimination_events,
                finals_events=finals_events,
                lambda_ridge=lambda_ridge,
                n_contestants=n_contestants,
            )
        except Exception:
            continue
        if best is None or result.fun < best[2].fun:
            best = (beta_opt, tau_opt, result)
        if result.success:
            if best_success is None or result.fun < best_success[2].fun:
                best_success = (beta_opt, tau_opt, result)
    if best_success is not None:
        return best_success
    if best is not None:
        return best
    raise RuntimeError("fit_multistart failed: no fits completed.")


def fitted_params_to_dict(
    beta_opt: dict,
    tau_opt: float,
    result,
) -> dict:
    """
    Build a JSON-serializable dict for reproducibility: named beta, tau, theta summary, optimizer status.
    beta_X (length-2: age, industry_dummy) is serialized as beta_age, beta_actor.
    theta: summary (mean, std, n_contestants) and full list for reproducibility.
    """
    beta_names = ["intercept", "beta_J", "beta_P", "beta_U"]
    beta_src = ["beta0", "beta_J", "beta_P", "beta_U"]
    beta_out = {}
    for name, key in zip(beta_names, beta_src):
        if key in beta_opt:
            v = beta_opt[key]
            beta_out[name] = float(np.asarray(v).flat[0])
    if "beta_X" in beta_opt:
        bx = np.atleast_1d(beta_opt["beta_X"])
        if len(bx) >= 1:
            beta_out["beta_age"] = float(bx[0])
        if len(bx) >= 2:
            beta_out["beta_actor"] = float(bx[1])
    out = {
        "beta": beta_out,
        "tau": float(tau_opt),
        "optimizer": {
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "final_objective": float(result.fun),
        },
    }
    if "theta" in beta_opt:
        th = np.asarray(beta_opt["theta"], dtype=float)
        out["theta_summary"] = {
            "n_contestants": int(len(th)),
            "mean": float(np.mean(th)),
            "std": float(np.std(th)) if len(th) > 1 else 0.0,
        }
        out["theta"] = th.tolist()
    return out
