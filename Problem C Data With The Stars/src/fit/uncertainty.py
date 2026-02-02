"""
Uncertainty quantification: bootstrap over seasons, optional Bayesian (PyMC).
H2: Bootstrap intervals for vote shares f_{i,t}: 5--95% per (season, week, contestant).
"""

import numpy as np
import pandas as pd

from src.models.vote_latent import build_contestant_week_covariates
from src.fit.fit_elimination import (
    build_elimination_events,
    build_finals_events,
    fit,
)
from src.fit.forward_pass import forward_pass_week_by_week


def bootstrap_seasons(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    beta_init: dict,
    tau_init: float,
    n_bootstrap: int = 50,
    *,
    random_state: int | None = None,
):
    """
    Resample seasons with replacement; refit each bootstrap sample.
    Returns list of (beta_opt, tau_opt) and optional summary (mean, percentiles).
    """
    rng = np.random.default_rng(random_state)
    seasons = cw["season"].dropna().unique()
    if len(seasons) < 2:
        return [], {}
    results = []
    for _ in range(n_bootstrap):
        boot_seasons = rng.choice(seasons, size=len(seasons), replace=True)
        cw_boot = cw[cw["season"].isin(boot_seasons)].copy()
        if cw_boot.empty or len(cw_boot["season"].unique()) < 2:
            continue
        elim = build_elimination_events(cw_boot)
        finals = build_finals_events(cw_boot)
        if len(elim) < 2:
            continue
        try:
            beta_opt, tau_opt, result = fit(
                beta_init, tau_init, cw_boot, raw,
                elimination_events=elim,
                finals_events=finals,
            )
            if result.success:
                results.append((beta_opt, tau_opt))
        except Exception:
            continue
    if not results:
        return results, {}
    betas = results
    tau_list = [t for _, t in betas]
    keys = list(beta_init.keys())
    summary = {
        "tau_mean": float(np.mean(tau_list)),
        "tau_2.5": float(np.percentile(tau_list, 2.5)),
        "tau_97.5": float(np.percentile(tau_list, 97.5)),
    }
    for k in keys:
        v0 = beta_init[k]
        if np.shape(v0) == ():
            arr = np.array([b[k] for b, _ in betas])
            summary[f"{k}_mean"] = float(np.mean(arr))
            summary[f"{k}_2.5"] = float(np.percentile(arr, 2.5))
            summary[f"{k}_97.5"] = float(np.percentile(arr, 97.5))
        else:
            arr = np.array([b[k] for b, _ in betas])
            summary[f"{k}_mean"] = np.mean(arr, axis=0)
            summary[f"{k}_2.5"] = np.percentile(arr, 2.5, axis=0)
            summary[f"{k}_97.5"] = np.percentile(arr, 97.5, axis=0)
    return results, summary


def bootstrap_fan_share_intervals(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    beta_init: dict,
    tau_init: float,
    n_bootstrap: int = 50,
    *,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[pd.DataFrame, list[tuple[dict, float]]]:
    """
    Resample seasons with replacement; refit each bootstrap sample; then run
    forward pass on the **full** cw with each bootstrap beta to get f_{i,t}
    for every (season, week, contestant). Compute percentile intervals across runs.

    cw must already have covariates (run build_contestant_week_covariates(cw, raw) first).

    Returns
    -------
    intervals_df : DataFrame with columns season, week, celebrity_name, ballroom_partner,
        f_mean, f_lower, f_upper (and optionally J, f_median).
    bootstrap_results : list of (beta_opt, tau_opt) that succeeded (same as bootstrap_seasons).
    """
    rng = np.random.default_rng(random_state)
    seasons = cw["season"].dropna().unique()
    if len(seasons) < 2:
        return pd.DataFrame(), []
    n_contestants = int(cw["contestant_id"].max()) + 1 if "contestant_id" in cw.columns else None
    p_lo, p_hi = percentiles
    # Collect per (season, week, celebrity_name, ballroom_partner) list of f across runs
    store: dict[tuple, list[float]] = {}
    bootstrap_results: list[tuple[dict, float]] = []
    for _ in range(n_bootstrap):
        boot_seasons = rng.choice(seasons, size=len(seasons), replace=True)
        cw_boot = cw[cw["season"].isin(boot_seasons)].copy()
        if cw_boot.empty or len(cw_boot["season"].unique()) < 2:
            continue
        elim = build_elimination_events(cw_boot)
        finals = build_finals_events(cw_boot)
        if len(elim) < 2:
            continue
        try:
            beta_opt, tau_opt, result = fit(
                beta_init, tau_init, cw_boot, raw,
                elimination_events=elim,
                finals_events=finals,
                n_contestants=n_contestants,
            )
            if not result.success:
                continue
            bootstrap_results.append((beta_opt, tau_opt))
        except Exception:
            continue
        events = forward_pass_week_by_week(cw, beta_opt)
        for ev in events:
            for i, (name, partner) in enumerate(ev["contestant_keys"]):
                key = (int(ev["season"]), int(ev["week"]), name, partner)
                if key not in store:
                    store[key] = []
                store[key].append(float(ev["f"][i]))
    if not store:
        return pd.DataFrame(), bootstrap_results
    rows = []
    for (season, week, celebrity_name, ballroom_partner), f_list in store.items():
        arr = np.array(f_list)
        rows.append({
            "season": season,
            "week": week,
            "celebrity_name": celebrity_name,
            "ballroom_partner": ballroom_partner,
            "f_mean": float(np.mean(arr)),
            "f_lower": float(np.percentile(arr, p_lo)),
            "f_upper": float(np.percentile(arr, p_hi)),
            "f_median": float(np.median(arr)),
            "n_bootstrap": len(arr),
        })
    return pd.DataFrame(rows), bootstrap_results


def bootstrap_fan_share_intervals_by_weeks(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    beta_init: dict,
    tau_init: float,
    n_bootstrap: int = 30,
    *,
    random_state: int | None = None,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[pd.DataFrame, list[tuple[dict, float]]]:
    """
    Alternative: bootstrap by resampling weeks within each season.
    For each season, resample weeks with replacement n_bootstrap times, refit,
    run forward pass on full cw; collect f and compute percentiles per (season, week, contestant).
    Same return shape as bootstrap_fan_share_intervals.
    """
    rng = np.random.default_rng(random_state)
    store: dict[tuple, list[float]] = {}
    bootstrap_results: list[tuple[dict, float]] = []
    n_contestants = int(cw["contestant_id"].max()) + 1 if "contestant_id" in cw.columns else None
    seasons = cw["season"].dropna().unique()
    for season in seasons:
        season_df = cw[cw["season"] == season]
        weeks = season_df["week"].unique()
        if len(weeks) < 2:
            continue
        for _ in range(n_bootstrap):
            boot_weeks = rng.choice(weeks, size=len(weeks), replace=True)
            idx = []
            for w in boot_weeks:
                idx.extend(season_df[season_df["week"] == w].index.tolist())
            cw_boot = cw.loc[sorted(set(idx))].copy()
            elim = build_elimination_events(cw_boot)
            finals = build_finals_events(cw_boot)
            if len(elim) < 2:
                continue
            try:
                beta_opt, tau_opt, result = fit(
                    beta_init, tau_init, cw_boot, raw,
                    elimination_events=elim,
                    finals_events=finals,
                    n_contestants=n_contestants,
                )
                if not result.success:
                    continue
                bootstrap_results.append((beta_opt, tau_opt))
            except Exception:
                continue
            events = forward_pass_week_by_week(cw, beta_opt)
            for ev in events:
                for i, (name, partner) in enumerate(ev["contestant_keys"]):
                    key = (int(ev["season"]), int(ev["week"]), name, partner)
                    if key not in store:
                        store[key] = []
                    store[key].append(float(ev["f"][i]))
    if not store:
        return pd.DataFrame(), bootstrap_results
    p_lo, p_hi = percentiles
    rows = []
    for (season, week, celebrity_name, ballroom_partner), f_list in store.items():
        arr = np.array(f_list)
        rows.append({
            "season": season,
            "week": week,
            "celebrity_name": celebrity_name,
            "ballroom_partner": ballroom_partner,
            "f_mean": float(np.mean(arr)),
            "f_lower": float(np.percentile(arr, p_lo)),
            "f_upper": float(np.percentile(arr, p_hi)),
            "f_median": float(np.median(arr)),
            "n_bootstrap": len(arr),
        })
    return pd.DataFrame(rows), bootstrap_results
