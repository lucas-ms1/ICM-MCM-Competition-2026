"""
K3: Evaluate proposed system using inferred votes.
Reports: changes vs historical, controversy mismatches, predictability (robustness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import (
    weighted_percent_with_saturation,
    percent_with_judges_save_trigger,
    percent_combined,
    rank_combined,
)
from src.fit.forward_pass import forward_pass_week_by_week, forward_pass_to_dataframe
from src.fit.margin_to_flip import (
    margin_to_flip_radius,
    margin_to_flip_radius_l2_f,
    margin_to_flip_radius_combined,
)
from src.models.vote_latent import build_contestant_week_covariates
from src.fit.fit_elimination import fit


# Default proposed rule params (Option A: weighted percent with saturation)
DEFAULT_W_JUDGE = 0.5
DEFAULT_SOFTCAP_ALPHA = 0.8
# Option B: percent + judges-save trigger (threshold in raw points)
DEFAULT_JUDGES_SAVE_THRESHOLD = 5.0


def _elim_proposed_saturation(J: np.ndarray, V: np.ndarray, w: float, alpha: float) -> int:
    """Eliminated index under weighted percent with saturation (Option A)."""
    _, elim = weighted_percent_with_saturation(
        J, V, w, softcap_method="power", alpha=alpha
    )
    return elim


def _elim_proposed_trigger(J: np.ndarray, V: np.ndarray, threshold: float) -> int:
    """Eliminated index under percent + judges-save trigger (Option B)."""
    _, elim = percent_with_judges_save_trigger(J, V, threshold)
    return elim


def evaluate_proposed_system(
    forward_events: list[dict],
    proposed_rule_name: str,
    *,
    w: float = DEFAULT_W_JUDGE,
    alpha: float = DEFAULT_SOFTCAP_ALPHA,
    threshold: float = DEFAULT_JUDGES_SAVE_THRESHOLD,
) -> pd.DataFrame:
    """
    For each (season, week) compute proposed elimination; compare to observed.
    proposed_rule_name: "weighted_saturation" | "percent_trigger".
    Returns DataFrame: season, week, elim_observed, elim_proposed, same_elim, rule_actual.
    """
    rows = []
    for ev in forward_events:
        J = np.asarray(ev["J"], dtype=float)
        V = np.asarray(ev["V"], dtype=float)
        n = len(J)
        if n < 2:
            continue
        obs = ev.get("observed_eliminated_index")
        if obs is None:
            obs = -1
        if proposed_rule_name == "weighted_saturation":
            prop = _elim_proposed_saturation(J, V, w, alpha)
        elif proposed_rule_name == "percent_trigger":
            prop = _elim_proposed_trigger(J, V, threshold)
        else:
            raise ValueError(f"Unknown proposed_rule_name: {proposed_rule_name}")
        same = int(obs == prop) if obs >= 0 else 0
        rows.append({
            "season": ev["season"],
            "week": ev["week"],
            "elim_observed": obs,
            "elim_proposed": prop,
            "same_elim": same,
            "rule_actual": ev["rule"],
            "n_contestants": n,
        })
    return pd.DataFrame(rows)


def controversy_mismatch_weeks(
    forward_events: list[dict],
    eval_df: pd.DataFrame,
    *,
    mismatch_def: str = "eliminated_topJ_bottomf",
) -> dict:
    """
    Controversy mismatch: eliminated contestant had J above median and f below median
    ("judges liked them, fans didn't"). Count weeks with mismatch under observed vs
    under proposed elimination. Returns dict with n_weeks, n_mismatch_observed,
    n_mismatch_proposed, reduction (count), reduction_pct.
    """
    eval_by_sw = eval_df.set_index(["season", "week"])
    n_mismatch_obs = 0
    n_mismatch_prop = 0
    n_weeks = 0
    for ev in forward_events:
        season, week = ev["season"], ev["week"]
        if (season, week) not in eval_by_sw.index:
            continue
        row = eval_by_sw.loc[(season, week)]
        J = np.asarray(ev["J"], dtype=float)
        f = np.asarray(ev["f"], dtype=float)
        n = len(J)
        if n < 2:
            continue
        n_weeks += 1
        j_med = np.median(J)
        f_med = np.median(f)
        # Mismatch: eliminated has J above median and f below median
        obs = int(row["elim_observed"])
        prop = int(row["elim_proposed"])
        if obs >= 0 and obs < n and J[obs] >= j_med - 1e-12 and f[obs] <= f_med + 1e-12:
            n_mismatch_obs += 1
        if prop >= 0 and prop < n and J[prop] >= j_med - 1e-12 and f[prop] <= f_med + 1e-12:
            n_mismatch_prop += 1
    reduction = n_mismatch_obs - n_mismatch_prop
    reduction_pct = (reduction / n_mismatch_obs * 100) if n_mismatch_obs > 0 else 0.0
    return {
        "n_weeks": n_weeks,
        "n_mismatch_observed": n_mismatch_obs,
        "n_mismatch_proposed": n_mismatch_prop,
        "reduction": reduction,
        "reduction_pct": reduction_pct,
    }


def predictability_summary(
    forward_events: list[dict],
    rule_name: str,
    *,
    use_l2_f: bool = True,
    w: float = DEFAULT_W_JUDGE,
    alpha: float = DEFAULT_SOFTCAP_ALPHA,
    threshold: float = DEFAULT_JUDGES_SAVE_THRESHOLD,
    max_weeks: int | None = None,
) -> pd.DataFrame:
    """
    Robustness radius (margin-to-flip) per week for the given rule.
    rule_name: "percent" | "rank" | "weighted_saturation" | "percent_trigger".
    Returns DataFrame: season, week, rule, robustness_radius, eliminated_index.
    """
    rows = []
    for ev in forward_events:
        J = np.asarray(ev["J"], dtype=float)
        f = np.asarray(ev["f"], dtype=float)
        n = len(J)
        if n < 2:
            continue
        V = ev["V"]
        if rule_name == "percent":
            _, elim = percent_combined(J, V)
            if use_l2_f:
                radius, _ = margin_to_flip_radius_l2_f(J, f, "percent", elim)
            else:
                radius, _ = margin_to_flip_radius(J, f, "percent", elim)
        elif rule_name == "rank":
            _, elim = rank_combined(J, V)
            if use_l2_f:
                radius, _ = margin_to_flip_radius_l2_f(J, f, "rank", elim)
            else:
                radius, _ = margin_to_flip_radius(J, f, "rank", elim)
        elif rule_name == "weighted_saturation":
            elim = _elim_proposed_saturation(J, V, w, alpha)
            # Combined score: w * j + (1-w) * softcap(f')
            j_norm = J / J.sum() if J.sum() > 0 else np.ones(n) / n

            def combined_fn(f_prime: np.ndarray) -> np.ndarray:
                from src.scoring import softcap_fan_share
                f_cap = softcap_fan_share(f_prime, "power", alpha=alpha)
                return w * j_norm + (1.0 - w) * f_cap

            radius, _ = margin_to_flip_radius_combined(J, f, elim, combined_fn)
            if use_l2_f and radius != np.inf:
                # L2 in share space not returned by margin_to_flip_radius_combined; use log-space radius
                pass
        elif rule_name == "percent_trigger":
            elim = _elim_proposed_trigger(J, V, threshold)
            j_norm = J / J.sum() if J.sum() > 0 else np.ones(n) / n

            def combined_fn(f_prime: np.ndarray) -> np.ndarray:
                return j_norm + f_prime

            radius, _ = margin_to_flip_radius_combined(J, f, elim, combined_fn)
        else:
            raise ValueError(f"Unknown rule_name: {rule_name}")
        rows.append({
            "season": ev["season"],
            "week": ev["week"],
            "rule": rule_name,
            "robustness_radius": float(radius),
            "eliminated_index": elim,
        })
    return pd.DataFrame(rows)


def run_k_evaluation(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    beta: dict | None = None,
    *,
    proposed_rule: str = "weighted_saturation",
    w: float = DEFAULT_W_JUDGE,
    alpha: float = DEFAULT_SOFTCAP_ALPHA,
    threshold: float = DEFAULT_JUDGES_SAVE_THRESHOLD,
    beta_init: dict | None = None,
    tau_init: float = 1.0,
    compute_predictability: bool = True,
    predictability_max_weeks: int | None = None,
) -> dict:
    """
    Full K3 evaluation: get forward_events (fit if beta not provided), evaluate proposed
    system, controversy mismatches, predictability. Returns dict with eval_df,
    changes_summary (fraction weeks changed, per-season counts), controversy_summary,
    predictability_current (DataFrame for actual rule), predictability_proposed (DataFrame).
    """
    cw = build_contestant_week_covariates(cw, raw)
    if beta is None:
        if beta_init is None:
            beta_init = {
                "beta0": 0.0,
                "beta_J": 1.0,
                "beta_P": 0.0,
                "beta_U": 0.0,
                "beta_X": np.array([0.0, 0.0]),
            }
        beta, _, _ = fit(beta_init, tau_init, cw, raw)
    events = forward_pass_week_by_week(cw, beta)

    eval_df = evaluate_proposed_system(
        events,
        proposed_rule,
        w=w,
        alpha=alpha,
        threshold=threshold,
    )
    if eval_df.empty:
        return {
            "eval_df": eval_df,
            "changes_summary": {},
            "controversy_summary": {},
            "predictability_current": pd.DataFrame(),
            "predictability_proposed": pd.DataFrame(),
            "forward_events": events,
        }

    # Changes vs historical
    n_weeks = len(eval_df)
    same = eval_df["same_elim"].sum()
    frac_same = same / n_weeks if n_weeks else 0.0
    frac_changed = 1.0 - frac_same
    per_season = (
        eval_df.groupby("season")
        .agg(n_weeks=("same_elim", "count"), n_same=("same_elim", "sum"))
        .assign(n_changed=lambda x: x["n_weeks"] - x["n_same"])
    )
    changes_summary = {
        "n_weeks": n_weeks,
        "n_same": int(same),
        "n_changed": int(n_weeks - same),
        "frac_changed": frac_changed,
        "per_season": per_season,
    }

    controversy_summary = controversy_mismatch_weeks(events, eval_df)

    # Predictability: robustness radius for actual rule (varies by week) and proposed rule
    predictability_current = pd.DataFrame()
    predictability_proposed = pd.DataFrame()
    if compute_predictability:
        radii_current = []
        events_for_predict = events if predictability_max_weeks is None else events[: predictability_max_weeks]
        for ev in events_for_predict:
            r = ev["rule"]
            obs = ev.get("observed_eliminated_index")
            if obs is None or len(ev["J"]) < 2:
                continue
            _, rad = margin_to_flip_radius_l2_f(ev["J"], ev["f"], r, obs)
            radii_current.append({
                "season": ev["season"],
                "week": ev["week"],
                "rule": r,
                "robustness_radius": rad,
            })
        predictability_current = pd.DataFrame(radii_current)
        predictability_proposed = predictability_summary(
            events,
            proposed_rule,
            use_l2_f=False,
            w=w,
            alpha=alpha,
            threshold=threshold,
            max_weeks=predictability_max_weeks,
        )

    return {
        "eval_df": eval_df,
        "changes_summary": changes_summary,
        "controversy_summary": controversy_summary,
        "predictability_current": predictability_current,
        "predictability_proposed": predictability_proposed,
        "forward_events": events,
    }
