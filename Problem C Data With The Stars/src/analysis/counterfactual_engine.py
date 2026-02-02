"""
Counterfactual engine: compare elimination under rank, percent, and judges-save
for each (season, week). Build I1 table and support I2 metrics (fraction differing,
fan influence index, sensitivity plots).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import (
    week_elimination_counterfactuals,
    weighted_percent_combined,
    use_judges_save,
    get_rule_for_season,
)
from src.fit.forward_pass import forward_pass_week_by_week
from src.models.vote_latent import shares_to_index_totals


def build_forward_events_from_panel_fan(
    panel_df: pd.DataFrame,
    fan_shares_df: pd.DataFrame,
    *,
    score_col: str = "score_week_total",
    fan_share_col: str = "fan_share",
    active_col: str = "active",
    elimination_week_col: str = "elimination_week",
) -> list[dict]:
    """
    Build forward_events from panel (season, week, celebrity_name, score, active, elimination_week)
    and fan_shares (season, week, celebrity_name, fan_share). Merge and for each (season, week)
    produce J, f, V, contestant_keys, observed_eliminated_index, rule.
    """
    merge_keys = ["season", "week", "celebrity_name"]
    if "ballroom_partner" in panel_df.columns and "ballroom_partner" in fan_shares_df.columns:
        merge_keys = ["season", "week", "celebrity_name", "ballroom_partner"]
    needed = merge_keys + [score_col, active_col, elimination_week_col]
    if not all(c in panel_df.columns for c in needed):
        raise ValueError(f"panel_df must have {needed}")
    if not all(c in fan_shares_df.columns for c in merge_keys + [fan_share_col]):
        raise ValueError(f"fan_shares_df must have {merge_keys} and {fan_share_col}")
    # Coerce merge keys to same dtypes so merge succeeds (e.g. season/week int vs object)
    panel = panel_df.copy()
    fan = fan_shares_df[merge_keys + [fan_share_col]].copy()
    for k in merge_keys:
        if k in ["season", "week"]:
            panel[k] = pd.to_numeric(panel[k], errors="coerce").fillna(0).astype(int)
            fan[k] = pd.to_numeric(fan[k], errors="coerce").fillna(0).astype(int)
    merged = panel.merge(fan, on=merge_keys, how="inner")
    merged = merged.sort_values(["season", "week"])
    out = []
    for (season, week), grp in merged.groupby(["season", "week"], sort=True):
        active = grp[grp[active_col] == 1] if active_col in grp.columns else grp
        if len(active) < 2:
            continue
        J = active[score_col].astype(float).values
        f = active[fan_share_col].astype(float).values
        f = np.clip(f, 1e-12, 1.0)
        f = f / f.sum()
        V = shares_to_index_totals(f)
        if "ballroom_partner" in active.columns:
            contestant_keys = list(
                zip(active["celebrity_name"].values, active["ballroom_partner"].values)
            )
        else:
            contestant_keys = [(n, "") for n in active["celebrity_name"].values]
        elim_week = active[elimination_week_col]
        eliminated_this_week = (elim_week == week).values
        if eliminated_this_week.any():
            observed_eliminated_index = int(np.where(eliminated_this_week)[0][0])
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


def build_counterfactual_table(
    forward_events: list[dict],
    *,
    always_include_judges_save: bool = True,
) -> pd.DataFrame:
    """
    For each (season, week) in forward_events, compute elimination under
    rank, percent, and bottom-two + judges save. Return a table with
    season, week, elim_observed, elim_rank, elim_percent, elim_judges_save,
    bottom_two_indices (tuple or None).
    If always_include_judges_save is True, compute judges-save for every week
    (for "what if" comparison); else only for seasons that use it.
    """
    rows = []
    for ev in forward_events:
        J = ev["J"]
        V = ev["V"]
        season = ev["season"]
        use_js = always_include_judges_save or use_judges_save(season)
        cf = week_elimination_counterfactuals(J, V, use_judges_save=use_js)
        elim_observed = ev.get("observed_eliminated_index")
        if elim_observed is None:
            elim_observed = -1
        is_elim_week = 1 if elim_observed >= 0 else 0
        rows.append({
            "season": ev["season"],
            "week": ev["week"],
            "is_elimination_week": is_elim_week,
            "elim_observed": elim_observed,
            "elim_rank": cf["elim_rank"],
            "elim_percent": cf["elim_percent"],
            "elim_judges_save": cf["elim_judges_save"] if use_js else -1,
            "bottom_two_indices": cf["bottom_two_indices"],
            "rule_actual": ev["rule"],
            "n_contestants": len(ev["contestant_keys"]),
        })
    return pd.DataFrame(rows)


def build_counterfactual_table_from_panel_fan(
    panel_df: pd.DataFrame,
    fan_shares_df: pd.DataFrame,
    *,
    judges_save_season_start: int = 28,
    always_include_judges_save: bool = True,
) -> pd.DataFrame:
    """
    Build counterfactual table from panel (contestant-week) and fan shares.
    judges_save_season_start is used only for documentation; use_judges_save(season)
    in scoring uses JUDGES_SAVE_FROM_SEASON (28) internally.
    """
    events = build_forward_events_from_panel_fan(panel_df, fan_shares_df)
    return build_counterfactual_table(
        events,
        always_include_judges_save=always_include_judges_save,
    )


def counterfactual_table_from_cw_beta(
    cw: pd.DataFrame,
    beta: dict,
    *,
    always_include_judges_save: bool = True,
) -> pd.DataFrame:
    """
    Run forward pass on cw with beta, then build counterfactual table.
    """
    events = forward_pass_week_by_week(cw, beta)
    return build_counterfactual_table(
        events,
        always_include_judges_save=always_include_judges_save,
    )


def fraction_weeks_elimination_differs(
    cf_table: pd.DataFrame,
    rule_a: str | None = None,
    rule_b: str | None = None,
) -> dict[str, float] | float:
    """
    Fraction of weeks where elimination differs between rule pairs.
    Returns dict with keys e.g. rank_vs_percent, rank_vs_observed, percent_vs_observed,
    percent_vs_judges_save, rank_vs_judges_save (when judges_save was computed).
    If rule_a and rule_b are provided, returns a single float for that pair.
    """
    col_map = {
        "percent": "elim_percent",
        "rank": "elim_rank",
        "observed": "elim_observed",
        "judges_save": "elim_judges_save",
    }
    n = len(cf_table)
    if n == 0:
        return {} if rule_a is None else float("nan")
    if rule_a is not None and rule_b is not None:
        ca, cb = col_map.get(rule_a), col_map.get(rule_b)
        if ca not in cf_table.columns or cb not in cf_table.columns:
            return float("nan")
        if rule_b == "judges_save" and (cf_table["elim_judges_save"] < 0).all():
            return float("nan")
        return float((cf_table[ca] != cf_table[cb]).mean())
    out = {}
    out["rank_vs_percent"] = (cf_table["elim_rank"] != cf_table["elim_percent"]).mean()
    out["rank_vs_observed"] = (cf_table["elim_rank"] != cf_table["elim_observed"]).mean()
    out["percent_vs_observed"] = (cf_table["elim_percent"] != cf_table["elim_observed"]).mean()
    has_js = (cf_table["elim_judges_save"] >= 0).any()
    if has_js:
        out["percent_vs_judges_save"] = (
            cf_table["elim_percent"] != cf_table["elim_judges_save"]
        ).mean()
        out["rank_vs_judges_save"] = (
            cf_table["elim_rank"] != cf_table["elim_judges_save"]
        ).mean()
        out["observed_vs_judges_save"] = (
            cf_table["elim_observed"] != cf_table["elim_judges_save"]
        ).mean()
    return out


def fan_influence_index(
    forward_events: list[dict],
    cf_table: pd.DataFrame,
    *,
    poor_judge_quantile: float = 0.5,
    high_fan_quantile: float = 0.5,
) -> dict[str, float]:
    """
    Fan influence index (FII): fraction of elimination weeks in which at least one
    contestant with poor judge score (bottom half J) but high fan share (top half f)
    **survives** (is not eliminated) under each method. Definition aligned with report
    Section "Definition audit (paper--code alignment)". Returns dict: rank, percent,
    judges_save -> fraction of (season, week) where such a contestant survived.
    """
    # Align cf_table to forward_events by (season, week)
    cf_by_sw = cf_table.set_index(["season", "week"])
    methods = ["rank", "percent", "judges_save"]
    elim_col = {"rank": "elim_rank", "percent": "elim_percent", "judges_save": "elim_judges_save"}
    counts = {m: 0 for m in methods}
    total_weeks = 0
    for ev in forward_events:
        season, week = ev["season"], ev["week"]
        if (season, week) not in cf_by_sw.index:
            continue
        row = cf_by_sw.loc[(season, week)]
        J = ev["J"]
        f = ev["f"]
        n = len(J)
        if n < 2:
            continue
        total_weeks += 1
        j_med = np.median(J)
        f_med = np.median(f)
        poor_judge = J <= j_med
        high_fan = f >= f_med
        poor_judge_high_fan = poor_judge & high_fan
        for m in methods:
            col = elim_col[m]
            elim_idx = row[col]
            if elim_idx < 0:
                continue
            if poor_judge_high_fan[elim_idx]:
                continue
            if np.any(poor_judge_high_fan):
                counts[m] += 1
    out = {}
    for m in methods:
        out[m] = counts[m] / total_weeks if total_weeks else 0.0
    out["total_weeks"] = total_weeks
    return out


def sensitivity_weight_sweep(
    forward_events: list[dict] | None = None,
    *,
    panel_df: pd.DataFrame | None = None,
    fan_shares_df: pd.DataFrame | None = None,
    w_judge_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    For each (season, week), compute eliminated contestant under weighted percent
    c_i = w_J * j_i + w_F * f_i for each w_J in grid (w_F = 1 - w_J). Record when
    the eliminated contestant flips **vs the previous grid point** (not vs historical).
    Definition aligned with report Section "Definition audit". Returns DataFrame:
    season, week, w_judge, eliminated_index, flip_from_prev (bool).
    Either pass forward_events or (panel_df, fan_shares_df).
    """
    if forward_events is None:
        if panel_df is not None and fan_shares_df is not None:
            forward_events = build_forward_events_from_panel_fan(panel_df, fan_shares_df)
        else:
            raise ValueError("Pass either forward_events or (panel_df, fan_shares_df)")
    if w_judge_grid is None:
        w_judge_grid = np.linspace(0.0, 1.0, 51)
    rows = []
    for ev in forward_events:
        J, V = ev["J"], ev["V"]
        n = len(J)
        if n < 2:
            continue
        prev_elim = -1
        for w_J in w_judge_grid:
            w_F = 1.0 - w_J
            _, elim = weighted_percent_combined(J, V, w_J, w_F)
            flip = prev_elim >= 0 and elim != prev_elim
            rows.append({
                "season": ev["season"],
                "week": ev["week"],
                "w_judge": float(w_J),
                "w_fan": float(w_F),
                "eliminated_index": elim,
                "flip_from_prev": flip,
            })
            prev_elim = elim
    return pd.DataFrame(rows)


def sensitivity_flip_summary(
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate sensitivity sweep: for each w_judge, fraction of (season, week) where
    the eliminated contestant flips at that weight vs the previous grid point.
    Flip is not defined at the first grid point (no previous). See report Definition audit.
    """
    if sensitivity_df.empty or "flip_from_prev" not in sensitivity_df.columns:
        return pd.DataFrame()
    grp = sensitivity_df.groupby("w_judge", as_index=False)
    return grp.agg(
        frac_weeks_flip=("flip_from_prev", "mean"),
        n_flips=("flip_from_prev", "sum"),
        n_weeks=("flip_from_prev", "count"),
    )
