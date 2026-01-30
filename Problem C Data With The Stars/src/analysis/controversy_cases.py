"""
Controversy case studies (I3): seasons 2 (Jerry Rice), 4 (Billy Ray Cyrus),
11 (Bristol Palin), 27 (Bobby Bones). For each: fan shares vs judge scores,
counterfactual elimination under rank/percent/judges-save, and whether
judges-save would have changed results.
"""

from __future__ import annotations

import pandas as pd

from src.scoring import get_rule_for_season, use_judges_save
from src.fit.forward_pass import forward_pass_week_by_week
from src.analysis.counterfactual_engine import build_counterfactual_table

CONTROVERSY_SEASONS = [2, 4, 11, 27]
CONTROVERSY_LABELS = {
    2: "Season 2 (Jerry Rice)",
    4: "Season 4 (Billy Ray Cyrus)",
    11: "Season 11 (Bristol Palin)",
    27: "Season 27 (Bobby Bones)",
}


def fan_shares_vs_judges_table(
    forward_events: list[dict],
    season: int,
) -> pd.DataFrame:
    """
    Table of estimated fan shares vs judge scores for one season.
    Rows: (week, celebrity_name, ballroom_partner), columns: J, f, j_share, combined_approx.
    """
    rows = []
    for ev in forward_events:
        if ev["season"] != season:
            continue
        week = ev["week"]
        J = ev["J"]
        f = ev["f"]
        j_sum = J.sum() or 1.0
        j_share = J / j_sum
        c = j_share + f
        for i, (name, partner) in enumerate(ev["contestant_keys"]):
            rows.append({
                "season": season,
                "week": week,
                "celebrity_name": name,
                "ballroom_partner": partner,
                "J": float(J[i]),
                "f": float(f[i]),
                "j_share": float(j_share[i]),
                "combined_approx": float(c[i]),
                "is_eliminated_this_week": ev.get("observed_eliminated_index") == i,
            })
    return pd.DataFrame(rows)


def counterfactual_elimination_table(
    forward_events: list[dict],
    cf_table: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """
    For each week in season: who would be eliminated under rank, percent,
    judges-save; who was actually eliminated; contestant names for each.
    """
    season_events = [e for e in forward_events if e["season"] == season]
    cf_season = cf_table[cf_table["season"] == season]
    if cf_season.empty or not season_events:
        return pd.DataFrame()
    rows = []
    for ev in season_events:
        week = ev["week"]
        keys = ev["contestant_keys"]
        cf_row = cf_season[cf_season["week"] == week]
        if cf_row.empty:
            continue
        cf_row = cf_row.iloc[0]
        def name_at(idx: int) -> str:
            if idx < 0 or idx >= len(keys):
                return ""
            return f"{keys[idx][0]} ({keys[idx][1]})"
        obs = ev.get("observed_eliminated_index")
        rows.append({
            "season": season,
            "week": week,
            "elim_observed": name_at(obs) if obs is not None else "",
            "elim_rank": name_at(int(cf_row["elim_rank"])),
            "elim_percent": name_at(int(cf_row["elim_percent"])),
            "elim_judges_save": name_at(int(cf_row["elim_judges_save"]))
            if cf_row["elim_judges_save"] >= 0
            else "",
            "rule_actual": ev["rule"],
        })
    return pd.DataFrame(rows)


def judges_save_would_have_changed(
    forward_events: list[dict],
    cf_table: pd.DataFrame,
    season: int,
) -> dict:
    """
    For this season: would bottom-two + judges save have changed the eliminated
    contestant vs the actual rule? Returns dict with would_differ (bool),
    weeks_differ (list of week numbers), narrative (str).
    """
    cf_season = cf_table[cf_table["season"] == season]
    if cf_season.empty:
        return {"would_differ": False, "weeks_differ": [], "narrative": ""}
    actual_rule = get_rule_for_season(season)
    uses_js = use_judges_save(season)
    # Compare elim_observed vs elim_judges_save when both are available
    differs = cf_season["elim_observed"] != cf_season["elim_judges_save"]
    weeks_differ = cf_season.loc[differs & (cf_season["elim_judges_save"] >= 0), "week"].tolist()
    would_differ = len(weeks_differ) > 0
    if uses_js:
        narrative = (
            f"Season {season} used judges save. "
            f"Observed elimination matched judges-save in all weeks."
            if not would_differ
            else f"Season {season} used judges save. "
            f"Observed vs counterfactual judges-save differed in weeks: {weeks_differ}."
        )
    else:
        narrative = (
            f"Season {season} used {actual_rule} only (no judges save). "
            f"If judges save had been used, elimination would have differed in weeks: {weeks_differ}."
            if would_differ
            else f"Season {season} used {actual_rule} only. "
            f"If judges save had been used, the same contestant would have been eliminated each week."
        )
    return {
        "would_differ": would_differ,
        "weeks_differ": weeks_differ,
        "narrative": narrative,
    }


def run_controversy_case(
    season: int,
    forward_events: list[dict],
    cf_table: pd.DataFrame,
) -> dict:
    """
    Run one controversy case. Returns dict with:
    - fan_shares_vs_judges: DataFrame
    - counterfactual_elimination: DataFrame
    - judges_save_impact: dict from judges_save_would_have_changed
    - label: str (e.g. "Season 2 (Jerry Rice)")
    """
    return {
        "label": CONTROVERSY_LABELS.get(season, f"Season {season}"),
        "fan_shares_vs_judges": fan_shares_vs_judges_table(forward_events, season),
        "counterfactual_elimination": counterfactual_elimination_table(
            forward_events, cf_table, season
        ),
        "judges_save_impact": judges_save_would_have_changed(
            forward_events, cf_table, season
        ),
    }


def run_all_controversy_cases(
    cw: pd.DataFrame,
    beta: dict,
    raw: pd.DataFrame | None = None,
    *,
    seasons: list[int] | None = None,
) -> dict[int, dict]:
    """
    Run controversy case studies for CONTROVERSY_SEASONS (or given seasons).
    cw must have covariates. Returns dict mapping season -> run_controversy_case output.
    """
    if seasons is None:
        seasons = CONTROVERSY_SEASONS
    events = forward_pass_week_by_week(cw, beta)
    cf_table = build_counterfactual_table(events, always_include_judges_save=True)
    out = {}
    for s in seasons:
        if s not in {e["season"] for e in events}:
            continue
        out[s] = run_controversy_case(s, events, cf_table)
    return out
