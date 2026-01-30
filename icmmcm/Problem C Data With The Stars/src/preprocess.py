"""Data preprocessing and cleaning."""

import re
import numpy as np
import pandas as pd
from src.rules import parse_elimination_week as parse_elim_week_from_results

# Score column pattern: week{N}_judge{J}_score
SCORE_COL_PATTERN = re.compile(r"week(\d+)_judge(\d+)_score")


def melt_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide score columns to long: season, celebrity_name, ballroom_partner, week, judge_id, score.
    Treat N/A and empty as missing (NaN); 0 stays 0.
    """
    id_cols = ["season", "celebrity_name", "ballroom_partner", "results", "placement"]
    available_ids = [c for c in id_cols if c in raw.columns]
    score_cols = [c for c in raw.columns if SCORE_COL_PATTERN.fullmatch(c)]
    if not score_cols:
        raise ValueError("No score columns found matching week{N}_judge{J}_score")

    # Build (week, judge_id) from column names
    week_judge = [(int(SCORE_COL_PATTERN.fullmatch(c).group(1)), int(SCORE_COL_PATTERN.fullmatch(c).group(2))) for c in score_cols]
    rows = []
    for _, row in raw.iterrows():
        base = {k: row[k] for k in available_ids}
        for (week_num, judge_num), col in zip(week_judge, score_cols):
            val = row[col]
            if pd.isna(val) or val == "" or (isinstance(val, str) and str(val).strip().upper() == "N/A"):
                score = float("nan")
            else:
                score = pd.to_numeric(val, errors="coerce")
            rows.append({**base, "week": week_num, "judge_id": judge_num, "score": score})
    long = pd.DataFrame(rows)
    return long


def aggregate_contestant_week(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate long to contestant-week: score_week_total (sum), num_judges, has_bonus.
    J_{i,t} = sum of non-NaN scores; has_bonus = 1 if any score is non-integer or > 10.
    """
    cw = (
        long_df.groupby(["season", "celebrity_name", "ballroom_partner", "week"], dropna=False)
        .agg(
            score_week_total=("score", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            num_judges=("score", lambda s: pd.to_numeric(s, errors="coerce").notna().sum()),
        )
        .reset_index()
    )

    def _has_bonus(grp):
        s = pd.to_numeric(grp["score"], errors="coerce").dropna()
        if s.empty:
            return False
        return ((s != s.astype(int)) | (s > 10)).any()

    bonus_ser = long_df.groupby(["season", "celebrity_name", "ballroom_partner", "week"]).apply(_has_bonus)
    bonus_ser.name = "has_bonus"
    cw = cw.merge(bonus_ser.reset_index(), on=["season", "celebrity_name", "ballroom_partner", "week"], how="left")
    cw["has_bonus"] = cw["has_bonus"].fillna(False).astype(int)
    return cw


def add_active(cw: pd.DataFrame) -> pd.DataFrame:
    """active(i,t)=1 if contestant has nonzero score that week. Base on score_week_total > 0."""
    cw = cw.copy()
    cw["active"] = (cw["score_week_total"] > 0).astype(int)
    return cw


def add_elimination(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add elimination_week (contestant-season): from results "Eliminated Week k" or first week of zeros.
    Withdrew: set elimination_week to week after last nonzero (or leave missing; we use last nonzero week).
    Optionally add eliminated = 1 in week t when t >= elimination_week (per contestant).
    """
    # Contestant-level elimination_week from raw
    elim_from_results = raw.apply(
        lambda r: parse_elim_week_from_results(r["results"]) if "results" in r and pd.notna(r.get("results")) else None,
        axis=1,
    )
    raw_aug = raw[["season", "celebrity_name", "ballroom_partner"]].copy()
    raw_aug["elimination_week_from_results"] = elim_from_results.values

    # From scores: first week where score_week_total == 0 and previous week had nonzero
    cw_sorted = cw.sort_values(["season", "celebrity_name", "ballroom_partner", "week"])
    cw_sorted["prev_total"] = cw_sorted.groupby(["season", "celebrity_name", "ballroom_partner"])[
        "score_week_total"
    ].shift(1)
    first_zero = cw_sorted[
        (cw_sorted["score_week_total"] == 0) & (cw_sorted["prev_total"].fillna(0) > 0)
    ].groupby(["season", "celebrity_name", "ballroom_partner"])["week"].min()
    first_zero.name = "elimination_week_from_scores"
    raw_aug = raw_aug.merge(
        first_zero.reset_index(),
        on=["season", "celebrity_name", "ballroom_partner"],
        how="left",
    )
    # Prefer results; else use from scores
    raw_aug["elimination_week"] = raw_aug["elimination_week_from_results"].fillna(
        raw_aug["elimination_week_from_scores"]
    )

    # Withdrew: set elimination_week to last week with nonzero score
    from src.rules import is_withdrew

    withdrew_mask = raw.apply(lambda r: is_withdrew(str(r.get("results", "") or "")), axis=1)
    if withdrew_mask.any():
        last_nonzero_week = (
            cw[cw["score_week_total"] > 0]
            .groupby(["season", "celebrity_name", "ballroom_partner"])["week"]
            .max()
            .rename("last_week_with_score")
        )
        raw_aug = raw_aug.merge(
            last_nonzero_week.reset_index(),
            on=["season", "celebrity_name", "ballroom_partner"],
            how="left",
        )
        raw_aug.loc[withdrew_mask, "elimination_week"] = raw_aug.loc[withdrew_mask, "last_week_with_score"]
        raw_aug = raw_aug.drop(columns=["last_week_with_score"], errors="ignore")

    contestant_elim = raw_aug[["season", "celebrity_name", "ballroom_partner", "elimination_week"]]
    cw = cw.merge(
        contestant_elim,
        on=["season", "celebrity_name", "ballroom_partner"],
        how="left",
    )
    cw["eliminated"] = (cw["week"] >= cw["elimination_week"]).fillna(False).astype(int)
    return cw


def add_finals_and_placement(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Finals week per season: last week where n_active in [3, 5].
    Carry placement and results from raw; add finals_week per season.
    """
    n_active = cw[cw["active"] == 1].groupby(["season", "week"]).size().rename("n_active")
    n_active = n_active.reset_index()
    finals_candidates = n_active[n_active["n_active"].between(3, 5)]
    finals_week_per_season = finals_candidates.groupby("season")["week"].max().rename("finals_week")
    cw = cw.merge(
        finals_week_per_season.reset_index(),
        on="season",
        how="left",
    )
    placement_cols = [c for c in ["placement", "results"] if c in raw.columns]
    if placement_cols:
        carry = raw[["season", "celebrity_name", "ballroom_partner"] + placement_cols].drop_duplicates()
        cw = cw.merge(
            carry,
            on=["season", "celebrity_name", "ballroom_partner"],
            how="left",
        )
    return cw


def add_zscore_judge(cw: pd.DataFrame) -> pd.DataFrame:
    """
    Add z-score of judge score within (season, week) among active contestants.
    S = (score_week_total - mean) / std over active rows; if std==0 or n_active==1, set S=0.
    """
    cw = cw.copy()
    cw["S"] = 0.0
    for (season, week), grp in cw.groupby(["season", "week"]):
        j = grp["score_week_total"].astype(float).values
        m = grp["active"].astype(bool).values
        if not np.any(m):
            continue
        j_active = j[m]
        mu = np.mean(j_active)
        std = np.std(j_active)
        if std == 0 or np.isnan(std) or len(j_active) == 1:
            s_vals = np.zeros_like(j)
        else:
            s_vals = (j - mu) / std
            s_vals[~m] = 0.0
        cw.loc[grp.index, "S"] = s_vals
    return cw


def run_pipeline(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full pipeline: melt -> contestant-week (agg, active, elimination, finals/placement).
    Returns (long_df, contestant_week_df).
    """
    long_df = melt_to_long(raw)
    cw = aggregate_contestant_week(long_df)
    cw = add_active(cw)
    cw = add_elimination(cw, raw)
    cw = add_finals_and_placement(cw, raw)
    return long_df, cw
