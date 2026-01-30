# src/fit/judges_save_alpha.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from scipy.optimize import minimize

from src.scoring import use_judges_save


@dataclass
class AlphaFitResult:
    alpha: float
    n_obs: int
    neg_loglik: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def fit_alpha_from_bottom_two(
    weeks_df: pd.DataFrame,
    *,
    eliminated_col: str = "eliminated_name",
    bottom1_col: str = "bottom1_name",
    bottom2_col: str = "bottom2_name",
    judge_total_col: str = "score_week_total",
    alpha_init: float = 1.0,
) -> AlphaFitResult:
    """
    Fit alpha in:
        P(eliminate i | {i,k} bottom2) = sigmoid(alpha * (J_k - J_i))
    using inferred bottom-two pairs + observed eliminated.
    weeks_df must have (season, week, bottom1_name, bottom2_name, eliminated_name) plus judge totals per contestant-week.

    NOTE: This is post-hoc conditional on your inferred bottom-two.
    """

    # Build observations: for each bottom-two pair, label which of the two was eliminated.
    obs = []
    # We'll need J for both members; easiest is to pre-index judge totals by (season, week, name)
    idx = (weeks_df["season"], weeks_df["week"], weeks_df["celebrity_name"])
    J_map = pd.Series(weeks_df[judge_total_col].values, index=pd.MultiIndex.from_arrays(idx))

    # Reduce to unique (season, week) rows where bottom-two are defined
    bw = weeks_df.dropna(subset=[bottom1_col, bottom2_col, eliminated_col])[
        ["season", "week", bottom1_col, bottom2_col, eliminated_col]
    ].drop_duplicates()

    for _, r in bw.iterrows():
        s, w = int(r["season"]), int(r["week"])
        b1, b2 = r[bottom1_col], r[bottom2_col]
        elim = r[eliminated_col]
        if elim not in (b1, b2):
            # If your pipeline sometimes records "eliminated" not among inferred bottom-two,
            # skip rather than contaminating alpha.
            continue

        try:
            J1 = float(J_map.loc[(s, w, b1)])
            J2 = float(J_map.loc[(s, w, b2)])
        except KeyError:
            continue

        # Define i = elim, k = other
        if elim == b1:
            Ji, Jk = J1, J2
        else:
            Ji, Jk = J2, J1

        x = (Jk - Ji)  # positive => elim has lower J, should increase P(elim)
        obs.append(x)

    x = np.asarray(obs, dtype=float)
    n = x.size
    if n == 0:
        return AlphaFitResult(alpha=float("nan"), n_obs=0, neg_loglik=float("nan"))

    # Negative log-likelihood: y is always "elim occurs", so p = sigmoid(alpha * x)
    def nll(a: np.ndarray) -> float:
        alpha = float(a[0])
        p = _sigmoid(alpha * x)
        eps = 1e-12
        return -np.sum(np.log(p + eps))

    res = minimize(nll, x0=np.array([alpha_init], dtype=float), method="L-BFGS-B", bounds=[(0.0, 50.0)])
    alpha_hat = float(res.x[0])
    return AlphaFitResult(alpha=alpha_hat, n_obs=int(n), neg_loglik=float(res.fun))


def build_weeks_df_for_alpha(
    forward_events: list[dict],
    cf_table: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    score_col: str = "score_week_total",
) -> pd.DataFrame:
    """
    Build a long DataFrame with (season, week, celebrity_name, score_week_total,
    bottom1_name, bottom2_name, eliminated_name) for judges-save weeks, so that
    fit_alpha_from_bottom_two can be called. Uses inferred bottom-two from cf_table
    and contestant_keys from forward_events.
    """
    events_by_sw = {(int(ev["season"]), int(ev["week"])): ev for ev in forward_events}
    week_rows = []
    for _, row in cf_table.iterrows():
        season, week = int(row["season"]), int(row["week"])
        if not use_judges_save(season):
            continue
        bt = row.get("bottom_two_indices")
        if bt is None or row.get("elim_judges_save", -1) < 0:
            continue
        ev = events_by_sw.get((season, week))
        if ev is None:
            continue
        keys = ev["contestant_keys"]
        i, j = bt[0], bt[1]
        if i >= len(keys) or j >= len(keys):
            continue
        bottom1_name = keys[i][0]
        bottom2_name = keys[j][0]
        obs_idx = ev.get("observed_eliminated_index")
        eliminated_name = keys[obs_idx][0] if obs_idx is not None and obs_idx < len(keys) else None
        week_rows.append({
            "season": season,
            "week": week,
            "bottom1_name": bottom1_name,
            "bottom2_name": bottom2_name,
            "eliminated_name": eliminated_name,
        })
    if not week_rows:
        return pd.DataFrame()
    week_df = pd.DataFrame(week_rows).drop_duplicates(subset=["season", "week"])
    panel = panel_df[["season", "week", "celebrity_name", score_col]].copy()
    panel["season"] = pd.to_numeric(panel["season"], errors="coerce").fillna(0).astype(int)
    panel["week"] = pd.to_numeric(panel["week"], errors="coerce").fillna(0).astype(int)
    week_df["season"] = week_df["season"].astype(int)
    week_df["week"] = week_df["week"].astype(int)
    merged = panel.merge(week_df, on=["season", "week"], how="inner")
    return merged


def fit_alpha_and_save(
    forward_events: list[dict],
    cf_table: pd.DataFrame,
    panel_df: pd.DataFrame,
    out_path: str | Path,
    *,
    alpha_init: float = 1.0,
) -> AlphaFitResult:
    """
    Build weeks table, fit alpha, write judges_save_alpha.json. Returns AlphaFitResult.
    """
    weeks_df = build_weeks_df_for_alpha(forward_events, cf_table, panel_df)
    result = fit_alpha_from_bottom_two(weeks_df, alpha_init=alpha_init)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"alpha": result.alpha, "n_obs": result.n_obs, "neg_loglik": result.neg_loglik},
            f,
            indent=2,
        )
    return result
