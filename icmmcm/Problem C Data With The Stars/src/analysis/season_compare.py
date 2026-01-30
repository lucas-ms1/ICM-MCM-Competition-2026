# src/analysis/season_compare.py
from __future__ import annotations

import os
import pandas as pd

from typing import Dict

from src.analysis.counterfactual_engine import (
    build_forward_events_from_panel_fan,
    build_counterfactual_table,
    fraction_weeks_elimination_differs,
    fan_influence_index,
    sensitivity_weight_sweep,
    sensitivity_flip_summary,
)


def run_season_compare(
    *,
    panel_df: pd.DataFrame,
    fan_shares_df: pd.DataFrame,
    out_dir: str = "reports/tables",
    judges_save_season_start: int = 28,
) -> Dict[str, str]:
    """
    Produces:
      - season_rule_comparison.csv
      - sensitivity_flip_summary.csv
    Assumes:
      panel_df contains season/week/celebrity_name/score_week_total/active/elimination info
      fan_shares_df contains season/week/celebrity_name/fan_share (your inferred f_{i,t})
    """
    os.makedirs(out_dir, exist_ok=True)

    events = build_forward_events_from_panel_fan(panel_df, fan_shares_df)
    cf = build_counterfactual_table(events)

    # Summary by season: how often outcomes differ across rules
    seasons = sorted(cf["season"].unique())
    rows = []
    for s in seasons:
        cfs = cf[cf["season"] == s].copy()
        events_s = [e for e in events if e["season"] == s]
        fi = fan_influence_index(events_s, cfs) if events_s else {}
        rows.append({
            "season": s,
            "n_elim_weeks": int(cfs["is_elimination_week"].sum()) if "is_elimination_week" in cfs.columns else int(len(cfs)),
            "pct_vs_rank_diff_frac": fraction_weeks_elimination_differs(cfs, rule_a="percent", rule_b="rank"),
            "pct_vs_judgessave_diff_frac": fraction_weeks_elimination_differs(cfs, rule_a="percent", rule_b="judges_save"),
            "rank_vs_judgessave_diff_frac": fraction_weeks_elimination_differs(cfs, rule_a="rank", rule_b="judges_save"),
            "fan_influence_index_percent": fi.get("percent", float("nan")),
            "fan_influence_index_rank": fi.get("rank", float("nan")),
        })

    season_summary = pd.DataFrame(rows).sort_values("season")
    out1 = os.path.join(out_dir, "season_rule_comparison.csv")
    season_summary.to_csv(out1, index=False)

    # Sensitivity sweep (weighted percent): where do flips occur as you change w (judge weight)?
    sweep = sensitivity_weight_sweep(panel_df=panel_df, fan_shares_df=fan_shares_df)
    flip = sensitivity_flip_summary(sweep)
    out2 = os.path.join(out_dir, "sensitivity_flip_summary.csv")
    flip.to_csv(out2, index=False)

    return {"season_rule_comparison": out1, "sensitivity_flip_summary": out2}
