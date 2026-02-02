# src/analysis/pro_dancer_effects.py
"""Pro dancer fixed-effects table from J1/J2 regressions."""
from __future__ import annotations

import os
import re
import pandas as pd
from typing import Dict, Optional

def _params_to_series(model_result) -> pd.Series:
    """
    Works with statsmodels-like results (result.params) or dicts.
    """
    if hasattr(model_result, "params"):
        p = model_result.params
        return p if isinstance(p, pd.Series) else pd.Series(p)
    if isinstance(model_result, dict):
        return pd.Series(model_result)
    raise TypeError("Unsupported model_result type; expected object with .params or dict.")


def extract_categorical_effects(
    params: pd.Series,
    *,
    cat_name: str,
    param_prefix: Optional[str] = None,
) -> pd.Series:
    """
    Extract coefficients for categorical fixed effects.
    Supports:
    - Patsy/statsmodels naming: C(ballroom_partner)[T.Some Pro]
    - Prefix naming (e.g. param_prefix="pro_"): pro_Some Pro -> "Some Pro"
    """
    # Patsy-style: C(cat_name)[T.Level]
    pat = re.compile(rf"C\({re.escape(cat_name)}\)\[T\.(.*)\]")
    out = {}
    for k, v in params.items():
        key = str(k)
        m = pat.match(key)
        if m:
            out[m.group(1)] = float(v)
    # If no patsy matches and prefix given, use prefix-based extraction
    if not out and param_prefix is not None:
        for k, v in params.items():
            key = str(k)
            if key.startswith(param_prefix):
                level = key[len(param_prefix):]
                out[level] = float(v)
    return pd.Series(out).sort_index()


def build_pro_dancer_effect_table(
    judges_model_result,
    fans_model_result,
    *,
    pro_cat: str = "ballroom_partner",
    param_prefix: Optional[str] = "pro_",
) -> pd.DataFrame:
    pj = _params_to_series(judges_model_result)
    pf = _params_to_series(fans_model_result)

    j_eff = extract_categorical_effects(pj, cat_name=pro_cat, param_prefix=param_prefix)
    f_eff = extract_categorical_effects(pf, cat_name=pro_cat, param_prefix=param_prefix)

    # Align and fill missing with 0 (baseline pro is absorbed in intercept)
    pros = sorted(set(j_eff.index).union(set(f_eff.index)))
    df = pd.DataFrame({
        "pro_dancer": pros,
        "judge_FE": [j_eff.get(p, 0.0) for p in pros],
        "fan_FE": [f_eff.get(p, 0.0) for p in pros],
    })
    df["fan_minus_judge"] = df["fan_FE"] - df["judge_FE"]

    # Ranks for reporting
    # Use nullable integer dtype so NaNs (e.g., if a regression failed) don't crash the pipeline.
    df["rank_fan"] = df["fan_FE"].rank(ascending=False, method="dense").astype("Int64")
    df["rank_judge"] = df["judge_FE"].rank(ascending=False, method="dense").astype("Int64")
    df["rank_gap"] = df["rank_fan"] - df["rank_judge"]

    return df.sort_values(["fan_minus_judge", "fan_FE"], ascending=False)


def run_pro_dancer_effects(
    *,
    judges_model_result,
    fans_model_result,
    out_dir: str = "reports/tables",
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    df = build_pro_dancer_effect_table(judges_model_result, fans_model_result)
    out = os.path.join(out_dir, "pro_dancer_effects_table.csv")
    df.to_csv(out, index=False)

    # Also produce top/bottom slices for memo-friendly inclusion
    top = os.path.join(out_dir, "pro_dancer_effects_top10.csv")
    bot = os.path.join(out_dir, "pro_dancer_effects_bottom10.csv")
    df.head(10).to_csv(top, index=False)
    df.tail(10).to_csv(bot, index=False)

    return {"all": out, "top10": top, "bottom10": bot}
