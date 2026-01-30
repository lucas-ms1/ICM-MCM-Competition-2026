"""
J1 (Judges) and J2 (Fans) regressions: pro dancer + celebrity attributes.
Compare whether the same factors impact judges scores vs fan share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults

from src.preprocess import add_zscore_judge
from src.models.vote_latent import build_contestant_week_covariates
from src.fit.fit_elimination import fit
from src.fit.forward_pass import forward_pass_week_by_week, forward_pass_to_dataframe


# Column name in raw CSV for country/region (may contain slash)
RAW_REGION_COL = "celebrity_homecountry/region"
F_REF_EPS = 1e-10


def add_region_to_cw(cw: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add region_us to cw: 1 if United States, else 0.
    Merges from raw on (season, celebrity_name, ballroom_partner).
    Uses celebrity_homecountry/region if present; otherwise region_us = 0.
    """
    cw = cw.copy()
    demo_cols = ["season", "celebrity_name", "ballroom_partner"]
    if RAW_REGION_COL not in raw.columns:
        cw["region_us"] = 0
        return cw
    region_raw = raw[demo_cols + [RAW_REGION_COL]].drop_duplicates()
    region_raw["region_us"] = (
        region_raw[RAW_REGION_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("united states", na=False)
        .astype(int)
    )
    region_raw = region_raw.drop(columns=[RAW_REGION_COL])
    cw = cw.merge(region_raw, on=demo_cols, how="left")
    if "region_us" not in cw.columns:
        cw["region_us"] = 0
    cw["region_us"] = cw["region_us"].fillna(0).astype(int)
    return cw


def _f_ref_geometric_mean(f_df: pd.DataFrame) -> pd.DataFrame:
    """Compute f_ref per (season, week) as geometric mean of f. Returns DataFrame with season, week, f_ref."""
    ref = (
        f_df.groupby(["season", "week"])["f"]
        .apply(lambda x: np.exp(np.log(x.clip(lower=F_REF_EPS)).mean()))
        .rename("f_ref")
        .reset_index()
    )
    return ref


def build_regression_table(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    f_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build contestant-week table ready for J1/J2 regressions.
    - Ensures S (z-score judge) and covariates (age, industry_dummy, region_us, underdog).
    - Restricts to active == 1.
    - If f_df is provided, merges estimated f and adds log_f_ratio = log(f / f_ref) with f_ref = geometric mean per (season, week).
    """
    if "S" not in cw.columns:
        cw = add_zscore_judge(cw)
    cw = build_contestant_week_covariates(cw, raw)
    cw = add_region_to_cw(cw, raw)

    reg = cw[cw["active"] == 1].copy()
    if reg.empty:
        return reg

    if f_df is not None and not f_df.empty:
        f_ref = _f_ref_geometric_mean(f_df)
        f_merge = f_df[["season", "week", "celebrity_name", "ballroom_partner", "f"]].copy()
        f_merge["season"] = f_merge["season"].astype(int)
        f_merge["week"] = f_merge["week"].astype(int)
        f_ref["season"] = f_ref["season"].astype(int)
        f_ref["week"] = f_ref["week"].astype(int)
        f_merge = f_merge.merge(f_ref, on=["season", "week"], how="left")
        f_merge["f_ref"] = f_merge["f_ref"].clip(lower=F_REF_EPS)
        f_merge["log_f_ratio"] = np.log(f_merge["f"] / f_merge["f_ref"])
        reg["season"] = reg["season"].astype(int)
        reg["week"] = reg["week"].astype(int)
        reg = reg.merge(
            f_merge[["season", "week", "celebrity_name", "ballroom_partner", "log_f_ratio", "f"]],
            on=["season", "week", "celebrity_name", "ballroom_partner"],
            how="inner",
        )

    return reg


def fit_judges_model(reg_df: pd.DataFrame) -> tuple[OLSResults, pd.DataFrame]:
    """
    J1: Regress z(J) = S on season FE, pro FE, age, industry_dummy, region_us, week.
    Returns (statsmodels OLS result, summary coefficients DataFrame).
    """
    if reg_df.empty or "S" not in reg_df.columns:
        raise ValueError("reg_df must have active rows and column S")
    reg_df = reg_df.copy()
    reg_df["season"] = reg_df["season"].astype(int)
    reg_df["week"] = pd.to_numeric(reg_df["week"], errors="coerce")
    reg_df["age"] = pd.to_numeric(reg_df["age"], errors="coerce")
    reg_df["industry_dummy"] = pd.to_numeric(reg_df["industry_dummy"], errors="coerce")
    reg_df["region_us"] = pd.to_numeric(reg_df["region_us"], errors="coerce")
    reg_df["S"] = pd.to_numeric(reg_df["S"], errors="coerce")
    reg_df = reg_df.dropna(subset=["S", "week", "age", "industry_dummy", "region_us"])
    if reg_df.empty:
        raise ValueError("No rows left after dropping NaN for judges model")
    y = reg_df["S"].astype(float)
    # Dummies: drop first to avoid collinearity
    season_dummies = pd.get_dummies(reg_df["season"], prefix="season", drop_first=True, dtype=float)
    pro_dummies = pd.get_dummies(reg_df["ballroom_partner"], prefix="pro", drop_first=True, dtype=float)
    X = pd.concat(
        [
            season_dummies,
            pro_dummies,
            reg_df[["week"]],
            reg_df[["age"]],
            reg_df[["industry_dummy"]],
            reg_df[["region_us"]],
        ],
        axis=1,
    )
    X = sm.add_constant(X)
    X = X.astype(float)
    model = sm.OLS(y, X).fit()
    coef_df = _summary_coef_df(model)
    return model, coef_df


def fit_fans_model(reg_df: pd.DataFrame) -> tuple[OLSResults, pd.DataFrame]:
    """
    J2: Regress log(f/f_ref) on season FE, pro FE, age, industry_dummy, region_us, week, S, underdog.
    Returns (statsmodels OLS result, summary coefficients DataFrame).
    """
    if reg_df.empty or "log_f_ratio" not in reg_df.columns:
        raise ValueError("reg_df must have log_f_ratio (run build_regression_table with f_df)")
    reg_df = reg_df.copy()
    reg_df["season"] = reg_df["season"].astype(int)
    reg_df["week"] = pd.to_numeric(reg_df["week"], errors="coerce")
    reg_df["age"] = pd.to_numeric(reg_df["age"], errors="coerce")
    reg_df["industry_dummy"] = pd.to_numeric(reg_df["industry_dummy"], errors="coerce")
    reg_df["region_us"] = pd.to_numeric(reg_df["region_us"], errors="coerce")
    reg_df["S"] = pd.to_numeric(reg_df["S"], errors="coerce")
    reg_df["underdog"] = pd.to_numeric(reg_df["underdog"], errors="coerce")
    reg_df["log_f_ratio"] = pd.to_numeric(reg_df["log_f_ratio"], errors="coerce")
    reg_df = reg_df.dropna(
        subset=["log_f_ratio", "S", "week", "age", "industry_dummy", "region_us", "underdog"]
    )
    if reg_df.empty:
        raise ValueError("No rows left after dropping NaN for fans model")
    y = reg_df["log_f_ratio"].astype(float)
    season_dummies = pd.get_dummies(reg_df["season"], prefix="season", drop_first=True, dtype=float)
    pro_dummies = pd.get_dummies(reg_df["ballroom_partner"], prefix="pro", drop_first=True, dtype=float)
    X = pd.concat(
        [
            season_dummies,
            pro_dummies,
            reg_df[["week"]],
            reg_df[["age"]],
            reg_df[["industry_dummy"]],
            reg_df[["region_us"]],
            reg_df[["S"]],
            reg_df[["underdog"]],
        ],
        axis=1,
    )
    X = sm.add_constant(X)
    X = X.astype(float)
    model = sm.OLS(y, X).fit()
    coef_df = _summary_coef_df(model)
    return model, coef_df


def _summary_coef_df(model: OLSResults) -> pd.DataFrame:
    """Build a DataFrame of coefficients, std err, p-values."""
    return pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "pvalue": model.pvalues,
    })


def compare_judges_fans(
    judges_model: OLSResults,
    fans_model: OLSResults,
) -> tuple[pd.DataFrame, str]:
    """
    Side-by-side comparison of coefficients for common terms (age, industry_dummy, region_us, week).
    Returns (comparison DataFrame, short summary string).
    """
    common = ["week", "age", "industry_dummy", "region_us"]
    j_params = judges_model.params
    f_params = fans_model.params
    j_pval = judges_model.pvalues
    f_pval = fans_model.pvalues
    rows = []
    for name in common:
        if name not in j_params.index or name not in f_params.index:
            continue
        j_coef = j_params[name]
        f_coef = f_params[name]
        j_sig = "***" if j_pval[name] < 0.01 else "**" if j_pval[name] < 0.05 else "*" if j_pval[name] < 0.1 else ""
        f_sig = "***" if f_pval[name] < 0.01 else "**" if f_pval[name] < 0.05 else "*" if f_pval[name] < 0.1 else ""
        same_sign = (j_coef >= 0) == (f_coef >= 0)
        rows.append({
            "covariate": name,
            "judges_coef": j_coef,
            "judges_pvalue": j_pval[name],
            "judges_sig": j_sig,
            "fans_coef": f_coef,
            "fans_pvalue": f_pval[name],
            "fans_sig": f_sig,
            "same_sign": same_sign,
        })
    comp_df = pd.DataFrame(rows)

    # Summary
    if comp_df.empty:
        summary = "No common covariates to compare."
    else:
        same_sign = comp_df["same_sign"].all()
        n_common = len(comp_df)
        summary = (
            f"Compared {n_common} common covariate(s). "
            f"Judges R² = {judges_model.rsquared:.4f}; Fans R² = {fans_model.rsquared:.4f}. "
        )
        if same_sign:
            summary += "All shared coefficients have the same sign across models."
        else:
            summary += "Some coefficients differ in sign between judges and fans models."
    return comp_df, summary


def run_full_pipeline(
    cw: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    beta_init: dict | None = None,
    tau_init: float = 1.0,
) -> dict:
    """
    Run full J1 + J2 pipeline: build covariates, fit elimination model, forward pass,
    build regression table, fit both models, compare.
    Returns dict with keys: reg_df, judges_model, judges_coef_df, fans_model, fans_coef_df,
    comparison_df, comparison_summary, beta_opt, tau_opt.
    """
    if beta_init is None:
        beta_init = {
            "beta0": 0.0,
            "beta_J": 1.0,
            "beta_P": 0.0,
            "beta_U": 0.0,
            "beta_X": np.array([0.0, 0.0]),
        }
    cw = build_contestant_week_covariates(cw, raw)
    beta_opt, tau_opt, fit_result = fit(beta_init, tau_init, cw, raw)
    events = forward_pass_week_by_week(cw, beta_opt)
    f_df = forward_pass_to_dataframe(events)
    reg_df = build_regression_table(cw, raw, f_df=f_df)
    if reg_df.empty:
        return {
            "reg_df": reg_df,
            "judges_model": None,
            "judges_coef_df": pd.DataFrame(),
            "fans_model": None,
            "fans_coef_df": pd.DataFrame(),
            "comparison_df": pd.DataFrame(),
            "comparison_summary": "No active rows for regression.",
            "beta_opt": beta_opt,
            "tau_opt": tau_opt,
            "fit_result": fit_result,
        }
    judges_model, judges_coef_df = fit_judges_model(reg_df)
    fans_model, fans_coef_df = fit_fans_model(reg_df)
    comparison_df, comparison_summary = compare_judges_fans(judges_model, fans_model)
    return {
        "reg_df": reg_df,
        "judges_model": judges_model,
        "judges_coef_df": judges_coef_df,
        "fans_model": fans_model,
        "fans_coef_df": fans_coef_df,
        "comparison_df": comparison_df,
        "comparison_summary": comparison_summary,
        "beta_opt": beta_opt,
        "tau_opt": tau_opt,
        "fit_result": fit_result,
    }
