"""Main entry point for MCM Problem C analysis.
One run produces all deliverable artifacts: I/J/K tables, vote shares, controversy cases,
judges-save α, pro-dancer effects, regressions, season comparison, K evaluation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.io import load_raw_data, save_long, save_contestant_week
from src.preprocess import run_pipeline
from src.models.vote_latent import build_contestant_week_covariates
from src.fit.forward_pass import forward_pass_week_by_week, forward_pass_to_dataframe
from src.fit.fit_elimination import (
    build_elimination_events,
    compute_fit_diagnostics,
    fitted_params_to_dict,
)
from src.fit.uncertainty import bootstrap_fan_share_intervals
from src.fit.judges_save_alpha import fit_alpha_and_save
from src.analysis.counterfactual_engine import build_counterfactual_table
from src.analysis.judges_fans_regressions import run_full_pipeline
from src.analysis.pro_dancer_effects import run_pro_dancer_effects
from src.analysis.proposed_system_eval import run_k_evaluation
from src.analysis.season_compare import run_season_compare
from src.analysis.controversy_cases import run_all_controversy_cases


def main() -> None:
    base = Path(__file__).resolve().parent
    data_path = base / "data" / "2026_MCM_Problem_C_Data.csv"

    raw = load_raw_data(str(data_path))
    long_df, cw = run_pipeline(raw)

    # Optional export
    save_long(long_df, str(base / "reports" / "tables" / "long_scores.csv"))
    save_contestant_week(cw, str(base / "reports" / "tables" / "contestant_week.csv"))

    n_elim = int(cw.drop_duplicates(subset=["season", "celebrity_name", "ballroom_partner"])["elimination_week"].notna().sum())
    n_seasons = cw["season"].nunique()
    print(f"Long: {long_df.shape[0]} rows; contestant-week: {cw.shape[0]} rows; {n_seasons} seasons; {n_elim} contestants with elimination_week set.")

    # J1/J2 regressions: judges vs fans (pro + celebrity attributes)
    results = run_full_pipeline(cw, raw)
    print("\n--- Judges vs Fans regressions (J1, J2) ---")
    print(results["comparison_summary"])
    if not results["comparison_df"].empty:
        print("\nComparison (common covariates):")
        print(results["comparison_df"].to_string(index=False))
    if results["judges_model"] is not None:
        print("\nJudges model (J1) R² =", round(results["judges_model"].rsquared, 4))
    if results["fans_model"] is not None:
        print("Fans model (J2) R² =", round(results["fans_model"].rsquared, 4))
    reports_tables = base / "reports" / "tables"
    reports_tables.mkdir(parents=True, exist_ok=True)
    # Fitted params (β, τ, optimizer) for reproducibility
    fitted_params = fitted_params_to_dict(
        results["beta_opt"], results["tau_opt"], results["fit_result"]
    )
    with open(reports_tables / "fitted_params.json", "w", encoding="utf-8") as f:
        json.dump(fitted_params, f, indent=2)
    print("Wrote: fitted_params.json")
    if not results["reg_df"].empty:
        results["reg_df"].to_csv(reports_tables / "regression_contestant_week.csv", index=False)
    if not results["comparison_df"].empty:
        results["comparison_df"].to_csv(reports_tables / "judges_fans_comparison.csv", index=False)
    if results["judges_model"] is not None and results["fans_model"] is not None:
        pro_paths = run_pro_dancer_effects(
            judges_model_result=results["judges_model"],
            fans_model_result=results["fans_model"],
            out_dir=str(reports_tables),
        )
        print("Pro dancer effects table wrote:", pro_paths)

    # Forward pass → fan_shares_df (vote shares), counterfactual table
    cw_aug = build_contestant_week_covariates(cw, raw)
    events = forward_pass_week_by_week(cw_aug, results["beta_opt"])
    # Per-(season, week) fit diagnostics for consistency narrative
    elimination_events = build_elimination_events(cw_aug)
    fit_diag_df = compute_fit_diagnostics(
        results["beta_opt"], results["tau_opt"], elimination_events
    )
    fit_diag_df.to_csv(reports_tables / "fit_diagnostics.csv", index=False)
    print("Wrote: fit_diagnostics.csv")
    # Fit diagnostics summary (LaTeX macros) so the report never hard-codes these numbers.
    try:
        df = fit_diag_df.copy()
        # Ensure booleans are numeric
        df["map_matches_observed"] = df["map_matches_observed"].astype(int)
        df["judges_only_map_matches_observed"] = df["judges_only_map_matches_observed"].astype(int)
        # Overall
        n_all = int(len(df))
        mean_logp_model = float(df["model_log_prob_observed"].mean()) if n_all else float("nan")
        mean_logp_judges = float(df["judges_only_log_prob_observed"].mean()) if n_all else float("nan")
        mean_logp_random = float(df["random_log_prob_observed"].mean()) if n_all else float("nan")
        match_model = float(df["map_matches_observed"].mean()) if n_all else float("nan")
        match_judges = float(df["judges_only_map_matches_observed"].mean()) if n_all else float("nan")
        rank_mean = float(df["rank_of_observed_eliminated"].mean()) if n_all else float("nan")
        cov1 = float((df["rank_of_observed_eliminated"] <= 1).mean()) if n_all else float("nan")
        cov2 = float((df["rank_of_observed_eliminated"] <= 2).mean()) if n_all else float("nan")
        cov3 = float((df["rank_of_observed_eliminated"] <= 3).mean()) if n_all else float("nan")

        def _by_rule(rule: str) -> dict:
            d = df[df["rule_regime"] == rule]
            n = int(len(d))
            return {
                "n": n,
                "match_model": float(d["map_matches_observed"].mean()) if n else float("nan"),
                "match_judges": float(d["judges_only_map_matches_observed"].mean()) if n else float("nan"),
                "mean_logp_model": float(d["model_log_prob_observed"].mean()) if n else float("nan"),
                "mean_logp_judges": float(d["judges_only_log_prob_observed"].mean()) if n else float("nan"),
                "mean_logp_random": float(d["random_log_prob_observed"].mean()) if n else float("nan"),
                "cov1": float((d["rank_of_observed_eliminated"] <= 1).mean()) if n else float("nan"),
                "cov2": float((d["rank_of_observed_eliminated"] <= 2).mean()) if n else float("nan"),
                "cov3": float((d["rank_of_observed_eliminated"] <= 3).mean()) if n else float("nan"),
            }

        pct_stats = _by_rule("percent")
        rank_stats = _by_rule("rank")

        fit_tex = reports_tables / "fit_diagnostics_summary.tex"
        fit_tex.write_text(
            "\n".join([
                "% Auto-generated by main.py from fit_diagnostics.csv",
                "% Do not edit by hand; re-run python main.py to update.",
                f"\\providecommand{{\\FitDiagNWeeks}}{{{n_all}}}",
                f"\\providecommand{{\\FitDiagMeanLogPModel}}{{{mean_logp_model:.3f}}}",
                f"\\providecommand{{\\FitDiagMeanLogPJudgesOnly}}{{{mean_logp_judges:.3f}}}",
                f"\\providecommand{{\\FitDiagMeanLogPRandom}}{{{mean_logp_random:.3f}}}",
                f"\\providecommand{{\\FitDiagMAPMatchModelPct}}{{{100.0*match_model:.1f}}}",
                f"\\providecommand{{\\FitDiagMAPMatchJudgesOnlyPct}}{{{100.0*match_judges:.1f}}}",
                f"\\providecommand{{\\FitDiagTrueRankMean}}{{{rank_mean:.2f}}}",
                f"\\providecommand{{\\FitDiagTrueRankLEOnePct}}{{{100.0*cov1:.1f}}}",
                f"\\providecommand{{\\FitDiagTrueRankLETwoPct}}{{{100.0*cov2:.1f}}}",
                f"\\providecommand{{\\FitDiagTrueRankLEThreePct}}{{{100.0*cov3:.1f}}}",
                # By rule regime
                f"\\providecommand{{\\FitDiagNPercent}}{{{pct_stats['n']}}}",
                f"\\providecommand{{\\FitDiagMAPMatchPercentModelPct}}{{{100.0*pct_stats['match_model']:.1f}}}",
                f"\\providecommand{{\\FitDiagMAPMatchPercentJudgesOnlyPct}}{{{100.0*pct_stats['match_judges']:.1f}}}",
                f"\\providecommand{{\\FitDiagMeanLogPPercentModel}}{{{pct_stats['mean_logp_model']:.3f}}}",
                f"\\providecommand{{\\FitDiagNRank}}{{{rank_stats['n']}}}",
                f"\\providecommand{{\\FitDiagMAPMatchRankModelPct}}{{{100.0*rank_stats['match_model']:.1f}}}",
                f"\\providecommand{{\\FitDiagMAPMatchRankJudgesOnlyPct}}{{{100.0*rank_stats['match_judges']:.1f}}}",
                f"\\providecommand{{\\FitDiagMeanLogPRankModel}}{{{rank_stats['mean_logp_model']:.3f}}}",
                "",
            ]),
            encoding="utf-8",
        )
        print("Wrote: fit_diagnostics_summary.tex")
    except Exception as e:
        print(f"WARNING: could not write fit_diagnostics_summary.tex: {e}")

    beta_init = {
        "beta0": 0.0,
        "beta_J": 1.0,
        "beta_P": 0.0,
        "beta_U": 0.0,
        "beta_X": np.array([0.0, 0.0]),
    }

    # Optional holdout evaluation: evaluate last 3 seasons
    seasons_sorted = sorted(cw_aug["season"].dropna().unique())
    holdout_seasons = seasons_sorted[-3:] if len(seasons_sorted) >= 3 else []
    if holdout_seasons:
        cw_train = cw_aug[~cw_aug["season"].isin(holdout_seasons)]
        cw_holdout = cw_aug[cw_aug["season"].isin(holdout_seasons)]
        if not cw_train.empty and not cw_holdout.empty:
            try:
                holdout_events = build_elimination_events(cw_holdout)
                holdout_diag = compute_fit_diagnostics(
                    results["beta_opt"], results["tau_opt"], holdout_events
                )
                holdout_diag.to_csv(
                    reports_tables / "fit_diagnostics_holdout.csv", index=False
                )
                print("Wrote: fit_diagnostics_holdout.csv")
            except Exception as e:
                print(f"WARNING: holdout diagnostics failed: {e}")

    # Uncertainty: bootstrap intervals for fan shares
    intervals_df, _ = bootstrap_fan_share_intervals(
        cw_aug, raw, beta_init, tau_init=1.0, n_bootstrap=2, random_state=0
    )
    if not intervals_df.empty:
        intervals_df.to_csv(
            reports_tables / "fan_shares_bootstrap_intervals.csv", index=False
        )
        print("Wrote: fan_shares_bootstrap_intervals.csv")
        width = intervals_df["f_upper"] - intervals_df["f_lower"]
        summary_tex = reports_tables / "fan_shares_bootstrap_summary.tex"
        summary_tex.write_text(
            "\n".join([
                "% Auto-generated by main.py from fan_shares_bootstrap_intervals.csv",
                "% Do not edit by hand; re-run python main.py to update.",
                f"\\providecommand{{\\BootstrapNCells}}{{{int(len(width))}}}",
                f"\\providecommand{{\\BootstrapWidthMedian}}{{{float(width.median()):.4f}}}",
                f"\\providecommand{{\\BootstrapWidthQOne}}{{{float(width.quantile(0.25)):.4f}}}",
                f"\\providecommand{{\\BootstrapWidthQThree}}{{{float(width.quantile(0.75)):.4f}}}",
                f"\\providecommand{{\\BootstrapWidthP95}}{{{float(width.quantile(0.95)):.4f}}}",
                "",
            ]),
            encoding="utf-8",
        )
        print("Wrote: fan_shares_bootstrap_summary.tex")
    fan_shares_df = forward_pass_to_dataframe(events)[["season", "week", "celebrity_name", "ballroom_partner", "f"]].copy()
    fan_shares_df = fan_shares_df.rename(columns={"f": "fan_share"})
    cf_table = build_counterfactual_table(events)

    # Vote shares (I/J artifact)
    fan_shares_df.to_csv(reports_tables / "fan_shares.csv", index=False)
    print("Wrote: fan_shares.csv (vote shares)")

    # I: Season rule comparison (deliverable #2)
    paths = run_season_compare(panel_df=cw, fan_shares_df=fan_shares_df, out_dir=str(reports_tables))
    print("Wrote:", paths)
    # Build LaTeX table snippets for rule comparison summaries
    try:
        rule_df = pd.read_csv(reports_tables / "season_rule_comparison.csv")
        if not rule_df.empty:
            def _pct(x: float) -> str:
                return f"{100.0 * x:.1f}\\%"
            def _fmt(x: float) -> str:
                return f"{x:.3f}"
            summary_rows = []
            for label, func in [("Min", "min"), ("Median", "median"), ("Mean", "mean"), ("Max", "max")]:
                stats = getattr(rule_df, func)()
                summary_rows.append(
                    f"{label} & {_pct(stats['pct_vs_rank_diff_frac'])} & "
                    f"{_pct(stats['pct_vs_judgessave_diff_frac'])} & {_pct(stats['rank_vs_judgessave_diff_frac'])} & "
                    f"{_fmt(stats['fan_influence_index_percent'])} / {_fmt(stats['fan_influence_index_rank'])}\\\\"
                )
            (reports_tables / "season_rule_comparison_summary.tex").write_text(
                "\n".join(summary_rows) + "\n",
                encoding="utf-8",
            )
            top = rule_df.sort_values("pct_vs_rank_diff_frac", ascending=False).head(4)
            top_rows = []
            for _, row in top.iterrows():
                top_rows.append(
                    f"{int(row['season'])} & {int(row['n_elim_weeks'])} & {_pct(row['pct_vs_rank_diff_frac'])} & "
                    f"{_fmt(row['fan_influence_index_percent'])} / {_fmt(row['fan_influence_index_rank'])}\\\\"
                )
            (reports_tables / "season_rule_comparison_top.tex").write_text(
                "\n".join(top_rows) + "\n",
                encoding="utf-8",
            )
    except Exception as e:
        print(f"WARNING: could not write rule comparison LaTeX snippets: {e}")

    # I3: Controversy case studies (optional but recommended)
    controversy = run_all_controversy_cases(cw_aug, results["beta_opt"], raw)
    for season, tables in controversy.items():
        for name, df in tables.items():
            if hasattr(df, "to_csv") and not df.empty:
                out = reports_tables / f"controversy_season{season}_{name}.csv"
                df.to_csv(out, index=False)
    if controversy:
        print("Wrote: controversy_season* CSVs")

    # D3: Fit α (post-hoc judges-save), write judges_save_alpha.json + print α
    alpha_result = fit_alpha_and_save(
        events, cf_table, cw,
        reports_tables / "judges_save_alpha.json",
        alpha_init=1.0,
    )
    print(f"Judges-save alpha = {alpha_result.alpha:.4f} (n_obs={alpha_result.n_obs}, neg_loglik={alpha_result.neg_loglik:.2f})")
    print("Wrote: judges_save_alpha.json")
    # Also export LaTeX macros so the report can pull α dynamically (avoid hard-coded numbers).
    alpha_tex = reports_tables / "judges_save_alpha.tex"
    alpha_tex.write_text(
        "\n".join([
            "% Auto-generated by main.py from judges_save_alpha.json",
            "% Do not edit by hand; re-run python main.py to update.",
            f"\\providecommand{{\\AlphaJudgesSave}}{{{alpha_result.alpha:.4f}}}",
            f"\\providecommand{{\\AlphaJudgesSaveNObs}}{{{int(alpha_result.n_obs)}}}",
            f"\\providecommand{{\\AlphaJudgesSaveNegLogLik}}{{{alpha_result.neg_loglik:.2f}}}",
            "",
        ]),
        encoding="utf-8",
    )
    print("Wrote: judges_save_alpha.tex")

    # K: Proposed better system (axiomatic + testable)
    k_results = run_k_evaluation(
        cw,
        raw,
        beta=results.get("beta_opt"),
        proposed_rule="weighted_saturation",
        w=0.5,
        alpha=0.8,
        compute_predictability=True,
    )
    print("\n--- Proposed system (K): weighted percent with saturation ---")
    cs = k_results.get("changes_summary", {})
    if cs:
        print(f"Weeks changed vs historical: {cs.get('n_changed', 0)} / {cs.get('n_weeks', 0)} ({cs.get('frac_changed', 0):.2%})")
    cv = k_results.get("controversy_summary", {})
    if cv:
        print(f"Controversy mismatches: observed {cv.get('n_mismatch_observed', 0)}, proposed {cv.get('n_mismatch_proposed', 0)} (reduction {cv.get('reduction', 0)}, {cv.get('reduction_pct', 0):.1f}%)")
    if not k_results.get("eval_df").empty:
        k_results["eval_df"].to_csv(reports_tables / "proposed_system_eval.csv", index=False)
    if not k_results.get("predictability_proposed").empty:
        predict_proposed = k_results["predictability_proposed"]
        predict_proposed.to_csv(
            reports_tables / "proposed_system_robustness.csv", index=False
        )
        rad = predict_proposed["robustness_radius"].astype(float)
        n_total = int(len(rad))
        n_zero = int((rad == 0).sum())
        n_inf = int(np.isinf(rad).sum())
        finite = rad[np.isfinite(rad)]
        finite_nonneg = finite[finite >= 0]
        if not finite_nonneg.empty:
            med = float(finite_nonneg.median())
            q1 = float(finite_nonneg.quantile(0.25))
            q3 = float(finite_nonneg.quantile(0.75))
            max_val = float(finite_nonneg.max())
        else:
            med = q1 = q3 = max_val = float("nan")
        summary_tex = reports_tables / "robustness_summary.tex"
        summary_tex.write_text(
            "\n".join([
                "% Auto-generated by main.py from proposed_system_robustness.csv",
                "% Do not edit by hand; re-run python main.py to update.",
                f"\\providecommand{{\\RobustNWeeks}}{{{n_total}}}",
                f"\\providecommand{{\\RobustZeroCount}}{{{n_zero}}}",
                f"\\providecommand{{\\RobustZeroPct}}{{{(100.0*n_zero/n_total) if n_total else 0.0:.1f}}}",
                f"\\providecommand{{\\RobustInfCount}}{{{n_inf}}}",
                f"\\providecommand{{\\RobustInfPct}}{{{(100.0*n_inf/n_total) if n_total else 0.0:.1f}}}",
                f"\\providecommand{{\\RobustMedian}}{{{med:.2f}}}",
                f"\\providecommand{{\\RobustQOne}}{{{q1:.2f}}}",
                f"\\providecommand{{\\RobustQThree}}{{{q3:.2f}}}",
                f"\\providecommand{{\\RobustMax}}{{{max_val:.2f}}}",
                "",
            ]),
            encoding="utf-8",
        )
    if not k_results.get("predictability_current").empty:
        k_results["predictability_current"].to_csv(reports_tables / "current_system_robustness.csv", index=False)

    # Parameter sensitivity (lightweight grid): scope of change vs (w, alpha)
    sensitivity_rows = []
    for w_grid in [0.4, 0.6]:
        for alpha_grid in [0.7, 0.9]:
            k_sens = run_k_evaluation(
                cw,
                raw,
                beta=results.get("beta_opt"),
                proposed_rule="weighted_saturation",
                w=w_grid,
                alpha=alpha_grid,
                compute_predictability=False,
            )
            cs = k_sens.get("changes_summary", {})
            sensitivity_rows.append({
                "w_judge": w_grid,
                "alpha": alpha_grid,
                "n_weeks": cs.get("n_weeks", 0),
                "n_changed": cs.get("n_changed", 0),
                "frac_changed": cs.get("frac_changed", 0.0),
            })
    if sensitivity_rows:
        pd.DataFrame(sensitivity_rows).to_csv(
            reports_tables / "proposed_system_sensitivity.csv", index=False
        )


if __name__ == "__main__":
    main()
