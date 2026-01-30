"""Main entry point for MCM Problem C analysis.
One run produces all deliverable artifacts: I/J/K tables, vote shares, controversy cases,
judges-save α, pro-dancer effects, regressions, season comparison, K evaluation.
"""

import json
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
    fan_shares_df = forward_pass_to_dataframe(events)[["season", "week", "celebrity_name", "ballroom_partner", "f"]].copy()
    fan_shares_df = fan_shares_df.rename(columns={"f": "fan_share"})
    cf_table = build_counterfactual_table(events)

    # Vote shares (I/J artifact)
    fan_shares_df.to_csv(reports_tables / "fan_shares.csv", index=False)
    print("Wrote: fan_shares.csv (vote shares)")

    # I: Season rule comparison (deliverable #2)
    paths = run_season_compare(panel_df=cw, fan_shares_df=fan_shares_df, out_dir=str(reports_tables))
    print("Wrote:", paths)

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
        k_results["predictability_proposed"].to_csv(reports_tables / "proposed_system_robustness.csv", index=False)


if __name__ == "__main__":
    main()
