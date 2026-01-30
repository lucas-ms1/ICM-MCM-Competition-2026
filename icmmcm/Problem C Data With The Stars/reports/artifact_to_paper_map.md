# Artifact-to-Paper Map

Use this as the spine when filling tables and figures in the report. All paths are relative to the repo root; CSVs/JSON live in `reports/tables/`.

| Report element | Artifact(s) | Notes |
|----------------|-------------|--------|
| **Table A (core output)** | `reports/tables/fan_shares.csv` | Weekly fan share per contestant |
| **Table B (uncertainty)** | Bootstrap intervals (from code) + `reports/tables/proposed_system_robustness.csv` | Margin-to-flip: e.g. Season 10 Week 1 radius 0; Season 12 Week 3 radius inf |
| **Table C (method comparison)** | `reports/tables/season_rule_comparison.csv` + `reports/tables/sensitivity_flip_summary.csv` | Rank vs percent by season; judge-weight sensitivity |
| **Table D (controversy)** | `reports/tables/controversy_season{2,4,11,27}_*.csv` | fan_shares_vs_judges + counterfactual_elimination |
| **Table E (drivers)** | `reports/tables/judges_fans_comparison.csv` + `reports/tables/pro_dancer_effects_table.csv` (and top10/bottom10) | Covariates + pro fixed effects |
| **Table F (judges-save)** | `reports/tables/judges_save_alpha.json` | Fitted α for judges-save regime |
| **Table G (proposal)** | `reports/tables/proposed_system_eval.csv` + `reports/tables/proposed_system_robustness.csv` | same_elim, robustness_radius by season/week |
| **Fitted params** | `reports/tables/fitted_params.json` | β (intercept, beta_J, beta_P, beta_U, beta_age, beta_actor), τ, optimizer (success, nfev, final_objective) |
| **Fit diagnostics** | `reports/tables/fit_diagnostics.csv` | Per-(season, week): rule_regime, n_active, observed_eliminated, model vs.\ random vs.\ judges-only (model_elim_prob_observed, map_matches_observed; random_log_prob_observed, random_match_expected; judges_only_log_prob_observed, judges_only_map_matches_observed), rank_of_observed_eliminated |
| **Definition audit** | Report Section 3 ``Definition audit (paper--code alignment)'' | FII = fraction of weeks a poor-judge/high-fan contestant survives (code-aligned). Sensitivity flip = eliminated contestant changes vs *previous* grid point (not vs historical). |

## Quick numbers (for Summary / Memo)

- **Consistency:** From `fit_diagnostics.csv`: match rate (MAP = observed) 38.2% percent-weeks (76/199), 31.8% rank-weeks (20/66); mean log probability ≈ -1.94; rank of true eliminated: 37.7% top-1, 60.8% top-2, 74.3% top-3. Baselines: random (prob 1/n) and judges-only (percent-era c=j; rank-era R=r^J); model match and log loss reported beside baselines; model beats judges-only on percent-era weeks. Fitted β and τ in `fitted_params.json`.
- **Rule comparison:** `season_rule_comparison.csv` — pct_vs_rank_diff_frac, fan_influence_index_percent/rank (FII = fraction of weeks a poor-judge/high-fan contestant survives; see report Definition audit).
- **Sensitivity:** `sensitivity_flip_summary.csv` — flip = eliminated contestant changes vs *previous* grid point (not vs historical). At w_judge=1.0, 26 flips (moving from 0.98 to 1.0); at w_judge=0, 0 flips (no previous point). See report Definition audit (Section 3).
- **Judges-save α:** `judges_save_alpha.json` — alpha ≈ 0.029.
- **Pro effects:** `pro_dancer_effects_top10.csv` — fan_minus_judge (positive = fan-favoring): Henry Byalikov, Andrea Hale; negative: Koko Iwasaki, Ashly DelGrosso.
