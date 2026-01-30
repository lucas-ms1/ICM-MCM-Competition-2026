# Go/No-Go Check Report

## 1. One-run reproducibility
- **PASS**: All required CSVs, fitted_params.json, fit_diagnostics.csv, and judges_save_alpha.json present in `reports/tables/`.

## 2. Rule-regime sanity
- **Mapping (paper)**: Percent = seasons 3--27; Rank = 1, 2, 28--34; Judges-save from season 28.
- **PASS**: Seasons map to rules correctly; elimination logic verified for percent, rank, and judges-save.

## 3. Elimination parsing integrity
- **PASS**: For each season, initial cast = finalists + eliminated + withdrew; no contestant active after elimination.

## 4. Fit quality
- **PASS**: Observed eliminated typically has low predicted combined score (percent) or high rank-sum (rank); finals ordering likelihood not degenerate.

## 5. Uncertainty varies by week
- **PASS**: Margin-to-flip (robustness_radius) shows variation across weeks.
- Example: Tight week: Season 10, Week 1 — robustness_radius = 0.0000 (small margin to flip)
- Example: Blowout week: Season 12, Week 3 — robustness_radius = inf (large margin)
