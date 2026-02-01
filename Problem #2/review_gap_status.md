# Review gap status (as of current repo state)

This note maps the earlier review feedback to what is already implemented in the `Problem #2` pipeline/report, and what is being added next.

## Addressed

- **Summary Sheet numbers inconsistent with Scenario table**: fixed.
  - Summary Sheet now uses the same bundle-level 2034 values as the generated scenario table.
  - See `Problem #2/reports/main.tex` (Summary Sheet bullets; `\item \textbf{Headline findings...}`) and `Problem #2/reports/tables/scenario_summary.tex`.

- **Calibration weights vs “writing-driven” narrative**: reconciled via explicit dual-index framing.
  - Report now defines **NetRisk (uncalibrated/mechanism)** vs **NetRisk (calibrated/predictive)**, and states which index drives scenarios.
  - See `Problem #2/reports/main.tex` (“Definitions & Provenance” box and “External calibration” subsection) and:
    - `Problem #2/reports/tables/netrisk_index_compare.tex`
    - `Problem #2/reports/tables/netrisk_index_disagree.tex`

- **Program sizing / local openings scaling fragility (LQ double-counting)**: addressed with updated method + robustness table.
  - Sizing now uses the standard decomposition (local total employment share × clipped LQ), and reports alternative scalings.
  - See `Problem #2/build_report_artifacts.py` (`build_program_sizing_table()`), `Problem #2/reports/main.tex` (“Local openings scaling robustness”), and `Problem #2/reports/tables/openings_scaling_robustness.tex`.

- **Career bundling clarity**: clarified and documented.
  - National results are explicitly stated as **employment-weighted SOC bundles**, and the bundle membership is shown.
  - See `Problem #2/reports/main.tex` (Summary Sheet + provenance box) and `Problem #2/reports/tables/bundle_contents_fragment.tex`.

- **NetRisk scale clarity (uncalibrated vs centered/rescaled calibrated)**: clarified.
  - Report now indicates “which NetRisk” drives scenarios and provides index comparison tables.
  - See `Problem #2/reports/main.tex` and `Problem #2/reports/tables/scenario_summary.tex`.

## Not fully addressed yet (implemented next)

- **External reality check vs an independent published exposure measure**: missing in current PDF.
  - Next step: correlate NetRisk (calibrated + uncalibrated) against Felten/Raj/Seamans **AIOE** occupation-level exposure and report alignment/discrepancies.

- **Mechanism sensitivity “zscore” labeling**: currently incorrect.
  - `Problem #2/build_mechanism_sensitivity.py` currently labels one option “zscore” but applies min–max scaling.
  - Next step: implement **true z-score** (mapped to [0,1] via normal CDF) and a separate explicit **min–max** option; update the table caption + report text to match.

