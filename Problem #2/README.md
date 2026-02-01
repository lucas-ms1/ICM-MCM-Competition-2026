# ICM 2026 Problem F (Problem #2) — Reproducible pipeline

This folder contains a fully reproducible pipeline that generates the report tables/figures from public labor-market data and O*NET descriptors, and then compiles the LaTeX report.

## Quickstart

1. **Create a Python environment** and install dependencies:

```bash
pip install -r requirements.txt
```

2. **Place required input data** (see below).

3. **Run the full pipeline**:

```bash
python run_all.py
```

4. **Compile the PDF** (from `reports/`):

```bash
cd reports
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The final submission PDF is `reports/main.pdf`.

## Required input data (expected locations)

### 1) BLS OEWS “all data” extract (May 2024)

- **Expected path**: `data/oesm24all/oesm24all/all_data_M_2024.xlsx`
- **Used by**: `build_tables.py` (reads sheet `"All May 2024 data"`)
- **Source**: BLS OEWS “all data” Excel download (May 2024 vintage).

### 2) BLS Employment Projections occupation matrix (2024–2034)

- **Expected path**: `data/occupation.xlsx`
- **Used by**: `build_tables.py` (extracts Table 1.10 and computes `g_baseline`)
- **Source**: BLS EP occupation matrix file (`occupation.xlsx`) linked from the EP tables.

**Unit note**: BLS EP employment and openings are in **thousands**. The scenario code converts EP employment to jobs by multiplying by 1000.

### 3) O*NET Database (text files, v30.1)

- **Expected directory**: `data/onet/`
- **Required files** (minimum):
  - `Work Activities.txt`
  - `Abilities.txt`
  - `Skills.txt`
  - `Occupation Data.txt` (only used as a fallback for titles in a sanity-check table)
- **Used by**: `build_mechanism_layer_expanded.py`

If you have the O*NET “db_30_1_text” zip, extract its contents into `data/onet/` so the above `.txt` files are directly under that folder.

### 4) External AI applicability dataset

- **Expected path**: `data/ai_applicability_scores.csv`
- **Used by**: `build_calibration.py` (calibrates mechanism weights)

This repo assumes an occupation-level SOC keyed dataset with columns:
- `SOC Code`
- `ai_applicability_score`

## What the pipeline produces

Running `python run_all.py` generates:

- **Core artifacts (CSV)** under `data/`
  - `mechanism_layer_all.csv`
  - `mechanism_risk_scored.csv`
  - `scenario_summary.csv`
  - `scenario_parameters.csv`
  - `uncertainty_summary.csv`
  - `policy_decision_scores.csv`, `policy_decision_summary.csv`, `policy_sensitivity.csv`
- **LaTeX-ready tables** under `reports/tables/`
- **Figures** under `reports/figures/`

The LaTeX report (`reports/main.tex`) inputs these tables/figures directly, so every numeric claim is traceable to an artifact.

## Pipeline structure

The top-level orchestrator is `run_all.py`, which runs:

- `build_tables.py` (OEWS/EP baseline tables + career extracts)
- `build_mechanism_layer_expanded.py` (O*NET mechanism layer)
- `build_calibration.py` (optional calibration if applicability file exists)
- `run_scenarios.py` (scenario projections + scenario parameter audit)
- `build_uncertainty.py` (Monte Carlo uncertainty)
- `build_policy_model.py` (policy decision model + sensitivity)
- `build_report_artifacts.py` (renders LaTeX tables + figures)

## Notes

- `reports/ai_use_report.tex` is intentionally left as a separate file to fill immediately before submission.
