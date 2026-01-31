# Data folder

## Output tables (from `build_tables.py`)

### OEWS cleaned baseline (levels + wages + wage distribution)
- **oews_baseline.csv** / **oews_baseline.xlsx** — Cleaned OEWS: occ_code, occ_title, area_code, area_title, area_type, emp, a_mean, h_mean, a_median, h_median, a_pct10/25/75/90, oews_year. All geographies; "All occupations" (00-0000) dropped. See **oews_baseline_data_dictionary.md** for units and special values.
- **oews_national.csv** / **oews_national.xlsx** — Same schema, **area_type = national** only (clean link to EP).
- **oews_institution_local.csv** / **oews_institution_local.xlsx** — Same schema, **area_type = state or metro** (for institution-level use).

### Occupation key and EP
- **occ_key.csv** / **occ_key.xlsx** — soc_code, soc_title, **major_group** (2-digit SOC), and placeholder columns for crosswalk IDs (onet_soc, crosswalk_notes). Join key for OEWS and EP (occ_code = soc_code).
- **ep_baseline.csv** / **ep_baseline.xlsx** — Occupation, national only, 10-year horizon (trend & openings). Populated only if BLS EP file is present.

### Join and validation
- **ep_oews_joined.csv** / **ep_oews_joined.xlsx** — EP (national) joined to OEWS (national) on **occ_code** (SOC). One row per occupation present in both; OEWS columns prefixed with `oews_`. Written only when ep_baseline has data.
- **validation/join_validation_report.txt** — Explicit validation: % of EP occupations matched to OEWS, top 10 unmatched EP codes (usually aggregation differences), employment comparison (OEWS vs EP), and limitation note (OEWS establishment wage-and-salary vs EP “all jobs”).
- **validation/ep_unmatched_top10.csv** — Top 10 unmatched EP occupation codes (occ_code, occ_title, emp_2024) for inspection.

## Populating ep_baseline

BLS blocks automated downloads. To build **ep_baseline**:

1. Download [occupation.xlsx](https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx) in a browser.
2. Save it as `data/occupation.xlsx`.
3. Run from the project root: `python build_tables.py`

The script reads **Table 1.10 Occupational separations and openings (projected 2024–2034)** from that file and writes `ep_baseline.csv` and `ep_baseline.xlsx`. See **ep_baseline_data_dictionary.md** for column definitions and BLS concepts.

## Join

Use **soc_code** to join **oews_baseline** and **ep_baseline** via **occ_key**. Both OEWS and EP use 2018 SOC.
