# Career views (3 chosen careers)

Filtered views from the merged baseline for modeling and narrative: **software engineer** (STEM), **electrician** (trade), **writer** (arts).

## SOC mapping (2018 SOC, one per career)

| Career             | SOC code  | BLS title             |
|--------------------|-----------|------------------------|
| software_engineer  | 15-1252   | Software Developers   |
| electrician        | 47-2111   | Electricians           |
| writer             | 27-3043   | Writers and Authors    |

## Contents of each view

- **Rows:** National (area_type = 1) + state (2) + metro (3) from OEWS for that occupation. So you get one national row plus all state/metro rows where the occupation has data—ideal for comparing national vs. an institution’s state/metro.
- **Columns:** All OEWS baseline columns (area_code, area_title, area_type, occ_code, occ_title, emp, a_mean, h_mean, a_median, wage percentiles, oews_year) plus EP columns when available: emp_2024, emp_2034, pct_change, annual_openings, labor_force_exits, occupational_transfers, total_separations, g_baseline.

## Use

- **National row:** area_type = 1 (one row per career).
- **Institution’s state/metro:** Filter to the institution’s area_code or area_title (state or metro) for local wages and employment.

EP columns are national-only; they repeat on every geography row for that occupation.

## Mechanism layer (GenAI “why” variables)

To add O*NET-based mechanism scores (writing_intensity, social_perceptiveness, physical_manual, creativity_originality, tool_technology), merge **mechanism_layer.csv** on **occ_code**:

```python
career = pd.read_csv("data/careers/software_engineer.csv")
mechanism = pd.read_csv("data/mechanism_layer.csv")
career = career.merge(mechanism.drop(columns=["occ_title"]), on="occ_code", how="left")
```

See **mechanism_layer_README.md** and `python build_mechanism_layer.py`.
