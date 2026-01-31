# EP baseline data dictionary

## Source
BLS Employment Projections (EP), **Table 1.10 Occupational separations and openings (projected 2024–2034)**. National employment matrix (2018 SOC).

## BLS definitions (for writeup / model assumptions)

- **Openings** = employment growth/decline (net change in jobs) + **separations**. So annual openings reflect both expansion/replacement demand and turnover.
- **Separations** = **labor force exits** (retirements, etc.) + **occupational transfers** (moves to other occupations). Total separations are the turnover component that, with net employment change, determine openings.

## Columns (cleaned)

| Column | Description | Units / notes |
|--------|-------------|---------------|
| occ_code | Occupation code (EP/National Employment Matrix; 2018 SOC) | e.g. 15-1252 |
| occ_title | Occupation title | |
| emp_2024 | Employment, 2024 | BLS typically in thousands; check source sheet |
| emp_2034 | Employment, 2034 (projected) | Same units as emp_2024 |
| pct_change | Employment change, 2024–34 | Percent |
| annual_openings | Occupational openings, annual average 2024–34 | Count (same scale as employment) |
| labor_force_exits | Labor force exits (annual average) | Count |
| occupational_transfers | Occupational transfers (annual average) | Count |
| total_separations | Total occupational separations (annual average) | labor_force_exits + occupational_transfers |
| g_baseline | Baseline annual growth rate (no-GenAI trajectory) | Dimensionless; see below |

## Baseline growth rate (scenarios)

- **g_baseline** = (emp_2034 / emp_2024)^(1/10) − 1  
- Interpret as the constant annual growth rate that would take 2024 employment to 2034 in the BLS “business-as-usual” projection. Use as the “no GenAI” trajectory in scenario comparisons.
- Computed only where emp_2024 > 0 and emp_2034 > 0; otherwise NaN.

## Join

Use **occ_code** (EP) = **soc_code** in **occ_key** to join with **oews_baseline** (occ_code). Both OEWS and EP use 2018 SOC.
