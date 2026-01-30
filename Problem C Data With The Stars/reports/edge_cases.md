# Edge Cases: How the Pipeline Handles Them

This document lists how the melt-and-contestant-week pipeline handles each edge case explicitly.

**Identifiability of fan votes:** See [reports/identifiability.md](identifiability.md) for what is identifiable (percent → shares only; rank → ordering only) and the reporting convention (primary = f_{i,t}; index-scaled totals for presentation only).

| Edge case | Handling |
|-----------|----------|
| **Weeks with no elimination** | Active set and scores still exist; we do not infer activity from eliminations. `active(i,t)=1` is based purely on presence of a nonzero score in that week. Finals week may be a no-elimination week in some seasons. |
| **Weeks with double elimination** | Two (or more) contestants can share the same `elimination_week`. Both get 0s from that week onward. No special aggregation change; each contestant–week row is computed independently. |
| **Seasons with different number of weeks** | We do not assume a fixed 11 weeks. Max week per season is derived from the data. Finals week and `elimination_week` are per-season. |
| **Sometimes 3 judges, sometimes 4 (N/A)** | Melt keeps N/A as missing (NaN). `num_judges` = count of non-NaN scores per (contestant, week). J_{i,t} is the sum over available judges only. |
| **0s after elimination** | Treated as structural (contestant no longer competing). Used only to infer `elimination_week` / active set; not as “judge gave 0” in fairness or scoring metrics. |
| **Withdrew** | Parsed separately from "Eliminated Week k". We set `elimination_week` to the last week in which the contestant had a nonzero score (i.e. week of last appearance). |
| **Decimals / bonus scores** | Stored in the long table as numeric values. Contestant–week has `has_bonus` flag (1 if any score is non-integer or > 10). J_{i,t} uses the sum of numeric values as-is. |

## Scoring (combination rules)

| Topic | Convention |
|-------|------------|
| **Rank ties** | Average ranks: `scipy.stats.rankdata(..., method="average")`. Sensitivity to min/max rank can be quantified later. |
| **Percent method: sum(J)=0 or sum(V)=0** | Uniform: j_i = 1/n or f_i = 1/n so combined c_i = j_i + f_i is defined. |
| **Elimination tie (argmin c or argmax R)** | First index with minimum (percent) or maximum (rank) value. |
