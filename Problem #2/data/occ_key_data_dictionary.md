# occ_key data dictionary

## Purpose
Canonical list of occupation codes and titles for joining **oews_baseline** (OEWS) and **ep_baseline** (EP). Built from unique occupations in OEWS; both OEWS and EP use 2018 SOC.

## Columns

| Column | Description |
|--------|-------------|
| soc_code | SOC (Standard Occupational Classification) code, 2018 — e.g. 15-1252. Same as occ_code in OEWS and EP. |
| soc_title | Occupation title from OEWS. |
| major_group | 2-digit SOC major group (e.g. 11, 15). Extracted from soc_code (digits before the hyphen). Useful for aggregation. |
| onet_soc | Placeholder for O*NET-SOC crosswalk ID; fill later if linking to O*NET. |
| crosswalk_notes | Placeholder for other crosswalk IDs (e.g. CIP, CBSA); fill as needed. |

## Join
- **OEWS:** oews_baseline.occ_code = occ_key.soc_code  
- **EP:** ep_baseline.occ_code = occ_key.soc_code  

Both OEWS and EP use the same 2018 SOC codes; Table 1.10 (EP) uses the National Employment Matrix occupation code, which aligns with SOC.
