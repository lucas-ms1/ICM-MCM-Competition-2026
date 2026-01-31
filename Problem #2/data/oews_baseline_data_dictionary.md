# OEWS baseline data dictionary

## Source
BLS Occupational Employment and Wage Statistics (OEWS), May 2024. Cross-industry only.

## Special values (treated as missing)
The following values in wage or employment fields are coerced to missing (NaN) and documented here:
- `*` — BLS suppression (insufficient data)
- `#` — BLS footnote
- `**` — BLS suppression/footnote
- `~` — BLS notation
- `annual not available` / `hourly not available`
- `n/a`, empty string

## Units
- **emp**: Employment (levels), count.
- **a_*** (e.g. a_mean, a_median, a_pct10): Annual wages, USD.
- **h_*** (e.g. h_mean, h_median): Hourly wages, USD. Kept for reference; annual is the primary unit for modeling.

## Geography
- **area_type**: BLS code — 1 = National, 2 = State, 3 = Metropolitan, 4 = Nonmetropolitan.
- **oews_national**: `area_type == 1` (clean link to EP).
- **oews_institution_local**: `area_type` in (2, 3) — state and metro for institution-level use.

## Aggregates
- "All occupations" (occ_code 00-0000) is dropped by default. Use `keep_all_occupations=True` in `clean_oews_baseline()` if you need it as a denominator.
