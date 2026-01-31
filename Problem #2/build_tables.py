"""
Build oews_baseline, ep_baseline, and occ_key from BLS OEWS (OES) and Employment Projections (EP) data.
Plan: OEWS baseline, EP baseline, and occ_key tables.
OEWS cleaning: levels + wages + wage distribution; standardized units; national + institution_local extracts.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OES_PATH = DATA_DIR / "oesm24all" / "oesm24all" / "all_data_M_2024.xlsx"
EP_PATH = DATA_DIR / "occupation.xlsx"
OUT_DIR = DATA_DIR

# BLS OEWS area type codes vary by file/vintage.
# For the OEWS "all data" extract used here (May 2024), we observe:
#   1 = National (U.S.)
#   2 = State
#   4 = Metropolitan (CBSA)
#   6 = Nonmetropolitan
AREA_TYPE_NATIONAL = 1
AREA_TYPE_STATE = 2
AREA_TYPE_METRO = 4
AREA_TYPE_NONMETRO = 6

# Three careers: STEM (software engineer), trade (electrician), arts (writer). One SOC each (2018 SOC).
CAREER_SOC = {
    "software_engineer": ["15-1252"],   # Software Developers
    "electrician": ["47-2111"],         # Electricians
    "writer": ["27-3043"],              # Writers and Authors
}

# Special values in OEWS wage/employment fields (treated as missing)
OEWS_SPECIAL_VALUES = {"*", "#", "**", "~", "annual not available", "hourly not available", "n/a", ""}


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric; replace special strings with NaN."""
    if s.dtype == object or s.dtype.name == "string":
        s = s.astype(str).str.strip().replace(list(OEWS_SPECIAL_VALUES), np.nan)
    return pd.to_numeric(s, errors="coerce")


def clean_oews_baseline(
    keep_all_occupations: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean OEWS "all data" into baseline: levels + wages + wage distribution.
    Standardize units (annual primary), handle special values, add oews_year.
    Returns (baseline_full, baseline_national, baseline_institution_local).
    """
    print("Reading OES file (this may take a minute)...")
    df = pd.read_excel(OES_PATH, sheet_name="All May 2024 data")
    df = df.loc[df["I_GROUP"] == "cross-industry"].copy()

    # 2.1 Keep only needed columns (occ, area, emp, mean, median, percentiles)
    wage_cols = ["A_MEAN", "H_MEAN", "A_MEDIAN", "H_MEDIAN", "A_PCT10", "A_PCT25", "A_PCT75", "A_PCT90"]
    wage_cols = [c for c in wage_cols if c in df.columns]
    base = df[["AREA", "AREA_TITLE", "AREA_TYPE", "OCC_CODE", "OCC_TITLE", "TOT_EMP"] + wage_cols].copy()

    rename = {
        "AREA": "area_code",
        "AREA_TITLE": "area_title",
        "AREA_TYPE": "area_type",
        "OCC_CODE": "occ_code",
        "OCC_TITLE": "occ_title",
        "TOT_EMP": "emp",
        "A_MEAN": "a_mean",
        "H_MEAN": "h_mean",
        "A_MEDIAN": "a_median",
        "H_MEDIAN": "h_median",
        "A_PCT10": "a_pct10",
        "A_PCT25": "a_pct25",
        "A_PCT75": "a_pct75",
        "A_PCT90": "a_pct90",
    }
    base = base.rename(columns={k: v for k, v in rename.items() if k in base.columns})

    # 2.2 Standardize units and flags
    for c in ["emp"] + [x for x in ["a_mean", "h_mean", "a_median", "h_median", "a_pct10", "a_pct25", "a_pct75", "a_pct90"] if x in base.columns]:
        if c in base.columns:
            base[c] = _to_numeric_series(base[c])

    base["area_code"] = base["area_code"].astype(str).str.strip()
    base["occ_code"] = base["occ_code"].astype(str).str.strip()
    base["oews_year"] = 2024

    # Drop "All occupations" (00-0000) unless needed as denominator
    if not keep_all_occupations:
        base = base.loc[base["occ_code"] != "00-0000"].copy()

    # 2.3 Two geography extracts: national + institution_local (state + metro)
    national = base.loc[base["area_type"] == AREA_TYPE_NATIONAL].copy()
    institution_local = base.loc[base["area_type"].isin([AREA_TYPE_STATE, AREA_TYPE_METRO])].copy()

    return base, national, institution_local


def build_oews_baseline() -> pd.DataFrame:
    """Build oews_baseline (legacy): occupation x geography, latest year. Levels & wages. Uses cleaned table."""
    full, _, _ = clean_oews_baseline(keep_all_occupations=True)
    # Legacy columns for backward compatibility
    out = full.rename(columns={"occ_code": "soc_code", "occ_title": "soc_title", "area_title": "area_name", "emp": "employment", "a_mean": "mean_wage"})
    out = out[["area_code", "area_name", "soc_code", "soc_title", "employment", "mean_wage"]].copy()
    return out


def _soc_to_major_group(soc_code: pd.Series) -> pd.Series:
    """Extract 2-digit major group from SOC code (e.g. 11-1011 -> 11)."""
    return soc_code.astype(str).str.strip().str.split("-").str[0]


def build_occ_key_from_oews(
    oews: pd.DataFrame,
    *,
    occ_code_col: str = "soc_code",
    occ_title_col: str = "soc_title",
    add_major_group: bool = True,
    add_crosswalk_placeholders: bool = True,
) -> pd.DataFrame:
    """
    Build occ_key from unique occupation code + title in OEWS (canonical list).
    Optionally adds major_group (2-digit SOC) and placeholder columns for crosswalk IDs.
    """
    occ = oews[[occ_code_col, occ_title_col]].drop_duplicates().sort_values(occ_code_col).reset_index(drop=True)
    occ = occ.rename(columns={occ_code_col: "soc_code", occ_title_col: "soc_title"})
    if add_major_group:
        occ["major_group"] = _soc_to_major_group(occ["soc_code"])
    if add_crosswalk_placeholders:
        occ["onet_soc"] = ""  # O*NET-SOC crosswalk; fill later if needed
        occ["crosswalk_notes"] = ""  # other IDs (e.g. CBSA, CIP) as needed
    return occ


def _ep_map_table_110_columns(df: pd.DataFrame) -> dict:
    """Map BLS Table 1.10 column names to standard names (flexible match).
    BLS occupation.xlsx Table 1.10 uses header row with e.g. '2024 National Employment Matrix code',
    'Employment, 2024', 'Occupational openings, 2024–34 annual average', etc."""
    col_map = {}
    for c in df.columns:
        s = str(c).lower().replace("\n", " ").replace("\r", " ").replace("\u2013", "-").replace("\u2014", "-")
        if "matrix code" in s or "employment matrix code" in s or ("occupation" in s and "code" in s and "title" not in s):
            col_map[c] = "occ_code"
        elif "matrix title" in s or "employment matrix title" in s or ("occupation" in s and "title" in s):
            col_map[c] = "occ_title"
        elif "employment, 2024" in s or (s.startswith("employment") and "2024" in s and "change" not in s and "2034" not in s):
            col_map[c] = "emp_2024"
        elif "employment, 2034" in s or ("employment" in s and "2034" in s and "change" not in s):
            col_map[c] = "emp_2034"
        elif "change, percent" in s or ("employment change" in s and "percent" in s):
            col_map[c] = "pct_change"
        elif "labor force exits" in s and "annual average" in s:
            col_map[c] = "labor_force_exits"
        elif "occupational transfers" in s and "annual average" in s:
            col_map[c] = "occupational_transfers"
        elif "total occupational separations" in s and "annual average" in s:
            col_map[c] = "total_separations"
        elif "occupational openings" in s or ("openings" in s and "annual" in s):
            col_map[c] = "annual_openings"
    return col_map


def build_ep_baseline_table_110() -> pd.DataFrame | None:
    """
    Pull BLS EP Table 1.10 (Occupational separations and openings, 2024–2034) and build
    business-as-usual trend baseline. Keeps: occ_code, emp_2024, emp_2034, pct_change,
    annual_openings, labor_force_exits, occupational_transfers, total_separations.
    Adds g_baseline = (emp_2034/emp_2024)^(1/10) - 1 (no-GenAI trajectory).
    """
    if not EP_PATH.exists():
        print("EP file not found at", EP_PATH)
        print("Download from https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx and save to data/occupation.xlsx")
        return None
    xl = pd.ExcelFile(EP_PATH)
    tbl110 = next(
        (n for n in xl.sheet_names if "1.10" in n or ("separations" in str(n).lower() and "openings" in str(n).lower())),
        None,
    )
    if not tbl110:
        tbl110 = next((n for n in xl.sheet_names if "1.10" in n or "separations" in str(n).lower()), xl.sheet_names[0])
    # BLS Table 1.10 has a title row (row 0) then column headers (row 1)
    df = pd.read_excel(EP_PATH, sheet_name=tbl110, header=1)
    col_map = _ep_map_table_110_columns(df)
    df = df.rename(columns={k: v for k, v in col_map.items() if v})
    # Keep first occurrence if any duplicate standard names after rename
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    # Keep only requested columns
    want = [
        "occ_code",
        "occ_title",
        "emp_2024",
        "emp_2034",
        "pct_change",
        "annual_openings",
        "labor_force_exits",
        "occupational_transfers",
        "total_separations",
    ]
    have = [c for c in want if c in df.columns]
    out = df[have].copy()

    # Fallback: if occ_code was not mapped (e.g. different Excel layout), try column index 1 (BLS often puts code there)
    if "occ_code" not in out.columns and len(df.columns) >= 2:
        cand = df.iloc[:, 1]
        if cand.astype(str).str.match(r"^\d{2}-\d{4}", na=False).any():
            out["occ_code"] = cand.astype(str).str.strip()
        if "occ_title" not in out.columns and len(df.columns) >= 1:
            out["occ_title"] = df.iloc[:, 0].astype(str).str.strip()
    if "occ_code" not in out.columns:
        print("EP Table 1.10: no occ_code column found after mapping; skipping EP baseline.")
        return None

    # Coerce numerics (BLS employment in thousands)
    for c in ["emp_2024", "emp_2034", "pct_change", "annual_openings", "labor_force_exits", "occupational_transfers", "total_separations"]:
        if c in out.columns:
            ser = out[c].squeeze() if hasattr(out[c], "squeeze") else out[c]
            if hasattr(ser, "iloc"):  # DataFrame -> take first column
                ser = ser.iloc[:, 0] if ser.ndim > 1 else ser
            out[c] = pd.to_numeric(ser, errors="coerce")
    # 3.2 Baseline growth rate (10-year): g_baseline = (emp_2034/emp_2024)^(1/10) - 1
    if "emp_2024" in out.columns and "emp_2034" in out.columns:
        e24 = out["emp_2024"]
        e34 = out["emp_2034"]
        out["g_baseline"] = np.where(
            (e24 > 0) & (e34 > 0),
            (e34 / e24) ** (1 / 10) - 1,
            np.nan,
        )
    out["occ_code"] = out["occ_code"].astype(str).str.strip()
    # Drop footers/source lines and any non-SOC codes (these can appear in the Excel sheet).
    out = out.loc[out["occ_code"].str.match(r"^\d{2}-\d{4}$", na=False)].copy()
    # Drop "All occupations" (00-0000) for consistency with oews_baseline
    out = out.loc[out["occ_code"] != "00-0000"].copy()
    return out if not out.empty and "occ_code" in out.columns else None


def build_ep_baseline() -> pd.DataFrame | None:
    """Build ep_baseline from Table 1.10 (business-as-usual trend). Legacy alias returns same as build_ep_baseline_table_110."""
    return build_ep_baseline_table_110()


def join_ep_oews_national(
    ep: pd.DataFrame,
    oews_national: pd.DataFrame,
    *,
    ep_occ_col: str = "occ_code",
    oews_occ_col: str = "occ_code",
) -> pd.DataFrame:
    """
    Join EP (national) to OEWS (national) on SOC/occ_code.
    OEWS national has one row per occupation (area_type=national); EP has one row per occupation.
    Returns inner join: one row per occupation present in both. OEWS columns get oews_ prefix.
    """
    oews_agg = oews_national.drop(
        columns=[c for c in oews_national.columns if c in ["area_code", "area_title", "area_type"]],
        errors="ignore",
    ).copy()
    oews_agg = oews_agg.rename(columns={oews_occ_col: "occ_code"})
    # Prefix all OEWS columns except join key so merged table has clear EP vs OEWS fields
    oews_agg = oews_agg.rename(columns={c: "oews_" + c for c in oews_agg.columns if c != "occ_code"})
    ep_join = ep.copy()
    if ep_occ_col != "occ_code":
        ep_join = ep_join.rename(columns={ep_occ_col: "occ_code"})
    elif "occ_code" not in ep_join.columns and "soc_code" in ep_join.columns:
        ep_join["occ_code"] = ep_join["soc_code"].astype(str).str.strip()
    merged = ep_join.merge(oews_agg, on="occ_code", how="inner")
    return merged


def validate_ep_oews_join(
    ep: pd.DataFrame,
    oews_national: pd.DataFrame,
    joined: pd.DataFrame,
    *,
    ep_occ_col: str = "occ_code",
    oews_occ_col: str = "occ_code",
) -> dict:
    """
    Run validation checks for EP–OEWS join. Returns dict with:
    - pct_ep_matched: % of EP occupations matched to OEWS
    - n_ep, n_oews, n_joined
    - top10_unmatched: list of up to 10 unmatched EP codes (by emp_2024 if available)
    - employment_comparison: correlation/ratio of OEWS emp vs EP emp_2024 (with limitation note)
    """
    _ep_occ = ep_occ_col if ep_occ_col in ep.columns else ("soc_code" if "soc_code" in ep.columns else "occ_code")
    ep_codes = set(ep[_ep_occ].astype(str).str.strip()) if _ep_occ in ep.columns else set()
    oews_codes = set(oews_national[oews_occ_col].astype(str).str.strip())
    matched_codes = ep_codes & oews_codes
    n_ep = len(ep_codes)
    n_matched = len(matched_codes)
    pct_matched = (n_matched / n_ep * 100) if n_ep else 0

    # Top 10 unmatched EP codes (by employment if available)
    ep_occ = ep[_ep_occ].astype(str).str.strip() if _ep_occ in ep.columns else pd.Series(dtype=object)
    unmatched_mask = ~ep_occ.isin(oews_codes) if _ep_occ in ep.columns else pd.Series(False, index=ep.index)
    unmatched_ep = ep.loc[unmatched_mask].copy()
    if "emp_2024" in unmatched_ep.columns:
        unmatched_ep = unmatched_ep.sort_values("emp_2024", ascending=False)
    top10 = unmatched_ep.head(10)
    top10_list = [
        {"occ_code": row.get(_ep_occ, row.get("occ_code", "")), "occ_title": row.get("occ_title", ""), "emp_2024": row.get("emp_2024", None)}
        for _, row in top10.iterrows()
    ]

    # Employment comparison (on joined rows): OEWS is establishment wage-and-salary, EP is "all jobs"
    emp_corr = None
    emp_ratio_median = None
    if joined is not None and not joined.empty:
        oews_emp_col = "oews_emp" if "oews_emp" in joined.columns else None
        ep_emp_col = "emp_2024" if "emp_2024" in joined.columns else None
        if oews_emp_col and ep_emp_col:
            a = joined[oews_emp_col].astype(float)
            b = joined[ep_emp_col].astype(float)
            valid = (a > 0) & (b > 0)
            if valid.any():
                emp_corr = float(np.corrcoef(a[valid], b[valid])[0, 1])
                emp_ratio_median = float(np.median((a[valid] / b[valid].replace(0, np.nan))))
    return {
        "pct_ep_matched": pct_matched,
        "n_ep": n_ep,
        "n_oews": len(oews_codes),
        "n_joined": len(joined) if joined is not None else 0,
        "top10_unmatched_ep": top10_list,
        "employment_correlation_oews_ep": emp_corr,
        "employment_ratio_median_oews_per_ep": emp_ratio_median,
        "limitation_note": "OEWS is establishment-based wage-and-salary employment; EP is 'all jobs' (includes self-employed, etc.). Levels are not expected to match exactly.",
    }


def write_join_validation_report(results: dict, out_path: Path) -> None:
    """Write validation summary and top 10 unmatched to a text report."""
    lines = [
        "EP–OEWS join validation",
        "========================\n",
        f"EP occupations (total):     {results['n_ep']}",
        f"OEWS national occupations:  {results['n_oews']}",
        f"Joined (matched) rows:      {results['n_joined']}",
        f"% of EP matched to OEWS:    {results['pct_ep_matched']:.2f}%\n",
        "Top 10 unmatched EP occupation codes (often aggregation/detail differences):",
    ]
    for i, row in enumerate(results["top10_unmatched_ep"], 1):
        emp = row.get("emp_2024")
        emp_s = f", emp_2024={emp}" if emp is not None and not (isinstance(emp, float) and np.isnan(emp)) else ""
        lines.append(f"  {i}. {row['occ_code']} — {row.get('occ_title', '')}{emp_s}")
    lines.extend([
        "\nEmployment comparison (joined rows):",
        f"  Correlation (OEWS emp vs EP emp_2024): {results.get('employment_correlation_oews_ep')}",
        f"  Median ratio (OEWS emp / EP emp_2024): {results.get('employment_ratio_median_oews_per_ep')}",
        "\nLimitation: " + results.get("limitation_note", ""),
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_career_view(
    oews_full: pd.DataFrame,
    ep: pd.DataFrame | None,
    soc_codes: list[str],
    *,
    area_types: tuple[int, ...] = (AREA_TYPE_NATIONAL, AREA_TYPE_STATE, AREA_TYPE_METRO),
) -> pd.DataFrame:
    """
    Filter OEWS to one career: chosen SOC(s) and national + state/metro rows.
    Left-join EP (national) on occ_code so each row has EP columns (emp_2024, g_baseline, etc.).
    """
    soc_set = {str(c).strip() for c in soc_codes}
    mask = (oews_full["occ_code"].astype(str).str.strip().isin(soc_set)) & (
        oews_full["area_type"].isin(area_types)
    )
    view = oews_full.loc[mask].copy()
    if ep is not None and not ep.empty and "occ_code" in ep.columns:
        ep_numeric = [c for c in ["emp_2024", "emp_2034", "pct_change", "annual_openings", "labor_force_exits", "occupational_transfers", "total_separations", "g_baseline"] if c in ep.columns]
        if ep_numeric:
            ep_sub = ep[["occ_code"] + ep_numeric].drop_duplicates(subset=["occ_code"])
            view = view.merge(ep_sub, on="occ_code", how="left")
    return view


def build_career_views(
    oews_full: pd.DataFrame,
    ep: pd.DataFrame | None,
    career_soc: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build filtered view per career: national + institution state/metro rows, EP merged. Returns {career_name: df}."""
    career_soc = career_soc or CAREER_SOC
    return {
        name: build_career_view(oews_full, ep, codes)
        for name, codes in career_soc.items()
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean OEWS into baseline (levels + wages + wage distribution), two geography extracts
    full, national, institution_local = clean_oews_baseline(keep_all_occupations=False)
    full.to_csv(OUT_DIR / "oews_baseline.csv", index=False)
    full.to_excel(OUT_DIR / "oews_baseline.xlsx", index=False)
    print("Wrote oews_baseline (cleaned, all geos):", full.shape)
    national.to_csv(OUT_DIR / "oews_national.csv", index=False)
    national.to_excel(OUT_DIR / "oews_national.xlsx", index=False)
    print("Wrote oews_national (area_type=national):", national.shape)
    institution_local.to_csv(OUT_DIR / "oews_institution_local.csv", index=False)
    institution_local.to_excel(OUT_DIR / "oews_institution_local.xlsx", index=False)
    print("Wrote oews_institution_local (state + metro):", institution_local.shape)

    # 2. occ_key (from OEWS unique occupations; soc_code for join to EP)
    occ_key = build_occ_key_from_oews(full, occ_code_col="occ_code", occ_title_col="occ_title")
    occ_key.to_csv(OUT_DIR / "occ_key.csv", index=False)
    occ_key.to_excel(OUT_DIR / "occ_key.xlsx", index=False)
    print("Wrote occ_key:", occ_key.shape)

    # 3. ep_baseline (if EP file present)
    ep = build_ep_baseline()
    if ep is not None and not ep.empty:
        ep.to_csv(OUT_DIR / "ep_baseline.csv", index=False)
        ep.to_excel(OUT_DIR / "ep_baseline.xlsx", index=False)
        print("Wrote ep_baseline:", ep.shape)
    else:
        # Placeholder so structure matches ep_baseline schema
        placeholder_cols = [
            "occ_code", "occ_title", "emp_2024", "emp_2034", "pct_change",
            "annual_openings", "labor_force_exits", "occupational_transfers", "total_separations", "g_baseline",
        ]
        ep = pd.DataFrame(columns=placeholder_cols)
        pd.DataFrame(columns=placeholder_cols).to_csv(OUT_DIR / "ep_baseline.csv", index=False)
        pd.DataFrame(columns=placeholder_cols).to_excel(OUT_DIR / "ep_baseline.xlsx", index=False)
        print("ep_baseline: placeholder (download occupation.xlsx to populate)")

    # 4. Join EP to OEWS (national) on SOC and validate
    ep_has_occ = ep is not None and hasattr(ep, "columns") and ("occ_code" in ep.columns or "soc_code" in ep.columns)
    if ep_has_occ:
        joined = join_ep_oews_national(ep, national)
        if not joined.empty:
            joined.to_csv(OUT_DIR / "ep_oews_joined.csv", index=False)
            joined.to_excel(OUT_DIR / "ep_oews_joined.xlsx", index=False)
            print("Wrote ep_oews_joined:", joined.shape)
        validation = validate_ep_oews_join(ep, national, joined)
        (OUT_DIR / "validation").mkdir(parents=True, exist_ok=True)
        write_join_validation_report(validation, OUT_DIR / "validation" / "join_validation_report.txt")
        print("Wrote validation/join_validation_report.txt")
        # Top 10 unmatched EP codes (for inspection)
        if validation["top10_unmatched_ep"]:
            pd.DataFrame(validation["top10_unmatched_ep"]).to_csv(
                OUT_DIR / "validation" / "ep_unmatched_top10.csv", index=False
            )
            print("Wrote validation/ep_unmatched_top10.csv")

    # 5. Career views: STEM (software engineer), trade (electrician), arts (writer). National + state/metro, EP merged.
    careers_dir = OUT_DIR / "careers"
    careers_dir.mkdir(parents=True, exist_ok=True)
    career_views = build_career_views(full, ep)
    for name, df in career_views.items():
        df.to_csv(careers_dir / f"{name}.csv", index=False)
        df.to_excel(careers_dir / f"{name}.xlsx", index=False)
        print("Wrote careers/", name, ":", df.shape)

    print("Done. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
