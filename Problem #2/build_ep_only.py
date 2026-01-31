"""
Build ep_baseline and update joins/career views WITHOUT rebuilding OEWS baseline.
Use this when oews_baseline.csv already exists but occupation.xlsx (EP) was just added.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from build_tables import (
    join_ep_oews_national, 
    validate_ep_oews_join, 
    write_join_validation_report,
    DATA_DIR, OUT_DIR
)

EP_PATH = DATA_DIR / "occupation.xlsx"

def _ep_map_table_110_columns(df: pd.DataFrame) -> dict:
    """Map BLS Table 1.10 column names to standard names (flexible match)."""
    col_map = {}
    for c in df.columns:
        s = str(c).lower().replace("\n", " ").replace("\r", " ")
        if "matrix code" in s:
            col_map[c] = "occ_code"
        elif "matrix title" in s:
            col_map[c] = "occ_title"
        elif "employment" in s and "2024" in s and "change" not in s and "percent" not in s:
            col_map[c] = "emp_2024"
        elif "employment" in s and "2034" in s and "change" not in s:
            col_map[c] = "emp_2034"
        elif "change, percent" in s or ("employment change" in s and "percent" in s):
            col_map[c] = "pct_change"
        elif "labor force exits" in s and "rate" not in s:
            col_map[c] = "labor_force_exits"
        elif "occupational transfers" in s and "rate" not in s:
            col_map[c] = "occupational_transfers"
        elif "total occupational separations" in s and "rate" not in s:
            col_map[c] = "total_separations"
        elif "occupational openings" in s:
            col_map[c] = "annual_openings"
    return col_map

def build_ep_baseline_table_110_local() -> pd.DataFrame | None:
    """Pull BLS EP Table 1.10 with header=1 correction."""
    if not EP_PATH.exists():
        print("EP file not found at", EP_PATH)
        return None
    
    xl = pd.ExcelFile(EP_PATH)
    sheet = next((n for n in xl.sheet_names if "1.10" in n), None)
    if not sheet:
        return None
    
    # Use header=1 (row 2)
    df = pd.read_excel(EP_PATH, sheet_name=sheet, header=1)
    print("Columns in Excel:", df.columns.tolist())
    
    col_map = _ep_map_table_110_columns(df)
    print("Column mapping:", col_map)
    df = df.rename(columns={k: v for k, v in col_map.items() if v})
    print("Renamed columns:", df.columns.tolist())
    
    want = [
        "occ_code", "occ_title", "emp_2024", "emp_2034", "pct_change",
        "annual_openings", "labor_force_exits", "occupational_transfers", "total_separations"
    ]
    have = [c for c in want if c in df.columns]
    out = df[have].copy()
    
    for c in have:
        if c not in ["occ_code", "occ_title"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Baseline growth
    if "emp_2024" in out.columns and "emp_2034" in out.columns:
        e24 = out["emp_2024"]
        e34 = out["emp_2034"]
        out["g_baseline"] = np.where(
            (e24 > 0) & (e34 > 0),
            (e34 / e24) ** (1 / 10) - 1,
            np.nan,
        )
    
    if "occ_code" in out.columns:
        out["occ_code"] = out["occ_code"].astype(str).str.strip()
        out = out.loc[out["occ_code"] != "00-0000"].copy()
        
    return out

def main():
    print("Building EP baseline from occupation.xlsx...")
    ep = build_ep_baseline_table_110_local()
    
    if ep is None or ep.empty:
        print("EP baseline build failed. Check data/occupation.xlsx.")
        return

    ep.to_csv(OUT_DIR / "ep_baseline.csv", index=False)
    print("Wrote ep_baseline.csv")

    # Load existing OEWS national
    oews_national_path = OUT_DIR / "oews_national.csv"
    if not oews_national_path.exists():
        print("oews_national.csv not found. Run full build_tables.py.")
        return
    
    oews_national = pd.read_csv(oews_national_path, dtype=str)
    cols_to_numeric = ["emp", "a_mean", "a_median"]
    for c in cols_to_numeric:
        if c in oews_national.columns:
            oews_national[c] = pd.to_numeric(oews_national[c], errors="coerce")

    # Join
    joined = join_ep_oews_national(ep, oews_national)
    joined.to_csv(OUT_DIR / "ep_oews_joined.csv", index=False)
    print(f"Wrote ep_oews_joined.csv: {joined.shape}")

    # Validation
    validation = validate_ep_oews_join(ep, oews_national, joined)
    write_join_validation_report(validation, OUT_DIR / "validation" / "join_validation_report.txt")
    print("Wrote validation report.")

    if validation["top10_unmatched_ep"]:
        pd.DataFrame(validation["top10_unmatched_ep"]).to_csv(
            OUT_DIR / "validation" / "ep_unmatched_top10.csv", index=False
        )

    # Update career views
    careers_dir = OUT_DIR / "careers"
    career_files = ["software_engineer.csv", "electrician.csv", "writer.csv"]
    
    ep_cols = ["occ_code", "emp_2024", "emp_2034", "pct_change", "annual_openings", 
               "labor_force_exits", "occupational_transfers", "total_separations", "g_baseline"]
    ep_sub = ep[[c for c in ep_cols if c in ep.columns]].drop_duplicates(subset=["occ_code"])

    # Rebuild career views from OEWS baseline to ensure we have all geographies (type 4/Metros)
    oews_path = OUT_DIR / "oews_baseline.csv"
    if not oews_path.exists():
        print("oews_baseline.csv not found.")
        return
        
    print("Reading oews_baseline.csv for career views...")
    oews_full = pd.read_csv(oews_path, dtype=str)
    
    CAREER_SOC = {
        "software_engineer": ["15-1252"],
        "electrician": ["47-2111"],
        "writer": ["27-3043"],
    }
    
    for name, socs in CAREER_SOC.items():
        print(f"Rebuilding {name}...")
        mask = oews_full["occ_code"].isin(socs)
        view = oews_full[mask].copy()
        
        if not view.empty:
            overlap = [c for c in ep_sub.columns if c in view.columns and c != "occ_code"]
            if overlap:
                view = view.drop(columns=overlap)
            
            merged = view.merge(ep_sub, on="occ_code", how="left")
            
            out_path = careers_dir / f"{name}.csv"
            merged.to_csv(out_path, index=False)
            print(f"  Saved {name}.csv: {merged.shape}")

if __name__ == "__main__":
    main()
