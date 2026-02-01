"""
Run GenAI impact scenarios for the 3 focus careers.
Model: g_adjusted = g_baseline - s_sub * max(Net_Risk,0) + s_comp * max(-Net_Risk,0)
Net_Risk = (Substitution - Defense)
Substitution = (Writing + Tool_Tech)/2
Defense = (Physical + Social + Creativity)/3

Updates: Now supports SOC bundles (weighted averaging).
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
CAREERS_DIR = DATA_DIR / "careers"
OUT_DIR = DATA_DIR

# Scenarios: shock_factor
# If Net_Risk is 1.0 (max), growth reduces by X% per year.
SCENARIOS = {
    "No_GenAI_Baseline": 0.0,
    "Moderate_Substitution": 0.015,  # default; may be overwritten by calibration
    "High_Disruption": 0.03,         # default; may be overwritten by calibration
}

# Track provenance of scenario parameters for auditability in the paper.
SCENARIO_SOURCES = {k: "default" for k in SCENARIOS.keys()}

# Dynamic adoption ramp scenarios: target shock factor at 2034
RAMP_SCENARIOS = {
    "Ramp_Moderate": 0.015,  # Linear ramp from 0 to 0.015 over 2024-2034
    "Ramp_High": 0.03,       # Linear ramp from 0 to 0.03 over 2024-2034
}


def get_g_adj(g_base: float, risk: float, shock_factor: float) -> float:
    """
    Compute adjusted growth using piecewise mapping.
    
    g_adj = g_base - s_sub * max(risk, 0) + s_comp * max(-risk, 0)
    
    Where s_sub = shock_factor
          s_comp = shock_factor * 0.2 (conservative complementarity)
    """
    s_sub = shock_factor
    s_comp = shock_factor * 0.2
    
    if risk >= 0:
        return g_base - (s_sub * risk)
    else:
        # risk < 0, so -risk > 0.
        return g_base + (s_comp * (-risk))

def s_t(t: int, target_shock: float, start_year: int = 2024, end_year: int = 2034) -> float:
    """
    Dynamic adoption function: linear ramp from 0 to target_shock.
    
    Args:
        t: Year (2024..2034)
        target_shock: Target shock factor at end_year
        start_year: Starting year (default 2024)
        end_year: Ending year (default 2034)
    
    Returns:
        Shock factor at year t
    """
    if t < start_year:
        return 0.0
    if t >= end_year:
        return target_shock
    # Linear ramp: s(t) = target_shock * (t - start_year) / (end_year - start_year)
    return target_shock * (t - start_year) / (end_year - start_year)


def compute_ramp_scenario(emp_base: float, g_base: float, risk: float, 
                          target_shock: float, start_year: int = 2024, end_year: int = 2034) -> tuple[float, float]:
    """
    Calculate employment in end_year using year-by-year compounding with dynamic adoption.
    
    Formula: E_{t+1} = E_t * (1 + g_baseline - s(t) * NetRisk)
    
    Args:
        emp_base: Employment at start_year
        g_base: Baseline growth rate
        risk: Net risk score
        target_shock: Target shock factor at end_year
        start_year: Starting year (default 2024)
        end_year: Ending year (default 2034)
    
    Returns:
        Tuple of (employment at end_year, average adjusted growth rate)
    """
    emp = emp_base
    total_growth = 0.0
    
    for t in range(start_year, end_year):
        s_current = s_t(t, target_shock, start_year, end_year)
        g_adj = get_g_adj(g_base, risk, s_current)
        emp = emp * (1 + g_adj)
        total_growth += g_adj
    
    # Average growth rate over the period
    avg_g = total_growth / (end_year - start_year)
    
    return emp, avg_g

def load_career(name: str) -> pd.DataFrame:
    path = CAREERS_DIR / f"{name}.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()
    return pd.read_csv(path)

def main():
    # 1. Load mechanism scores
    mech_path = DATA_DIR / "mechanism_layer_all.csv"
    if not mech_path.exists():
        print("Run build_mechanism_layer_expanded.py first.")
        return
    mech = pd.read_csv(mech_path)
    
    # Compute Risk Index on the full mechanism dataframe first (for reference)
    # Norm cols are 0-1 percentiles
    
    # Handle missing cols
    for c in ["writing_intensity", "tool_technology", "physical_manual", "social_perceptiveness", "creativity_originality"]:
        if c not in mech.columns:
            mech[c] = 0.5
    
    mech["substitution_score"] = (mech["writing_intensity"] + mech["tool_technology"]) / 2
    mech["defense_score"] = (mech["physical_manual"] + mech["social_perceptiveness"] + mech["creativity_originality"]) / 3
    mech["net_risk"] = mech["substitution_score"] - mech["defense_score"]

    # If calibrated NetRisk is available, use it for scenarios
    calib_risk_path = DATA_DIR / "mechanism_risk_calibrated.csv"
    if calib_risk_path.exists():
        calib = pd.read_csv(calib_risk_path)
        if "occ_code" in calib.columns and "net_risk_calibrated" in calib.columns:
            mech = mech.merge(
                calib[["occ_code", "net_risk_calibrated"]],
                on="occ_code",
                how="left",
            )
            mech["net_risk_uncalibrated"] = mech["net_risk"]
            mech["net_risk"] = mech["net_risk_calibrated"].combine_first(mech["net_risk"])
            mech["net_risk_source"] = np.where(
                mech["net_risk_calibrated"].notna(), "calibrated", "uncalibrated"
            )
    
    # Save the scored mechanism layer
    mech.to_csv(DATA_DIR / "mechanism_risk_scored.csv", index=False)
    print("Wrote mechanism_risk_scored.csv")

    # If calibrated scenario strengths are available, override defaults
    s_path = DATA_DIR / "calibration_recommended_s.csv"
    s_used_extra_cols: dict[str, dict] = {}
    if s_path.exists():
        try:
            s_df = pd.read_csv(s_path)
            s_map = dict(zip(s_df["scenario"], s_df["s_value"]))
            if "Moderate_Substitution" in s_map:
                SCENARIOS["Moderate_Substitution"] = float(s_map["Moderate_Substitution"])
                SCENARIO_SOURCES["Moderate_Substitution"] = "calibrated"
            if "High_Disruption" in s_map:
                SCENARIOS["High_Disruption"] = float(s_map["High_Disruption"])
                SCENARIO_SOURCES["High_Disruption"] = "calibrated"

            # Store calibration metadata (if present) so the report can cite it.
            for _, r in s_df.iterrows():
                scen = str(r.get("scenario", "")).strip()
                if scen:
                    s_used_extra_cols[scen] = {
                        "target_delta_g_at_p90": r.get("target_delta_g_at_p90"),
                        "p90_positive_netrisk": r.get("p90_positive_netrisk"),
                    }
        except Exception:
            pass

    # Write scenario parameter audit table (used in the LaTeX report).
    param_rows = []
    for scen, s_val in SCENARIOS.items():
        row = {
            "scenario": scen,
            "s_value": float(s_val),
            "source": SCENARIO_SOURCES.get(scen, "default"),
        }
        if scen in s_used_extra_cols:
            row.update(s_used_extra_cols[scen])
        param_rows.append(row)
    for scen, target in RAMP_SCENARIOS.items():
        param_rows.append(
            {
                "scenario": scen,
                "s_value": float(target),
                "source": "default",
            }
        )
    pd.DataFrame(param_rows).to_csv(OUT_DIR / "scenario_parameters.csv", index=False)

    # 2. Process each career
    careers = ["software_engineer", "electrician", "writer"]
    
    summary_rows = []
    
    for cname in careers:
        df = load_career(cname)
        if df.empty:
            continue
        
        # We need ALL national rows for the bundle
        national = df[df["area_type"] == 1].copy()
        if national.empty:
            print(f"No national rows for {cname}")
            continue
            
        # Merge mechanism scores
        cols_to_drop = [c for c in mech.columns if c in national.columns and c != "occ_code"]
        if cols_to_drop:
            national = national.drop(columns=cols_to_drop)
            
        merged = national.merge(mech, on="occ_code", how="left")
        
        # --- Aggregation for Bundle ---
        # Weighted average of NetRisk, Sub, Def based on Emp 2024
        # Sum of Emp 2024, Emp 2034
        # Weighted average g_baseline? Or calculated from aggregated Emp? -> Calculated from Aggregated
        
        # Ensure numeric
        merged["emp_2024_abs"] = merged["emp_2024"] * 1000 # Convert thousands to units
        merged["emp_2034_abs"] = merged["emp_2034"] * 1000
        
        total_emp_24 = merged["emp_2024_abs"].sum()
        total_emp_34_base = merged["emp_2034_abs"].sum()
        
        if total_emp_24 == 0:
            print(f"Zero employment for {cname}, skipping")
            continue
            
        weights = merged["emp_2024_abs"] / total_emp_24
        
        # Fill missing risks with 0 if needed, but better to warn
        if merged["net_risk"].isna().any():
            print(f"Warning: Missing risk scores for some SOCs in {cname}")
            merged["net_risk"] = merged["net_risk"].fillna(0)
            merged["substitution_score"] = merged["substitution_score"].fillna(0)
            merged["defense_score"] = merged["defense_score"].fillna(0)
            
        # Weighted averages
        avg_risk = (merged["net_risk"] * weights).sum()
        avg_sub = (merged["substitution_score"] * weights).sum()
        avg_def = (merged["defense_score"] * weights).sum()
        
        # Aggregated baseline growth
        agg_g_base = (total_emp_34_base / total_emp_24)**(1/10) - 1
        
        # Titles in bundle
        titles = merged["occ_title"].unique()
        main_title = titles[0] if len(titles) > 0 else cname
        if len(titles) > 1:
            main_title = f"{cname} Bundle ({len(titles)} SOCs)"
            
        # Ranges
        min_risk = merged["net_risk"].min()
        max_risk = merged["net_risk"].max()
        
        print(f"\nCareer: {cname} ({main_title})")
        print(f"  Bundle Net Risk: {avg_risk:.3f} (Range: {min_risk:.3f} to {max_risk:.3f})")
        print(f"  Aggregated Baseline g: {agg_g_base:.4f}")
        
        summary = {
            "career": cname,
            "occ_title": main_title,
            "net_risk": avg_risk,
            "net_risk_min": min_risk,
            "net_risk_max": max_risk,
            "substitution": avg_sub,
            "defense": avg_def,
            "emp_2024": total_emp_24,
            "g_baseline": agg_g_base
        }
        
        # Apply scenarios to the AGGREGATE risk
        # Alternative: Apply to each SOC then sum. 
        # Applying to each SOC then summing is more accurate if risk varies significantly.
        # Let's do row-wise projection then sum.
        
        # Constant shock scenarios
        for scen, shock in SCENARIOS.items():
            # Vectorized calculation
            risks = merged["net_risk"]
            g_bases = merged["g_baseline"].fillna(0) 
            # Note: merged["g_baseline"] is per-SOC. If EP missing, use 0.
            
            # Helper for vectorized get_g_adj
            # g_adj = g_base - s_sub * max(risk, 0) + s_comp * max(-risk, 0)
            s_sub = shock
            s_comp = shock * 0.2
            
            term1 = np.maximum(risks, 0) * s_sub
            term2 = np.maximum(-risks, 0) * s_comp
            g_adjs = g_bases - term1 + term2
            
            emps_34 = merged["emp_2024_abs"] * ((1 + g_adjs) ** 10)
            total_emp_34_scen = emps_34.sum()
            
            # Implied aggregate growth rate for summary
            agg_g_scen = (total_emp_34_scen / total_emp_24)**(1/10) - 1
            
            summary[f"g_{scen}"] = agg_g_scen
            summary[f"emp_2034_{scen}"] = total_emp_34_scen
            summary[f"chg_{scen}"] = total_emp_34_scen - total_emp_24
            
            print(f"  {scen}: emp_2034={total_emp_34_scen:,.0f}")
        
        # Ramp scenarios
        for scen, target_shock in RAMP_SCENARIOS.items():
            # Row-wise ramp
            # We need to loop years for each row... slow if done naively.
            # But we only have <10 rows.
            
            row_emps_34 = []
            for _, r in merged.iterrows():
                e24 = r["emp_2024_abs"]
                gb = r["g_baseline"] if not pd.isna(r["g_baseline"]) else 0.0
                rk = r["net_risk"]
                e34, _ = compute_ramp_scenario(e24, gb, rk, target_shock)
                row_emps_34.append(e34)
            
            total_emp_34_ramp = sum(row_emps_34)
            agg_g_ramp = (total_emp_34_ramp / total_emp_24)**(1/10) - 1
            
            summary[f"g_{scen}"] = agg_g_ramp
            summary[f"emp_2034_{scen}"] = total_emp_34_ramp
            summary[f"chg_{scen}"] = total_emp_34_ramp - total_emp_24
            
            print(f"  {scen}: emp_2034={total_emp_34_ramp:,.0f}")
            
        summary_rows.append(summary)
        
    # Save summary
    summ_df = pd.DataFrame(summary_rows)
    summ_df.to_csv(OUT_DIR / "scenario_summary.csv", index=False)
    print(f"\nWrote scenario_summary.csv")

    # Sanity check / calibration report
    check_path = OUT_DIR / "validation" / "calibration_check.txt"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(check_path, "w", encoding="utf-8") as f:
        f.write("Scenario Calibration Check (Bundled)\n")
        f.write("====================================\n")
        f.write(f"Aggregation: Projected SOC-level employment then summed.\n\n")
        
        for _, row in summ_df.iterrows():
            career = row["career"]
            risk = row["net_risk"]
            f.write(f"Career: {career}, Weighted NetRisk: {risk:.3f}\n")
            for scen, shock in SCENARIOS.items():
                if scen == "No_GenAI_Baseline": continue
                emp_base = row["emp_2024"]
                emp_scen = row[f"emp_2034_{scen}"]
                delta = emp_scen - emp_base
                f.write(f"  {scen}: E2024={emp_base:,.0f} -> E2034={emp_scen:,.0f} (delta={delta:,.0f})\n")


if __name__ == "__main__":
    main()
