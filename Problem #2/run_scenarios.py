"""
Run GenAI impact scenarios for the 3 focus careers.
Model: g_adjusted = g_baseline - s_sub * max(Net_Risk,0) + s_comp * max(-Net_Risk,0)
Net_Risk = (Substitution - Defense)
Substitution = (Writing + Tool_Tech)/2
Defense = (Physical + Social + Creativity)/3
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
    "Moderate_Substitution": 0.015,  # 1.5% annual growth reduction for max risk
    "High_Disruption": 0.03,         # 3% annual growth reduction for max risk
}

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
    
    # Compute Risk Index on the full mechanism mechanism dataframe first (for reference)
    # Norm cols are 0-1 percentiles
    
    # Handle missing cols
    for c in ["writing_intensity", "tool_technology", "physical_manual", "social_perceptiveness", "creativity_originality"]:
        if c not in mech.columns:
            mech[c] = 0.5
    
    mech["substitution_score"] = (mech["writing_intensity"] + mech["tool_technology"]) / 2
    mech["defense_score"] = (mech["physical_manual"] + mech["social_perceptiveness"] + mech["creativity_originality"]) / 3
    mech["net_risk"] = mech["substitution_score"] - mech["defense_score"]
    
    # Save the scored mechanism layer
    mech.to_csv(DATA_DIR / "mechanism_risk_scored.csv", index=False)
    print("Wrote mechanism_risk_scored.csv")

    # 2. Process each career
    careers = ["software_engineer", "electrician", "writer"]
    
    summary_rows = []
    
    for cname in careers:
        df = load_career(cname)
        if df.empty:
            continue
        
        # We need the national row for the primary projection
        # area_type = 1
        national = df[df["area_type"] == 1].copy()
        if national.empty:
            print(f"No national row for {cname}")
            continue
            
        # Merge mechanism scores
        # drop old mechanism cols if they exist in df to avoid dups
        cols_to_drop = [c for c in mech.columns if c in national.columns and c != "occ_code"]
        if cols_to_drop:
            national = national.drop(columns=cols_to_drop)
            
        merged = national.merge(mech, on="occ_code", how="left")
        row = merged.iloc[0]
        occ_title = row["occ_title"]
        # Use EP emp_2024 * 1000 as the national 2024 baseline
        emp_base = row["emp_2024"] * 1000
            
        g_base = row["g_baseline"]
        
        if pd.isna(g_base):
            # Fallback if EP missing
            g_base = 0.0
            
        risk = row["net_risk"]
        
        print(f"\nCareer: {cname} ({occ_title})")
        print(f"  Net Risk: {risk:.3f} (Sub={row['substitution_score']:.3f}, Def={row['defense_score']:.3f})")
        print(f"  Baseline g: {g_base:.4f}")
        
        summary = {
            "career": cname,
            "occ_title": occ_title,
            "net_risk": risk,
            "substitution": row["substitution_score"],
            "defense": row["defense_score"],
            "emp_2024": emp_base,
            "g_baseline": g_base
        }
        
        # Constant shock scenarios (backward compatible)
        for scen, shock in SCENARIOS.items():
            g_adj = get_g_adj(g_base, risk, shock)
            emp_2034 = emp_base * ((1 + g_adj) ** 10)
            
            summary[f"g_{scen}"] = g_adj
            summary[f"emp_2034_{scen}"] = emp_2034
            summary[f"chg_{scen}"] = emp_2034 - emp_base
            
            print(f"  {scen}: g={g_adj:.4f}, emp_2034={emp_2034:,.0f}")
        
        # Dynamic adoption ramp scenarios
        for scen, target_shock in RAMP_SCENARIOS.items():
            emp_2034, avg_g = compute_ramp_scenario(emp_base, g_base, risk, target_shock)
            
            summary[f"g_{scen}"] = avg_g
            summary[f"emp_2034_{scen}"] = emp_2034
            summary[f"chg_{scen}"] = emp_2034 - emp_base
            
            print(f"  {scen}: avg_g={avg_g:.4f}, emp_2034={emp_2034:,.0f}")
            
        summary_rows.append(summary)
        
    # Save summary
    summ_df = pd.DataFrame(summary_rows)
    summ_df.to_csv(OUT_DIR / "scenario_summary.csv", index=False)
    print(f"\nWrote scenario_summary.csv")

    # Sanity check / calibration report
    check_path = OUT_DIR / "validation" / "calibration_check.txt"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(check_path, "w", encoding="utf-8") as f:
        f.write("Scenario Calibration Check\n")
        f.write("==========================\n")
        f.write(f"Equation: g_adj = g_base - s * max(NetRisk, 0) + 0.2*s * max(-NetRisk, 0)\n\n")
        
        for _, row in summ_df.iterrows():
            career = row["career"]
            risk = row["net_risk"]
            f.write(f"Career: {career}, NetRisk: {risk:.3f}\n")
            for scen, shock in SCENARIOS.items():
                if scen == "No_GenAI_Baseline": continue
                g_base = row["g_baseline"]
                g_adj = row[f"g_{scen}"]
                delta_g = g_adj - g_base
                f.write(f"  {scen} (s={shock}): g_base={g_base:.4f} -> g_adj={g_adj:.4f} (delta={delta_g:.4f})\n")


if __name__ == "__main__":
    main()
