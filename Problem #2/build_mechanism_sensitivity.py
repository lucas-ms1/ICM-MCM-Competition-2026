"""
Build sensitivity analysis for the O*NET mechanism layer.
Perturbs descriptors (leave-one-out) and normalization methods.
Outputs:
  - data/mechanism_sensitivity.csv
  - reports/tables/mechanism_sensitivity.tex
"""
from pathlib import Path
import pandas as pd
import numpy as np
import math

DATA_DIR = Path(__file__).resolve().parent / "data"
ONET_DIR = DATA_DIR / "onet"
OUT_DIR = DATA_DIR
TABLES_DIR = Path(__file__).resolve().parent / "reports" / "tables"

# 5 dimensions from original script
DIMENSION_DESCRIPTORS = {
    "writing_intensity": [
        "Documenting/Recording Information",
        "Written Expression",
        "Writing",
        "Performing Administrative Activities",
    ],
    "social_perceptiveness": [
        "Establishing and Maintaining Interpersonal Relationships",
        "Assisting and Caring for Others",
        "Social Perceptiveness",
        "Communicating with People Outside the Organization",
        "Performing for or Working Directly with the Public",
    ],
    "physical_manual": [
        "Handling and Moving Objects",
        "Performing General Physical Activities",
        "Static Strength",
        "Stamina",
        "Manual Dexterity",
        "Trunk Strength",
        "Arm-Hand Steadiness",
        "Finger Dexterity",
        "Repairing and Maintaining Mechanical Equipment",
        "Repairing and Maintaining Electronic Equipment",
    ],
    "creativity_originality": [
        "Thinking Creatively",
        "Originality",
        "Fluency of Ideas",
    ],
    "tool_technology": [
        "Working with Computers",
        "Programming",
        "Technology Design",
        "Interacting With Computers",
    ],
}

FOCAL_CAREERS = {
    "15-1252": "Software Dev",
    "47-2111": "Electrician",
    "27-3043": "Writer",
}

def load_onet_file(filename: str) -> pd.DataFrame:
    path = ONET_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", on_bad_lines='skip')
    if "Scale ID" in df.columns:
        df = df[df["Scale ID"] == "IM"].copy()
    return df[["O*NET-SOC Code", "Element ID", "Element Name", "Data Value"]]

def compute_net_risk(mech_df: pd.DataFrame, normalization: str = "percentile") -> pd.DataFrame:
    """
    Compute NetRisk given a raw mechanism dataframe (SOC x Dimensions).
    normalization:
      - 'percentile': percentile ranks in [0,1]
      - 'zscore': true z-scores mapped to [0,1] via Normal CDF (erf)
      - 'minmax': min-max scaling in [0,1]
      - 'rank_within_group': percentile ranks within 2-digit SOC major group
    """
    df = mech_df.copy()
    dims = list(DIMENSION_DESCRIPTORS.keys())
    
    # 1. Normalize
    if normalization == "percentile":
        for d in dims:
            if d in df.columns:
                df[f"norm_{d}"] = df[d].rank(pct=True)
            else:
                df[f"norm_{d}"] = 0.5
                
    elif normalization == "zscore":
        for d in dims:
            if d in df.columns:
                mu = df[d].mean()
                sig = df[d].std()
                if sig == 0 or (isinstance(sig, float) and math.isnan(sig)):
                    sig = 1.0
                z = (df[d] - mu) / sig
                # Map to [0,1] using Normal CDF: Phi(z) = 0.5*(1+erf(z/sqrt(2)))
                df[f"norm_{d}"] = 0.5 * (1.0 + z.map(lambda t: math.erf(float(t) / math.sqrt(2.0))))
            else:
                df[f"norm_{d}"] = 0.5
    elif normalization == "minmax":
        for d in dims:
            if d in df.columns:
                mn = float(df[d].min())
                mx = float(df[d].max())
                den = (mx - mn) if (mx > mn) else 1.0
                df[f"norm_{d}"] = (df[d] - mn) / den
            else:
                df[f"norm_{d}"] = 0.5

    elif normalization == "rank_within_group":
        # Extract major group
        df["major"] = df["soc_code"].str.slice(0, 2)
        for d in dims:
            if d in df.columns:
                df[f"norm_{d}"] = df.groupby("major")[d].rank(pct=True)
            else:
                df[f"norm_{d}"] = 0.5
    
    # 2. Compute NetRisk
    # Substitution = (Writing + ToolTech)/2
    # Defense = (Physical + Social + Creativity)/3
    sub_cols = [f"norm_{d}" for d in ["writing_intensity", "tool_technology"]]
    def_cols = [f"norm_{d}" for d in ["physical_manual", "social_perceptiveness", "creativity_originality"]]
    
    df["substitution"] = df[sub_cols].mean(axis=1)
    df["defense"] = df[def_cols].mean(axis=1)
    df["net_risk"] = df["substitution"] - df["defense"]
    
    return df

def build_mechanism_sensitivity():
    print("Loading O*NET files...")
    activities = load_onet_file("Work Activities.txt")
    abilities = load_onet_file("Abilities.txt")
    skills = load_onet_file("Skills.txt")
    
    full = pd.concat([activities, abilities, skills], ignore_index=True)
    if full.empty:
        print("No O*NET data loaded.")
        return

    full["soc_code"] = full["O*NET-SOC Code"].astype(str).str.slice(0, 7)
    
    # Prepare base element mapping
    unique_elements = full[["Element ID", "Element Name"]].drop_duplicates()
    
    def get_raw_scores(descriptors_map):
        # Build element -> dim map
        elem_to_dim = {}
        for dim, keywords in descriptors_map.items():
            for k in keywords:
                # Find matching elements
                matches = unique_elements[unique_elements["Element Name"].str.contains(k, case=False, regex=False)]
                for _, row in matches.iterrows():
                    elem_to_dim[row["Element Name"]] = dim
        
        relevant_df = full[full["Element Name"].isin(elem_to_dim.keys())].copy()
        relevant_df["dimension"] = relevant_df["Element Name"].map(elem_to_dim)
        
        scores = relevant_df.groupby(["soc_code", "dimension"])["Data Value"].mean().reset_index()
        pivot = scores.pivot(index="soc_code", columns="dimension", values="Data Value").reset_index()
        return pivot

    results = []

    # 1. Baseline
    print("Running Baseline...")
    raw_base = get_raw_scores(DIMENSION_DESCRIPTORS)
    scored_base = compute_net_risk(raw_base, "percentile")
    
    for soc, label in FOCAL_CAREERS.items():
        row = scored_base[scored_base["soc_code"] == soc]
        if not row.empty:
            val = row.iloc[0]["net_risk"]
            rank = scored_base["net_risk"].rank(ascending=False)[scored_base["soc_code"] == soc].iloc[0]
            results.append({
                "variant": "Baseline",
                "career": label,
                "net_risk": val,
                "rank": rank,
                "sign_stable": True # defined as matching baseline sign
            })

    # 2. Leave-one-out
    print("Running Leave-one-out variants...")
    for dim, keywords in DIMENSION_DESCRIPTORS.items():
        for i, k in enumerate(keywords):
            variant_name = f"Drop {k[:20]}..."
            
            # Create perturbed map
            new_map = DIMENSION_DESCRIPTORS.copy()
            new_list = [x for x in keywords if x != k]
            if not new_list: continue # Don't drop the only descriptor
            new_map[dim] = new_list
            
            raw = get_raw_scores(new_map)
            scored = compute_net_risk(raw, "percentile")
            
            for soc, label in FOCAL_CAREERS.items():
                row = scored[scored["soc_code"] == soc]
                if not row.empty:
                    val = row.iloc[0]["net_risk"]
                    rank = scored["net_risk"].rank(ascending=False)[scored["soc_code"] == soc].iloc[0]
                    
                    # Check sign stability against baseline
                    base_val = next(r["net_risk"] for r in results if r["variant"] == "Baseline" and r["career"] == label)
                    stable = (val * base_val) > 0 or (abs(val) < 0.05 and abs(base_val) < 0.05)
                    
                    results.append({
                        "variant": variant_name,
                        "career": label,
                        "net_risk": val,
                        "rank": rank,
                        "sign_stable": stable
                    })

    # 3. Normalization variants
    print("Running Normalization variants...")
    for norm_method in ["zscore", "minmax", "rank_within_group"]:
        # Use baseline descriptors
        scored = compute_net_risk(raw_base, norm_method)
        
        for soc, label in FOCAL_CAREERS.items():
            row = scored[scored["soc_code"] == soc]
            if not row.empty:
                val = row.iloc[0]["net_risk"]
                rank = scored["net_risk"].rank(ascending=False)[scored["soc_code"] == soc].iloc[0]
                
                base_val = next(r["net_risk"] for r in results if r["variant"] == "Baseline" and r["career"] == label)
                stable = (val * base_val) > 0 or (abs(val) < 0.05 and abs(base_val) < 0.05)
                
                results.append({
                    "variant": f"Norm: {norm_method}",
                    "career": label,
                    "net_risk": val,
                    "rank": rank,
                    "sign_stable": stable
                })

    # Save CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_DIR / "mechanism_sensitivity.csv", index=False)
    print("Wrote mechanism_sensitivity.csv")

    # Generate LaTeX Table
    # Pivot to show range of NetRisk and % Stable
    summary = df_res.groupby("career").agg(
        baseline_net_risk=("net_risk", lambda x: x[df_res["variant"]=="Baseline"].iloc[0]),
        min_net_risk=("net_risk", "min"),
        max_net_risk=("net_risk", "max"),
        stable_pct=("sign_stable", "mean")
    ).reset_index()
    
    # Reorder
    order = ["Software Dev", "Electrician", "Writer"]
    summary["__ord"] = summary["career"].apply(lambda x: order.index(x) if x in order else 99)
    summary = summary.sort_values("__ord").drop(columns="__ord")
    
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Sensitivity of the \emph{uncalibrated} (mechanism) NetRisk index to descriptor perturbations (leave-one-out) and normalization choices at the SOC-occupation level. Normalization variants include percentiles, z-score mapped via Normal CDF, min--max scaling, and within-major-group percentile ranks.}"
    )
    lines.append(r"\label{tab:mechanism_sensitivity}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Career & Baseline $\NetRisk$ & Range [Min, Max] & Sign Stability (\%) \\")
    lines.append(r"\midrule")
    
    for _, r in summary.iterrows():
        # Format
        base = f"{r['baseline_net_risk']:.3f}"
        rng = f"[{r['min_net_risk']:.3f}, {r['max_net_risk']:.3f}]"
        stable = f"{r['stable_pct']*100:.0f}\%"
        
        lines.append(f"{r['career']} & {base} & {rng} & {stable} \\\\")
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    (TABLES_DIR / "mechanism_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/tables/mechanism_sensitivity.tex")

if __name__ == "__main__":
    build_mechanism_sensitivity()
