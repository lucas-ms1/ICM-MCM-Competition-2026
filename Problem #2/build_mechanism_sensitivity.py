"""
Build sensitivity analysis for the O*NET mechanism layer.
Perturbs descriptors (leave-one-out) and normalization methods.
Outputs:
  - data/mechanism_sensitivity.csv
  - reports/tables/mechanism_sensitivity.tex
  - data/mechanism_accountability_sensitivity.csv
  - reports/tables/mechanism_accountability_sensitivity.tex
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

ACCOUNTABILITY_DESCRIPTORS = {
    # Sensitivity-only channel to address construct validity concerns:
    # capture system responsibility / accountability using Work Context (CX scale).
    "accountability_responsibility": [
        "Consequence of Error",
        "Impact of Decisions on Co-workers or Company Results",
    ]
}

FOCAL_CAREERS = {
    "15-1252": "Software Dev",
    "47-2111": "Electrician",
    "27-3043": "Writer",
}

def load_onet_file(filename: str, scale_id: str | None = "IM") -> pd.DataFrame:
    path = ONET_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", on_bad_lines='skip')
    if "Scale ID" in df.columns:
        if scale_id is not None:
            df = df[df["Scale ID"] == str(scale_id)].copy()
    return df[["O*NET-SOC Code", "Element ID", "Element Name", "Data Value"]]

def _soc_major_group(soc_code: str) -> str:
    try:
        return str(soc_code).strip()[:2]
    except Exception:
        return ""

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

def compute_net_risk_with_accountability(mech_df: pd.DataFrame, normalization: str = "percentile") -> pd.DataFrame:
    """
    Compute an alternate NetRisk that includes an accountability channel in defense:
      Defense6 = mean(Physical, Social, Creativity, Accountability)

    This is used ONLY as a construct-validity sensitivity check.
    """
    df = mech_df.copy()
    dims6 = list(DIMENSION_DESCRIPTORS.keys()) + list(ACCOUNTABILITY_DESCRIPTORS.keys())

    # Normalize (same options as compute_net_risk)
    if normalization == "percentile":
        for d in dims6:
            if d in df.columns:
                df[f"norm_{d}"] = df[d].rank(pct=True)
            else:
                df[f"norm_{d}"] = 0.5
    elif normalization == "zscore":
        for d in dims6:
            if d in df.columns:
                mu = df[d].mean()
                sig = df[d].std()
                if sig == 0 or (isinstance(sig, float) and math.isnan(sig)):
                    sig = 1.0
                z = (df[d] - mu) / sig
                df[f"norm_{d}"] = 0.5 * (1.0 + z.map(lambda t: math.erf(float(t) / math.sqrt(2.0))))
            else:
                df[f"norm_{d}"] = 0.5
    elif normalization == "minmax":
        for d in dims6:
            if d in df.columns:
                mn = float(df[d].min())
                mx = float(df[d].max())
                den = (mx - mn) if (mx > mn) else 1.0
                df[f"norm_{d}"] = (df[d] - mn) / den
            else:
                df[f"norm_{d}"] = 0.5
    elif normalization == "rank_within_group":
        df["major"] = df["soc_code"].astype(str).str.slice(0, 2)
        for d in dims6:
            if d in df.columns:
                df[f"norm_{d}"] = df.groupby("major")[d].rank(pct=True)
            else:
                df[f"norm_{d}"] = 0.5

    # Compute Sub/Defense6/NetRisk6
    sub_cols = [f"norm_{d}" for d in ["writing_intensity", "tool_technology"]]
    def6_cols = [f"norm_{d}" for d in ["physical_manual", "social_perceptiveness", "creativity_originality", "accountability_responsibility"]]
    df["substitution6"] = df[sub_cols].mean(axis=1)
    df["defense6"] = df[def6_cols].mean(axis=1)
    df["net_risk6"] = df["substitution6"] - df["defense6"]
    return df

def build_mechanism_sensitivity():
    print("Loading O*NET files...")
    activities = load_onet_file("Work Activities.txt", scale_id="IM")
    abilities = load_onet_file("Abilities.txt", scale_id="IM")
    skills = load_onet_file("Skills.txt", scale_id="IM")
    # Work Context for accountability sensitivity (CX scale)
    context = load_onet_file("Work Context.txt", scale_id="CX")

    full = pd.concat([activities, abilities, skills], ignore_index=True)
    if full.empty:
        print("No O*NET data loaded.")
        return

    full["soc_code"] = full["O*NET-SOC Code"].astype(str).str.slice(0, 7)
    
    # Prepare base element mapping
    unique_elements = full[["Element ID", "Element Name"]].drop_duplicates()
    
    def get_raw_scores_from(full_df: pd.DataFrame, unique_elems: pd.DataFrame, descriptors_map: dict) -> pd.DataFrame:
        # Build element -> dim map
        elem_to_dim = {}
        for dim, keywords in descriptors_map.items():
            for k in keywords:
                # Find matching elements
                matches = unique_elems[unique_elems["Element Name"].str.contains(k, case=False, regex=False)]
                for _, row in matches.iterrows():
                    elem_to_dim[row["Element Name"]] = dim
        
        relevant_df = full_df[full_df["Element Name"].isin(elem_to_dim.keys())].copy()
        relevant_df["dimension"] = relevant_df["Element Name"].map(elem_to_dim)
        
        scores = relevant_df.groupby(["soc_code", "dimension"])["Data Value"].mean().reset_index()
        pivot = scores.pivot(index="soc_code", columns="dimension", values="Data Value").reset_index()
        return pivot

    results = []

    # 1. Baseline
    print("Running Baseline...")
    raw_base = get_raw_scores_from(full, unique_elements, DIMENSION_DESCRIPTORS)
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
            
            raw = get_raw_scores_from(full, unique_elements, new_map)
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
        stable = f"{r['stable_pct']*100:.0f}\\%"
        
        lines.append(f"{r['career']} & {base} & {rng} & {stable} \\\\")
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    (TABLES_DIR / "mechanism_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/tables/mechanism_sensitivity.tex")

    # ---------------------------------------------------------------------
    # Construct-validity sensitivity: add an accountability channel (Work Context)
    # ---------------------------------------------------------------------
    if context.empty:
        print("Work Context.txt missing; skipping accountability sensitivity.")
        return

    full2 = pd.concat([full, context], ignore_index=True)
    full2["soc_code"] = full2["O*NET-SOC Code"].astype(str).str.slice(0, 7)
    unique2 = full2[["Element ID", "Element Name"]].drop_duplicates()
    desc6 = {}
    desc6.update(DIMENSION_DESCRIPTORS)
    desc6.update(ACCOUNTABILITY_DESCRIPTORS)
    raw6 = get_raw_scores_from(full2, unique2, desc6)

    # Baseline 5D sign for comparison (percentile normalization)
    base5 = scored_base.set_index("soc_code")["net_risk"].to_dict()
    rows6 = []
    for norm_method in ["percentile", "zscore", "minmax", "rank_within_group"]:
        scored6 = compute_net_risk_with_accountability(raw6, norm_method)
        for soc, label in FOCAL_CAREERS.items():
            row = scored6[scored6["soc_code"] == soc]
            if row.empty:
                continue
            val6 = float(row.iloc[0]["net_risk6"])
            base_val = float(base5.get(soc, 0.0))
            stable = (val6 * base_val) > 0 or (abs(val6) < 0.05 and abs(base_val) < 0.05)
            rows6.append(
                {
                    "career": label,
                    "soc_code": soc,
                    "normalization": norm_method,
                    "net_risk6": val6,
                    "baseline_net_risk5": base_val,
                    "sign_stable_vs_5d": bool(stable),
                }
            )
    df6 = pd.DataFrame(rows6)
    if df6.empty:
        print("No accountability sensitivity rows computed.")
        return
    df6.to_csv(OUT_DIR / "mechanism_accountability_sensitivity.csv", index=False)

    # Summarize: range of NetRisk6 and sign stability across normalizations
    summ6 = (
        df6.groupby("career")
        .agg(
            baseline_net_risk5=("baseline_net_risk5", "first"),
            min_net_risk6=("net_risk6", "min"),
            max_net_risk6=("net_risk6", "max"),
            stable_pct=("sign_stable_vs_5d", "mean"),
        )
        .reset_index()
    )
    order = ["Software Dev", "Electrician", "Writer"]
    summ6["__ord"] = summ6["career"].apply(lambda x: order.index(x) if x in order else 99)
    summ6 = summ6.sort_values("__ord").drop(columns="__ord")

    lines2 = []
    lines2.append(r"\begin{table}[H]")
    lines2.append(r"\centering")
    lines2.append(
        r"\caption{Construct-validity sensitivity: alternate NetRisk that adds an accountability/system-responsibility channel from O*NET Work Context (CX scale) into the defense score. Reported values are at the SOC-occupation level for the three focal careers, across normalization variants.}"
    )
    lines2.append(r"\label{tab:mechanism_accountability_sensitivity}")
    lines2.append(r"\begin{tabular}{lrrrr}")
    lines2.append(r"\toprule")
    lines2.append(r"Career & Baseline $\NetRisk$ (5D) & NetRisk$_6$ Range [Min, Max] & Sign Stability (\%) \\")
    lines2.append(r"\midrule")
    for _, rr in summ6.iterrows():
        base = f"{float(rr['baseline_net_risk5']):.3f}"
        rng = f"[{float(rr['min_net_risk6']):.3f}, {float(rr['max_net_risk6']):.3f}]"
        stable = f"{float(rr['stable_pct']) * 100:.0f}\\%"
        lines2.append(f"{rr['career']} & {base} & {rng} & {stable} \\\\")
    lines2.append(r"\bottomrule")
    lines2.append(r"\end{tabular}")
    lines2.append(r"\end{table}")
    (TABLES_DIR / "mechanism_accountability_sensitivity.tex").write_text("\n".join(lines2), encoding="utf-8")
    print("Wrote reports/tables/mechanism_accountability_sensitivity.tex")

if __name__ == "__main__":
    build_mechanism_sensitivity()
