"""
Build expanded mechanism layer using full O*NET database (db_30_1_text).
Computes 5 mechanism scores for ALL occupations, normalized by percentile.
"""
from pathlib import Path
import pandas as pd
import numpy as np

# If running from root, __file__ is ./build_mechanism_layer_expanded.py
DATA_DIR = Path(__file__).resolve().parent / "data"
ONET_DIR = DATA_DIR / "onet"
OUT_DIR = DATA_DIR

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

def load_onet_file(filename: str) -> pd.DataFrame:
    """Load O*NET text file (tab-separated), filter for Scale ID = IM (Importance)."""
    path = ONET_DIR / filename
    if not path.exists():
        print(f"Warning: {path} not found.")
        return pd.DataFrame()
    
    # O*NET text files are tab-separated
    df = pd.read_csv(path, sep="\t", on_bad_lines='skip')
    
    # Filter for Importance (IM)
    if "Scale ID" in df.columns:
        df = df[df["Scale ID"] == "IM"].copy()
    
    return df[["O*NET-SOC Code", "Element ID", "Element Name", "Data Value"]]

def build_expanded_mechanism():
    print(f"Loading O*NET files from {ONET_DIR}...")
    # Load the three main descriptor files
    activities = load_onet_file("Work Activities.txt")
    activities["domain_file"] = "Work Activities"
    
    abilities = load_onet_file("Abilities.txt")
    abilities["domain_file"] = "Abilities"
    
    skills = load_onet_file("Skills.txt")
    skills["domain_file"] = "Skills"
    
    # Combine
    full = pd.concat([activities, abilities, skills], ignore_index=True)
    
    if full.empty:
        print("No O*NET data loaded. Check data/onet folder.")
        return

    # Map O*NET-SOC (11-1011.00) to SOC 2018 (11-1011)
    full["soc_code"] = full["O*NET-SOC Code"].astype(str).str.slice(0, 7)
    
    # Pre-compute descriptor to dimension mapping
    # We want a manifest of (Element ID, Element Name, Dimension, Match Rule)
    unique_elements = full[["Element ID", "Element Name", "domain_file"]].drop_duplicates()
    element_to_dims = {elem: [] for elem in unique_elements["Element Name"].unique()}
    
    manifest_rows = []
    
    for _, row in unique_elements.iterrows():
        elem_name = row["Element Name"]
        elem_id = row["Element ID"]
        domain = row["domain_file"]
        
        elem_lower = str(elem_name).lower()
        for dim, keywords in DIMENSION_DESCRIPTORS.items():
            for k in keywords:
                k_lower = k.lower()
                if k_lower in elem_lower or elem_lower in k_lower:
                    element_to_dims[elem_name].append(dim)
                    manifest_rows.append({
                        "dimension": dim,
                        "domain_file": domain,
                        "element_id": elem_id,
                        "element_name": elem_name,
                        "match_rule": k
                    })
                    break # Matched this dimension, move to next dimension

    # Save manifest
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "mechanism_element_map.csv", index=False)
    print(f"Wrote mechanism_element_map.csv: {manifest.shape}")
    
    # Filter only relevant elements
    relevant_elements = [e for e, dims in element_to_dims.items() if dims]
    relevant_df = full[full["Element Name"].isin(relevant_elements)].copy()
    
    # Add a column for dimensions (list)
    relevant_df["dimensions"] = relevant_df["Element Name"].map(element_to_dims)
    # Explode
    exploded = relevant_df.explode("dimensions")
    
    # Group by SOC, dimensions -> Mean Data Value
    scores = exploded.groupby(["soc_code", "dimensions"])["Data Value"].mean().reset_index()
    
    # Pivot: SOC x Dimensions
    pivot = scores.pivot(index="soc_code", columns="dimensions", values="Data Value").reset_index()
    
    # Normalize: Percentiles (0-1)
    print("Computing percentiles...")
    dims = list(DIMENSION_DESCRIPTORS.keys())
    for d in dims:
        if d in pivot.columns:
            # Rank pct=True gives 0-1
            pivot[f"norm_{d}"] = pivot[d].rank(pct=True)
        else:
            pivot[f"norm_{d}"] = np.nan
            pivot[d] = np.nan # Ensure raw col exists

    # Save
    pivot.to_csv(OUT_DIR / "mechanism_layer_expanded.csv", index=False)
    print(f"Wrote mechanism_layer_expanded.csv: {pivot.shape}")

    # Create the merge-ready version (occ_code, norm_*)
    merge_cols = ["soc_code"] + [f"norm_{d}" for d in dims]
    merge_ready = pivot[merge_cols].rename(columns={"soc_code": "occ_code"})
    # Rename norm_X -> X
    merge_ready = merge_ready.rename(columns={f"norm_{d}": d for d in dims})
    
    merge_ready.to_csv(OUT_DIR / "mechanism_layer_all.csv", index=False)
    print(f"Wrote mechanism_layer_all.csv: {merge_ready.shape}")

if __name__ == "__main__":
    build_expanded_mechanism()
