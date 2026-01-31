"""
Build expanded mechanism layer using full O*NET database (db_30_1_text).
Computes 5 mechanism scores for ALL occupations, normalized by percentile.
"""
from pathlib import Path
import pandas as pd
import numpy as np

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
    df = pd.read_csv(path, sep="\t")
    
    # Filter for Importance (IM)
    if "Scale ID" in df.columns:
        df = df[df["Scale ID"] == "IM"].copy()
    
    return df[["O*NET-SOC Code", "Element Name", "Data Value"]]

def build_expanded_mechanism():
    print("Loading O*NET files...")
    # Load the three main descriptor files
    activities = load_onet_file("Work Activities.txt")
    abilities = load_onet_file("Abilities.txt")
    skills = load_onet_file("Skills.txt")
    
    # Combine
    full = pd.concat([activities, abilities, skills], ignore_index=True)
    
    if full.empty:
        print("No O*NET data loaded. Check data/onet folder.")
        return

    # Map O*NET-SOC (11-1011.00) to SOC 2018 (11-1011)
    full["soc_code"] = full["O*NET-SOC Code"].astype(str).str.slice(0, 7)
    
    # Pre-compute descriptor to dimension mapping
    # descriptor_name -> [list of dimensions it belongs to]
    desc_map = {}
    for dim, keywords in DIMENSION_DESCRIPTORS.items():
        for k in keywords:
            # We match if keyword is substring of Element Name (case insensitive)
            # OR Element Name is substring of keyword (as per original script logic)
            # But here we have the full dataframe, so we can iterate unique elements
            pass

    unique_elements = full["Element Name"].unique()
    element_to_dims = {elem: [] for elem in unique_elements}
    
    for elem in unique_elements:
        elem_lower = elem.lower()
        for dim, keywords in DIMENSION_DESCRIPTORS.items():
            for k in keywords:
                k_lower = k.lower()
                if k_lower in elem_lower or elem_lower in k_lower:
                    element_to_dims[elem].append(dim)
                    break # Matched this dimension, move to next dimension
    
    # Add dimension columns
    # We'll pivot or group by SOC and Dimension
    
    # Filter only relevant elements
    relevant_elements = [e for e, dims in element_to_dims.items() if dims]
    relevant_df = full[full["Element Name"].isin(relevant_elements)].copy()
    
    # Explode: one row per (SOC, Element) might map to multiple Dimensions? 
    # Logic: "Return mean importance for descriptors that match"
    # So for a SOC, we group by Dimension.
    
    # Create a long format: SOC, Dimension, Score
    records = []
    # Optimization: iterate rows is slow. Use mapping.
    
    # Add a column for dimensions (list)
    relevant_df["dimensions"] = relevant_df["Element Name"].map(element_to_dims)
    # Explode
    exploded = relevant_df.explode("dimensions")
    
    # Group by SOC, dimensions -> Mean Data Value
    scores = exploded.groupby(["soc_code", "dimensions"])["Data Value"].mean().reset_index()
    
    # Pivot: SOC x Dimensions
    pivot = scores.pivot(index="soc_code", columns="dimensions", values="Data Value").reset_index()
    
    # Add occ_title from occ_key if available, or just SOC
    # We can join with occ_key later.
    
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
