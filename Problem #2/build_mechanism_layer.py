"""
Build the "mechanism layer" from O*NET descriptors: 3–6 normalized scores per occupation
(writing/documentation, social perceptiveness, physical/manual, creativity, tool/technology).
Used to explain why GenAI substitutes vs complements (defensible "why" variables).

Requires: O*NET descriptor CSVs per occupation (work_activities, abilities, skills).
Join: SOC (BLS) ↔ O*NET-SOC via folder mapping; crosswalk can be extended later.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA_DIR

# SOC (7-digit) → folder name containing O*NET CSVs (work_activities_*.csv, abilities_*.csv, skills_*.csv)
# O*NET-SOC 2019 uses 8-digit (e.g. 15-1252.00); our careers use 7-digit SOC.
SOC_TO_ONET_FOLDER = {
    "15-1252": "Software Developer",   # Software Developers
    "47-2111": "Electrician",           # Electricians
    "27-3043": "Creative Writing",     # Writers and Authors (O*NET 27-3043.05)
}

# Dimension → O*NET descriptor names (exact or substring match on Work Activity / Ability / Skill column)
# Sources: O*NET Work Activities, Abilities, Skills. Importance scale 0–100.
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

# Occupation titles for output (SOC → title)
SOC_TITLE = {
    "15-1252": "Software Developers",
    "47-2111": "Electricians",
    "27-3043": "Writers and Authors",
}


def _load_onet_descriptors(folder_path: Path) -> list[tuple[str, float]]:
    """Load all O*NET descriptor CSVs from a folder. Returns [(descriptor_name, importance), ...]."""
    pairs = []
    for pattern in ["work_activities_*.csv", "abilities_*.csv", "skills_*.csv"]:
        for f in folder_path.glob(pattern):
            try:
                df = pd.read_csv(f)
                if df.shape[1] < 2:
                    continue
                imp_col = df.columns[0]
                name_col = df.columns[1]
                df[imp_col] = pd.to_numeric(df[imp_col], errors="coerce")
                for _, row in df.iterrows():
                    name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
                    imp = row[imp_col]
                    if name and pd.notna(imp):
                        pairs.append((name, float(imp)))
            except Exception:
                continue
    return pairs


def _score_dimension(pairs: list[tuple[str, float]], descriptor_list: list[str]) -> float:
    """Return mean importance for descriptors that match any of descriptor_list (substring)."""
    values = []
    for name, imp in pairs:
        for d in descriptor_list:
            if d.lower() in name.lower() or name.lower() in d.lower():
                values.append(imp)
                break
    return float(np.mean(values)) if values else np.nan


def build_mechanism_scores(
    soc_to_folder: dict[str, str] | None = None,
    dimension_descriptors: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Build raw then normalized mechanism scores per occupation from O*NET CSVs.
    Returns DataFrame: occ_code, occ_title, raw_* and norm_* for each dimension.
    """
    soc_to_folder = soc_to_folder or SOC_TO_ONET_FOLDER
    dimension_descriptors = dimension_descriptors or DIMENSION_DESCRIPTORS
    dims = list(dimension_descriptors.keys())

    rows = []
    for soc, folder_name in soc_to_folder.items():
        folder_path = DATA_DIR / folder_name
        if not folder_path.is_dir():
            continue
        pairs = _load_onet_descriptors(folder_path)
        if not pairs:
            continue
        raw = {d: _score_dimension(pairs, dimension_descriptors[d]) for d in dims}
        row = {"occ_code": soc, "occ_title": SOC_TITLE.get(soc, folder_name)}
        for d in dims:
            row["raw_" + d] = raw[d]
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize each dimension to 0–1 (min-max across occupations)
    for d in dims:
        col = "raw_" + d
        if col not in df.columns:
            continue
        x = df[col].astype(float)
        lo, hi = x.min(), x.max()
        if hi > lo:
            df["norm_" + d] = (x - lo) / (hi - lo)
        else:
            df["norm_" + d] = 0.5 if (lo == lo) else np.nan

    return df


def mechanism_layer_for_merge(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to columns suitable for merging with occ_key/careers: occ_code + normalized scores only.
    Column names: writing_intensity, social_perceptiveness, physical_manual, creativity_originality, tool_technology (0–1).
    """
    if scores.empty:
        return pd.DataFrame()
    norm_cols = [c for c in scores.columns if c.startswith("norm_")]
    out = scores[["occ_code", "occ_title"] + norm_cols].copy()
    out = out.rename(columns={c: c.replace("norm_", "") for c in norm_cols})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = build_mechanism_scores()
    if scores.empty:
        print("No O*NET data found. Place work_activities_*.csv, abilities_*.csv, skills_*.csv in data/Software Developer, data/Electrician, data/Creative Writing.")
        return
    # Full table: raw + normalized
    scores.to_csv(OUT_DIR / "mechanism_scores.csv", index=False)
    scores.to_excel(OUT_DIR / "mechanism_scores.xlsx", index=False)
    print("Wrote mechanism_scores:", scores.shape)
    # Merge-ready: occ_code + norm_* only
    layer = mechanism_layer_for_merge(scores)
    layer.to_csv(OUT_DIR / "mechanism_layer.csv", index=False)
    layer.to_excel(OUT_DIR / "mechanism_layer.xlsx", index=False)
    print("Wrote mechanism_layer (merge-ready):", layer.shape)
    print("Done. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
