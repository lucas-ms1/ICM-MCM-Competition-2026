# Mechanism layer (O*NET task/skill/tool descriptors)

Normalized scores per occupation for **why GenAI substitutes vs complements**—defensible “mechanism” variables for modeling and narrative.

## Source

- **O*NET-SOC** descriptors (Work Activities, Abilities, Skills), importance 0–100.
- **Join:** SOC (BLS 7-digit) ↔ O*NET-SOC via folder mapping. Crosswalk: [O*NET-SOC 2019 to 2018 SOC](https://www.onetcenter.org/taxonomy/2019/soc/2019_to_SOC_Crosswalk.xlsx) for full occupation set; this project uses per-career O*NET CSVs in `data/Software Developer`, `data/Electrician`, `data/Creative Writing`.

## Dimensions (3–6 normalized scores)

| Score (0–1) | Meaning | Example O*NET descriptors |
|-------------|--------|----------------------------|
| **writing_intensity** | Writing/documentation intensity | Documenting/Recording Information, Written Expression, Writing, Performing Administrative Activities |
| **social_perceptiveness** | Customer interaction / social perceptiveness | Establishing and Maintaining Interpersonal Relationships, Assisting and Caring for Others, Social Perceptiveness, Communicating with People Outside the Organization |
| **physical_manual** | Physical/manual demands | Handling and Moving Objects, Performing General Physical Activities, Manual Dexterity, Stamina, Repairing and Maintaining Equipment |
| **creativity_originality** | Creativity/originality | Thinking Creatively, Originality, Fluency of Ideas |
| **tool_technology** | Tool/technology reliance | Working with Computers, Programming, Technology Design, Interacting With Computers |

Each dimension starts as the mean of matching descriptor importances (0–100). The normalization depends on which build you run:

- **Expanded (report) build**: `python build_mechanism_layer_expanded.py` reads the O*NET text database in `data/onet/` and converts each dimension to a **percentile rank (0–1)** across the scored occupation set (so 1 = highest percentile among occupations in this build).
- **Small demo build**: `python build_mechanism_layer.py` reads per-career CSV folders and uses **min–max normalization** across only the included occupations (useful for quick inspection, not for the full 774-occupation index).

## Outputs

- **mechanism_scores.csv** / **.xlsx** — occ_code, occ_title, raw_* (0–100), norm_* (0–1) for each dimension.
- **mechanism_layer.csv** / **.xlsx** — Merge-ready: occ_code, occ_title, writing_intensity, social_perceptiveness, physical_manual, creativity_originality, tool_technology (all 0–1).
 - **mechanism_layer_all.csv** — Expanded merge-ready mechanism layer for the full scored set (occ_code + the 5 normalized dimension percentiles).
 - **mechanism_layer_expanded.csv** — Expanded raw + normalized table (raw dimension means and percentile-normalized columns).

## Merge

Join to careers or occ_key on **occ_code**:

```python
careers = pd.read_csv("data/careers/software_engineer.csv")
mechanism = pd.read_csv("data/mechanism_layer.csv")
careers_with_mechanism = careers.merge(mechanism[["occ_code", "writing_intensity", "social_perceptiveness", "physical_manual", "creativity_originality", "tool_technology"]], on="occ_code", how="left")
```

## Build

From project root (choose one):

- `python build_mechanism_layer_expanded.py` (recommended for the report; uses `data/onet/*.txt` and produces the 774-occupation mechanism layer used in scenarios).
- `python build_mechanism_layer.py` (small per-career build; uses `data/Software Developer`, `data/Electrician`, `data/Creative Writing`).

Expanded build requires O*NET text files in `data/onet/` (e.g., `Work Activities.txt`, `Abilities.txt`, `Skills.txt`). Small build requires O*NET descriptor CSVs in the three per-career folders (work_activities_*.csv, abilities_*.csv, skills_*.csv).
