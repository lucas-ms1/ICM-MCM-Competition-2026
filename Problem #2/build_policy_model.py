"""
Lightweight multi-objective policy decision model.
Outputs:
  - data/policy_decision_scores.csv
  - data/policy_decision_summary.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def main() -> None:
    summ_path = DATA_DIR / "scenario_summary.csv"
    if not summ_path.exists():
        raise FileNotFoundError("Missing data/scenario_summary.csv")

    summ = pd.read_csv(summ_path)
    summ = summ.set_index("career")

    institutions = [
        {"institution": "SDSU", "career": "software_engineer"},
        {"institution": "LATTC", "career": "electrician"},
        {"institution": "Academy of Art", "career": "writer"},
    ]

    # Exposure proxy from NetRisk (positive tail only)
    net_risk = summ["net_risk"].astype(float)
    exposure = net_risk.clip(lower=0.0)
    exp_max = exposure.max() if exposure.max() > 0 else 1.0
    exposure_norm = exposure / exp_max

    # Baseline growth proxy (normalized)
    g_base = summ["g_baseline"].astype(float)
    g_min, g_max = g_base.min(), g_base.max()
    g_norm = (g_base - g_min) / (g_max - g_min) if g_max > g_min else g_base * 0 + 0.5

    regimes = {
        "Ban": {"employ_boost": 0.15, "integrity_risk": 0.10, "sustain_cost": 0.05},
        "Allow_with_Audit": {"employ_boost": 0.55, "integrity_risk": 0.35, "sustain_cost": 0.35},
        "Require": {"employ_boost": 0.85, "integrity_risk": 0.60, "sustain_cost": 0.70},
    }

    weight_regimes = {
        "Balanced": (1.0, 1.0, 1.0),
        "Integrity_First": (1.0, 2.0, 1.0),
        "Sustainability_First": (1.0, 1.0, 2.0),
    }

    rows = []
    for inst in institutions:
        career = inst["career"]
        exp = float(exposure_norm.loc[career])
        base_emp = 0.5 + 0.5 * float(g_norm.loc[career])

        for w_name, (wE, wI, wS) in weight_regimes.items():
            for r_name, r in regimes.items():
                employability = base_emp + r["employ_boost"] * exp
                integrity_risk = r["integrity_risk"] * (0.5 + exp)
                sustainability_cost = r["sustain_cost"] * (0.5 + exp)
                score = (wE * employability) - (wI * integrity_risk) - (wS * sustainability_cost)

                rows.append(
                    {
                        "institution": inst["institution"],
                        "career": career,
                        "weight_regime": w_name,
                        "policy_regime": r_name,
                        "score": score,
                        "employability": employability,
                        "integrity_risk": integrity_risk,
                        "sustainability_cost": sustainability_cost,
                        "exposure_norm": exp,
                    }
                )

    scores = pd.DataFrame(rows)
    scores.to_csv(DATA_DIR / "policy_decision_scores.csv", index=False)

    # Summary: pick max score per institution + weight regime
    idx = scores.groupby(["institution", "weight_regime"])["score"].idxmax()
    summary = scores.loc[idx, ["institution", "weight_regime", "policy_regime", "score"]].copy()
    summary = summary.sort_values(["institution", "weight_regime"])
    summary.to_csv(DATA_DIR / "policy_decision_summary.csv", index=False)

    print("Wrote policy decision outputs.")


if __name__ == "__main__":
    main()
