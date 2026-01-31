"""
Monte Carlo uncertainty analysis for scenario employment.
Perturbs:
  - scenario strength s
  - NetRisk (multiplicative noise)
  - complementarity factor
Outputs:
  - data/uncertainty_summary.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SCENARIOS_DEFAULT = {
    "Moderate_Substitution": 0.015,
    "High_Disruption": 0.03,
}


def get_g_adj(g_base: float, risk: float, shock_factor: float, comp_factor: float) -> float:
    if risk >= 0:
        return g_base - (shock_factor * risk)
    return g_base + (comp_factor * shock_factor * (-risk))


def main() -> None:
    summ_path = DATA_DIR / "scenario_summary.csv"
    if not summ_path.exists():
        raise FileNotFoundError("Missing data/scenario_summary.csv")
    summ = pd.read_csv(summ_path)

    # Scenario strengths (use calibrated if available)
    scenarios = SCENARIOS_DEFAULT.copy()
    s_path = DATA_DIR / "calibration_recommended_s.csv"
    if s_path.exists():
        s_df = pd.read_csv(s_path)
        s_map = dict(zip(s_df["scenario"], s_df["s_value"]))
        for k in ["Moderate_Substitution", "High_Disruption"]:
            if k in s_map:
                scenarios[k] = float(s_map[k])

    rng = np.random.default_rng(42)
    n_draws = 2000

    rows = []
    for _, r in summ.iterrows():
        career = r["career"]
        emp_2024 = float(r["emp_2024"])
        g_base = float(r["g_baseline"])
        risk_base = float(r["net_risk"])

        # Risk noise: multiplicative with a small absolute floor
        risk_sigma = 0.15 * max(abs(risk_base), 0.05)

        for scen, s_mean in scenarios.items():
            s_sigma = 0.2 * s_mean
            s_draws = rng.normal(s_mean, s_sigma, n_draws)
            s_draws = np.clip(s_draws, 0.0, None)

            comp_draws = rng.normal(0.2, 0.05, n_draws)
            comp_draws = np.clip(comp_draws, 0.0, 0.5)

            risk_draws = rng.normal(risk_base, risk_sigma, n_draws)

            g_adj = np.array(
                [
                    get_g_adj(g_base, risk_draws[i], s_draws[i], comp_draws[i])
                    for i in range(n_draws)
                ]
            )
            emp_2034 = emp_2024 * ((1.0 + g_adj) ** 10)

            rows.append(
                {
                    "career": career,
                    "scenario": scen,
                    "emp_p05": np.percentile(emp_2034, 5),
                    "emp_p50": np.percentile(emp_2034, 50),
                    "emp_p95": np.percentile(emp_2034, 95),
                    "g_adj_mean": np.mean(g_adj),
                    "g_adj_p05": np.percentile(g_adj, 5),
                    "g_adj_p95": np.percentile(g_adj, 95),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "uncertainty_summary.csv", index=False)
    print("Wrote uncertainty_summary.csv")


if __name__ == "__main__":
    main()
