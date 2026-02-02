"""
Robust rule-based policy decision model.
Replaces complex coefficients with defensible logic based on NetRisk sign and institution constraints.

Outputs:
  - data/policy_decision_summary.csv
  - data/policy_sensitivity.csv (Robustness check)
  - data/policy_decision_scores.csv (policy option scores for plotting)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _compute_EISA(
    net_risk: float,
    audit_cap: float,
    sustain_cap: float,
    access_cap: float,
    policy: str,
) -> tuple[float, float, float, float]:
    """
    Transparent 0-1 scoring model:
      - E: employability gain (higher is better)
      - I: integrity risk (higher is worse)
      - S: sustainability cost (higher is worse)
      - A: access/equity score (higher is better)

    Inputs:
      net_risk in [-1,1] (pipeline index), audit_cap in [0,1], sustain_cap in [0,1], access_cap in [0,1].
    """
    x_pos = _clip01(max(float(net_risk), 0.0))      # exposed-to-substitution share proxy
    x_neg = _clip01(max(-float(net_risk), 0.0))     # sheltered/complementarity proxy
    a = _clip01(audit_cap)
    sc = _clip01(sustain_cap)
    ac = _clip01(access_cap)

    # Sustainability: baseline compute intensity by policy, then scaled by (1 - sustainability capacity).
    # (Require > Allow > Ban), but keep magnitudes moderate to avoid dominating E/I.
    compute_intensity = {"Ban": 0.05, "Allow_with_Audit": 0.25, "Require": 0.35}
    ci = float(compute_intensity.get(policy, 0.35))
    S = _clip01(ci * (1.0 - sc))

    # Integrity risk: increases with exposure; audit capacity reduces risk for permissive regimes.
    if policy == "Ban":
        I = 0.05
    elif policy == "Allow_with_Audit":
        I = _clip01(0.08 + 0.70 * x_pos * (1.0 - a))
    else:  # Require
        # Stricter than Allow because ubiquitous use increases leakage/overreliance risks when exposed.
        I = _clip01(0.10 + 0.90 * x_pos * (1.0 - 0.6 * a))

    # Employability gain: tool fluency helps when exposed; Require helps most when sheltered (complementarity).
    if policy == "Ban":
        # Ban protects integrity but reduces tool fluency; more appropriate only when exposure is high and audit is weak.
        E = _clip01(0.35 + 0.20 * x_pos - 0.15 * x_neg)
    elif policy == "Allow_with_Audit":
        E = _clip01(0.55 + 0.20 * x_neg + 0.20 * a + 0.10 * x_pos)
    else:  # Require
        E = _clip01(0.60 + 0.30 * x_neg + 0.10 * a - 0.10 * x_pos)

    # Access/Equity score: coarse, transparent proxies.
    #
    # Two channels:
    #  (i) Financial access: whether students can access required tools without undue cost burden.
    #      - Ban has low tool cost burden, Require has the highest unless the institution subsidizes access.
    #  (ii) Learning access: whether GenAI can be used as a scaffold (esp. for non-native speakers / lower-prep students),
    #       but only if the institution provides equitable access and guardrails (captured by ac).
    #
    # This is intentionally a simple proxy (0–1) to match ICM auditability constraints.
    if policy == "Ban":
        # Low direct cost; but bans reduce availability of tool-based scaffolding.
        financial_access = 0.85
        learning_access = _clip01(0.35 + 0.10 * (1.0 - x_pos))  # slightly better when exposure is low
    elif policy == "Allow_with_Audit":
        # Moderate cost; improves learning access when the institution can provide accounts/support.
        financial_access = _clip01(0.60 + 0.25 * ac)
        learning_access = _clip01(0.50 + 0.25 * ac + 0.10 * x_pos)
    else:  # Require
        # Highest cost unless subsidized; best learning access only when access capacity is high.
        financial_access = _clip01(0.40 + 0.50 * ac)
        learning_access = _clip01(0.55 + 0.35 * ac + 0.05 * x_neg - 0.05 * x_pos)

    A = _clip01(0.5 * financial_access + 0.5 * learning_access)

    return float(E), float(I), float(S), float(A)


def _score_policy(E: float, I: float, S: float, A: float, wE: float, wI: float, wS: float, wA: float) -> float:
    """
    Policy objective score (higher is better):
      Score = wE*E + wI*(1-I) + wS*(1-S) + wA*A

    Notes:
      - E and A are benefits (higher is better).
      - I and S are risks/costs (higher is worse), so we convert them to benefits via (1-I) and (1-S).
      - This keeps the ranking identical to the benefit–cost form wE*E - wI*I - wS*S + wA*A,
        but avoids judge confusion about negative totals.
    """
    E = float(E)
    I = float(I)
    S = float(S)
    A = float(A)
    return float(wE) * E + float(wI) * (1.0 - I) + float(wS) * (1.0 - S) + float(wA) * A

def main() -> None:
    summ_path = DATA_DIR / "scenario_summary.csv"
    if not summ_path.exists():
        raise FileNotFoundError("Missing data/scenario_summary.csv")

    summ = pd.read_csv(summ_path)
    summ = summ.set_index("career")

    # Institution parameters (0-1 proxies)
    # audit_capacity: ability to verify provenance (oral defenses, version control, proctoring, staff time)
    # sustainability_capacity: ability to absorb compute cost (infrastructure + budget + policy constraints)
    # access_capacity: ability to provide equitable access to tools (institution-provided accounts, lab hardware, accommodations)
    institutions = [
        {"institution": "SDSU", "career": "software_engineer", "audit_capacity": 0.85, "sustainability_capacity": 0.70, "access_capacity": 0.75},
        {"institution": "LATTC", "career": "electrician", "audit_capacity": 0.60, "sustainability_capacity": 0.55, "access_capacity": 0.65},
        {"institution": "Academy of Art", "career": "writer", "audit_capacity": 0.40, "sustainability_capacity": 0.45, "access_capacity": 0.55},
    ]

    # Weight regimes for Score = wE*E + wI*(1-I) + wS*(1-S) + wA*A
    weight_regimes = {
        "Balanced": {"wE": 1.0, "wI": 1.0, "wS": 1.0, "wA": 1.0},
        "Integrity_First": {"wE": 1.0, "wI": 2.0, "wS": 1.0, "wA": 1.0},
        "Sustainability_First": {"wE": 1.0, "wI": 1.0, "wS": 2.0, "wA": 1.0},
        "Equity_First": {"wE": 1.0, "wI": 1.0, "wS": 1.0, "wA": 2.0},
    }

    results = []
    score_rows = []
    
    for inst in institutions:
        career = inst["career"]
        risk = float(summ.loc[career, "net_risk"])
        audit = inst["audit_capacity"]
        sustain = inst["sustainability_capacity"]
        access = inst["access_capacity"]
        
        for regime, w in weight_regimes.items():
            policies = ["Ban", "Allow_with_Audit", "Require"]
            scored = []
            for p in policies:
                E, I, S, A = _compute_EISA(risk, audit, sustain, access, p)
                score = _score_policy(E, I, S, A, w["wE"], w["wI"], w["wS"], w["wA"])
                scored.append((p, score, E, I, S, A))
                score_rows.append(
                    {
                        "institution": inst["institution"],
                        "career": career,
                        "weight_regime": regime,
                        "policy_regime": p,
                        "score": float(score),
                        "E": float(E),
                        "I": float(I),
                        "S": float(S),
                        "A": float(A),
                        "audit_capacity": float(audit),
                        "sustainability_capacity": float(sustain),
                        "access_capacity": float(access),
                        "net_risk": float(risk),
                    }
                )
            policy = max(scored, key=lambda t: t[1])[0]
            results.append({
                "institution": inst["institution"],
                "career": career,
                "weight_regime": regime,
                "policy_regime": policy,
                "score": float(max(scored, key=lambda t: t[1])[1]),
            })

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(DATA_DIR / "policy_decision_summary.csv", index=False)
    
    pd.DataFrame(score_rows).to_csv(DATA_DIR / "policy_decision_scores.csv", index=False)

    # Sensitivity Analysis
    # Perturb Risk (+/- 0.1), Audit (+/- 0.1), Access (+/- 0.1)
    sens_rows = []
    for inst in institutions:
        base_risk = float(summ.loc[inst["career"], "net_risk"])
        base_audit = inst["audit_capacity"]
        base_sustain = inst["sustainability_capacity"]
        base_access = inst["access_capacity"]
        
        for regime, w in weight_regimes.items():
            # compute base policy via scoring
            base_scores = []
            for p in ["Ban", "Allow_with_Audit", "Require"]:
                E, I, S, A = _compute_EISA(base_risk, base_audit, base_sustain, base_access, p)
                base_scores.append((p, _score_policy(E, I, S, A, w["wE"], w["wI"], w["wS"], w["wA"])))
            base_policy = max(base_scores, key=lambda t: t[1])[0]
            
            match_count = 0
            total_count = 0
            seen_policies = set()
            
            for dr in [-0.1, 0.0, 0.1]:
                for da in [-0.1, 0.0, 0.1]:
                    for dacc in [-0.1, 0.0, 0.1]:
                        r = base_risk + dr
                        aud = max(0.0, min(1.0, base_audit + da))
                        acc = max(0.0, min(1.0, base_access + dacc))
                        # Hold sustainability capacity fixed in this robustness check.
                        scores = []
                        for pol in ["Ban", "Allow_with_Audit", "Require"]:
                            E, I, S, A = _compute_EISA(r, aud, base_sustain, acc, pol)
                            scores.append((pol, _score_policy(E, I, S, A, w["wE"], w["wI"], w["wS"], w["wA"])))
                        p = max(scores, key=lambda t: t[1])[0]
                        seen_policies.add(p)
                        if p == base_policy:
                            match_count += 1
                        total_count += 1
            
            robustness = f"Stable ({match_count}/{total_count})"
            if match_count < total_count:
                others = sorted(list(seen_policies - {base_policy}))
                robustness = f"Flips ({match_count}/{total_count}); seen: {', '.join(others)}"
                
            sens_rows.append({
                "institution": inst["institution"],
                "weight_regime": regime,
                "baseline_policy": base_policy,
                "robustness": robustness
            })
            
    pd.DataFrame(sens_rows).to_csv(DATA_DIR / "policy_sensitivity.csv", index=False)
    print("Wrote policy artifacts.")

if __name__ == "__main__":
    main()
