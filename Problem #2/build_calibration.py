"""
Calibrate O*NET-based mechanism weights against an external AI applicability dataset.
Outputs:
  - data/calibration_results.csv (metrics)
  - data/calibration_weights.csv (weights + feature stats)
  - data/calibration_fit.csv (SOC-level actual vs predicted)
  - data/mechanism_risk_calibrated.csv (NetRisk per SOC using calibrated weights)
  - data/calibration_recommended_s.csv (recommended scenario s based on NetRisk distribution)
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MECH_PATH = DATA_DIR / "mechanism_layer_all.csv"
EXPOSURE_PATH = DATA_DIR / "ai_applicability_scores.csv"

FEATURES = [
    ("writing_intensity", 1.0),
    ("tool_technology", 1.0),
    ("physical_manual", -1.0),
    ("social_perceptiveness", -1.0),
    ("creativity_originality", -1.0),
]


def _projected_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.05, iters: int = 4000) -> tuple[np.ndarray, float]:
    """Nonnegative weights via projected gradient descent; unconstrained intercept."""
    n, k = X.shape
    w = np.zeros(k)
    b = float(np.mean(y))
    for i in range(iters):
        y_pred = X @ w + b
        err = y_pred - y
        grad_w = (2.0 / n) * (X.T @ err)
        grad_b = float((2.0 / n) * np.sum(err))
        w -= lr * grad_w
        b -= lr * grad_b
        w = np.maximum(w, 0.0)
        if i % 1000 == 0 and i > 0:
            lr = lr * 0.7
    return w, b


def _metrics(y: np.ndarray, y_pred: np.ndarray) -> dict:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def main() -> None:
    if not MECH_PATH.exists():
        raise FileNotFoundError(f"Missing mechanism layer: {MECH_PATH}")
    if not EXPOSURE_PATH.exists():
        raise FileNotFoundError(f"Missing external exposure file: {EXPOSURE_PATH}")

    mech = pd.read_csv(MECH_PATH)
    exposure = pd.read_csv(EXPOSURE_PATH)
    exposure = exposure.rename(columns={"SOC Code": "occ_code", "ai_applicability_score": "ai_applicability"})
    exposure["occ_code"] = exposure["occ_code"].astype(str).str.strip()

    mech["occ_code"] = mech["occ_code"].astype(str).str.strip()
    df = mech.merge(exposure[["occ_code", "ai_applicability"]], on="occ_code", how="inner")

    # Build feature matrix with sign convention (defense dims are negative)
    feat_names = [f for f, _ in FEATURES]
    X_raw = []
    for f, sgn in FEATURES:
        X_raw.append(df[f].astype(float).to_numpy() * sgn)
    X_raw = np.column_stack(X_raw)
    y = df["ai_applicability"].astype(float).to_numpy()

    # Standardize features for stable optimization
    means = np.nanmean(X_raw, axis=0)
    stds = np.nanstd(X_raw, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X = (X_raw - means) / stds

    # Fit nonnegative weights
    w_std, b_std = _projected_gd(X, y)

    # Convert back to raw scale
    w_raw = w_std / stds
    b_raw = float(b_std - np.sum((w_std * means) / stds))

    y_pred = X @ w_std + b_std
    metrics = _metrics(y, y_pred)

    # Save fit table for diagnostics
    fit = df[["occ_code", "ai_applicability"]].copy()
    fit["ai_applicability_pred"] = y_pred
    fit["residual"] = fit["ai_applicability"] - fit["ai_applicability_pred"]
    fit.to_csv(DATA_DIR / "calibration_fit.csv", index=False)

    # Save weights + stats
    weights = pd.DataFrame(
        {
            "feature": feat_names,
            "sign_convention": [sgn for _, sgn in FEATURES],
            "weight_std": w_std,
            "weight_raw": w_raw,
            "feature_mean": means,
            "feature_std": stds,
        }
    )
    weights.to_csv(DATA_DIR / "calibration_weights.csv", index=False)

    # Metrics summary
    results = pd.DataFrame(
        {
            "metric": [
                "n_samples",
                "r2",
                "mae",
                "rmse",
                "intercept_raw",
            ],
            "value": [
                int(len(df)),
                metrics["r2"],
                metrics["mae"],
                metrics["rmse"],
                b_raw,
            ],
        }
    )
    results.to_csv(DATA_DIR / "calibration_results.csv", index=False)

    # Build calibrated NetRisk for all occupations in mechanism layer
    mech_all = mech.copy()
    X_all = []
    for f, sgn in FEATURES:
        X_all.append(mech_all[f].astype(float).to_numpy() * sgn)
    X_all = np.column_stack(X_all)
    net_risk_raw = X_all @ w_raw
    net_risk_centered = net_risk_raw - np.nanmean(net_risk_raw)
    max_abs = float(np.nanmax(np.abs(net_risk_centered))) if np.isfinite(np.nanmax(np.abs(net_risk_centered))) else 1.0
    max_abs = max(max_abs, 1e-6)
    net_risk_cal = net_risk_centered / max_abs

    mech_all["net_risk_raw"] = net_risk_raw
    mech_all["net_risk_calibrated"] = net_risk_cal
    mech_all[["occ_code", "net_risk_raw", "net_risk_calibrated"]].to_csv(
        DATA_DIR / "mechanism_risk_calibrated.csv", index=False
    )

    # Recommend scenario s based on calibrated distribution (p90 of positive tail)
    pos = net_risk_cal[net_risk_cal > 0]
    p90_pos = float(np.quantile(pos, 0.9)) if len(pos) > 0 else 1.0
    p90_pos = max(p90_pos, 1e-6)
    s_moderate = 0.015 / p90_pos
    s_high = 0.03 / p90_pos

    s_df = pd.DataFrame(
        {
            "scenario": ["Moderate_Substitution", "High_Disruption"],
            "target_delta_g_at_p90": [0.015, 0.03],
            "p90_positive_netrisk": [p90_pos, p90_pos],
            "s_value": [s_moderate, s_high],
        }
    )
    s_df.to_csv(DATA_DIR / "calibration_recommended_s.csv", index=False)

    # Persist weights for downstream use
    weights_out = {
        "features": feat_names,
        "sign_convention": [sgn for _, sgn in FEATURES],
        "weight_raw": [float(w) for w in w_raw],
        "intercept_raw": float(b_raw),
        "p90_positive_netrisk": p90_pos,
        "s_moderate": float(s_moderate),
        "s_high": float(s_high),
    }
    (DATA_DIR / "calibration_weights.json").write_text(json.dumps(weights_out, indent=2), encoding="utf-8")

    print("Calibration complete.")


if __name__ == "__main__":
    main()
