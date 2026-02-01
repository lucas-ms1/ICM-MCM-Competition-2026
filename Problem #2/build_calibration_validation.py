"""
Run extensive validation for the calibration model.
1. K-fold CV (R2, MAE, RMSE)
2. Bootstrap weight stability
3. Ablation studies (Calibrated vs Uncalibrated vs ToolTech-only)
Outputs:
  - data/calibration_cv_results.csv
  - data/calibration_weight_bootstrap.csv
  - data/calibration_ablation.csv
  - reports/tables/calibration_validation.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
from build_calibration import _projected_gd, _metrics, FEATURES

DATA_DIR = Path(__file__).resolve().parent / "data"
TABLES_DIR = Path(__file__).resolve().parent / "reports" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent / "reports" / "figures"
MECH_PATH = DATA_DIR / "mechanism_layer_all.csv"
EXPOSURE_PATH = DATA_DIR / "ai_applicability_scores.csv"

FOCAL_CAREERS = {
    "15-1252": "Software Dev",
    "47-2111": "Electrician",
    "27-3043": "Writer",
}

def load_data():
    if not MECH_PATH.exists() or not EXPOSURE_PATH.exists():
        return None, None
    mech = pd.read_csv(MECH_PATH)
    exposure = pd.read_csv(EXPOSURE_PATH)
    exposure = exposure.rename(columns={"SOC Code": "occ_code", "ai_applicability_score": "ai_applicability"})
    exposure["occ_code"] = exposure["occ_code"].astype(str).str.strip()
    mech["occ_code"] = mech["occ_code"].astype(str).str.strip()
    
    df = mech.merge(exposure[["occ_code", "ai_applicability"]], on="occ_code", how="inner")
    
    # Construct X matrix
    feat_names = [f for f, _ in FEATURES]
    X_raw = []
    for f, sgn in FEATURES:
        X_raw.append(df[f].astype(float).to_numpy() * sgn)
    X_raw = np.column_stack(X_raw)
    y = df["ai_applicability"].astype(float).to_numpy()
    
    return df, X_raw, y, feat_names

def run_cv(X_raw, y, k=5):
    # Standardize whole set to get means/stds for consistency, or inside fold?
    # Strict CV requires inside fold.
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    metrics_list = []
    
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train_raw = X_raw[train_idx]
        y_train = y[train_idx]
        X_test_raw = X_raw[test_idx]
        y_test = y[test_idx]
        
        # Standardize based on TRAIN stats
        means = np.nanmean(X_train_raw, axis=0)
        stds = np.nanstd(X_train_raw, axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        
        X_train = (X_train_raw - means) / stds
        X_test = (X_test_raw - means) / stds
        
        w_std, b_std = _projected_gd(X_train, y_train)
        
        y_pred = X_test @ w_std + b_std
        m = _metrics(y_test, y_pred)
        m["fold"] = i + 1
        metrics_list.append(m)
        
    return pd.DataFrame(metrics_list)

def run_bootstrap(X_raw, y, feat_names, n_boot=200):
    n = len(y)
    weights = []
    
    for i in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_b_raw = X_raw[idx]
        y_b = y[idx]
        
        means = np.nanmean(X_b_raw, axis=0)
        stds = np.nanstd(X_b_raw, axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        X_b = (X_b_raw - means) / stds
        
        w_std, b_std = _projected_gd(X_b, y_b)
        w_raw = w_std / stds
        weights.append(w_raw)
        
    return pd.DataFrame(weights, columns=feat_names)

def run_ablation(df, X_raw, y, w_calibrated, b_calibrated):
    # 1. Uncalibrated (Equal Weights)
    # NetRisk = (Writing + Tool)/2 - (Phys + Soc + Create)/3
    # = 0.5*W + 0.5*T - 0.33*P - 0.33*S - 0.33*C
    # In our X matrix, columns are W, T, -P, -S, -C
    # So weights are 0.5, 0.5, 0.33, 0.33, 0.33
    w_uncal = np.array([0.5, 0.5, 0.333, 0.333, 0.333])
    
    # 2. ToolTech Only
    # Just column 1 (tool_technology)
    w_tool = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    
    # Compute scores
    risk_cal = X_raw @ w_calibrated + b_calibrated # This approximates the AI score
    risk_uncal = X_raw @ w_uncal
    risk_tool = X_raw @ w_tool
    
    # Correlations with Y (Actual AI applicability)
    res = []
    res.append({"model": "Calibrated", "corr": np.corrcoef(risk_cal, y)[0,1]})
    res.append({"model": "Uncalibrated (Equal)", "corr": np.corrcoef(risk_uncal, y)[0,1]})
    res.append({"model": "ToolTech Only", "corr": np.corrcoef(risk_tool, y)[0,1]})
    
    # Focal career changes
    # We need to map df rows to careers
    focal_res = []
    for occ, label in FOCAL_CAREERS.items():
        row_idx = df.index[df["occ_code"] == occ].tolist()
        if row_idx:
            idx = row_idx[0]
            # Normalize each score to z-score for fair comparison?
            # Or just raw correlation? 
            # The prompt asks "How much outcome changes". 
            # Let's check Rank.
            rank_cal = (pd.Series(risk_cal).rank(pct=True)[idx])
            rank_uncal = (pd.Series(risk_uncal).rank(pct=True)[idx])
            
            focal_res.append({
                "career": label,
                "rank_calibrated": rank_cal,
                "rank_uncalibrated": rank_uncal,
                "delta_rank": rank_cal - rank_uncal
            })
            
    return pd.DataFrame(res), pd.DataFrame(focal_res)

def main():
    print("Loading data for validation...")
    df, X_raw, y, feat_names = load_data()
    if df is None:
        print("Data missing.")
        return
        
    # 1. CV
    print("Running 5-fold CV...")
    cv_df = run_cv(X_raw, y)
    cv_df.to_csv(DATA_DIR / "calibration_cv_results.csv", index=False)
    
    # 2. Bootstrap
    print("Running Bootstrap...")
    boot_df = run_bootstrap(X_raw, y, feat_names)
    boot_df.to_csv(DATA_DIR / "calibration_weight_bootstrap.csv", index=False)
    
    # 3. Ablation
    # Need full fit first
    means = np.nanmean(X_raw, axis=0)
    stds = np.nanstd(X_raw, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    X = (X_raw - means) / stds
    w_std, b_std = _projected_gd(X, y)
    w_raw = w_std / stds
    b_raw = float(b_std - np.sum((w_std * means) / stds))
    
    print("Running Ablation...")
    ablation_metrics, focal_changes = run_ablation(df, X_raw, y, w_raw, b_raw)
    ablation_metrics.to_csv(DATA_DIR / "calibration_ablation.csv", index=False)
    
    # Generate LaTeX Summary
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Calibration Validation: Out-of-sample performance (5-fold CV) and Baseline Comparisons.}")
    lines.append(r"\label{tab:calibration_validation}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{4}{l}{\textbf{Panel A: Out-of-Sample CV Metrics}} \\")
    lines.append(r"Metric & Mean & Std. Dev & Range \\")
    lines.append(r"\midrule")
    
    for m in ["r2", "mae", "rmse"]:
        vals = cv_df[m]
        lines.append(f"{m.upper()} & {vals.mean():.3f} & {vals.std():.3f} & [{vals.min():.3f}, {vals.max():.3f}] \\\\")
        
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textbf{Panel B: Correlation with AI Applicability}} \\")
    lines.append(r"Model & Correlation ($r$) & & \\")
    lines.append(r"\midrule")
    
    for _, r in ablation_metrics.iterrows():
        lines.append(f"{r['model']} & {r['corr']:.3f} & & \\\\")
        
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    (TABLES_DIR / "calibration_validation.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote validation artifacts.")
    
    # Optional: Boxplot of weights
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    boot_df.boxplot(rot=45)
    plt.title("Bootstrapped Mechanism Weights (n=200)")
    plt.ylabel("Weight Magnitude")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "calibration_weights_box.png", dpi=150)
    print("Wrote calibration_weights_box.png")

if __name__ == "__main__":
    main()
