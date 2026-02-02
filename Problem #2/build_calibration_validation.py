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

def _projected_gd_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    lr: float = 0.05,
    iters: int = 4000,
    w_floor: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Ridge-regularized nonnegative fit via projected gradient descent.
    Objective: (1/n)||Xw+b-y||^2 + lam||w||^2, s.t. w>=0. Intercept unconstrained.
    """
    n, k = X.shape
    w = np.zeros(k)
    b = float(np.mean(y))
    for i in range(iters):
        y_pred = X @ w + b
        err = y_pred - y
        grad_w = (2.0 / n) * (X.T @ err) + 2.0 * lam * w
        grad_b = float((2.0 / n) * np.sum(err))
        w -= lr * grad_w
        b -= lr * grad_b
        w = np.maximum(w, 0.0)
        if w_floor is not None:
            w = np.maximum(w, w_floor)
        if i % 1000 == 0 and i > 0:
            lr = lr * 0.7
    return w, b


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    xs = pd.Series(x).rank(pct=False, method="average")
    ys = pd.Series(y).rank(pct=False, method="average")
    return float(xs.corr(ys, method="pearson"))


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

    # 4. Interpretability add-on: collinearity + bootstrap intervals + ridge alternative
    try:
        # (A) Feature correlation (unsigned mechanism percentiles)
        corr = df[feat_names].corr(method="pearson")

        # (B) Bootstrap weight summary (raw-scale weights; treat ~0 as 0)
        q05 = boot_df.quantile(0.05)
        q50 = boot_df.quantile(0.50)
        q95 = boot_df.quantile(0.95)
        p_zero = (boot_df.abs() <= 1e-6).mean()

        # (C) Alternative calibration that preserves interpretability:
        # enforce small minimum weights on Writing/Creativity to avoid "headline awkwardness"
        base_r2 = float(_metrics(y, X @ w_std + b_std)["r2"])
        lam_grid = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        # Floors specified in RAW scale, then converted to standardized scale.
        w_floor_raw = np.zeros(len(feat_names), dtype=float)
        w_floor_raw[feat_names.index("writing_intensity")] = 0.005
        w_floor_raw[feat_names.index("creativity_originality")] = 0.005
        w_floor_std = w_floor_raw * stds

        candidates = []
        for lam in lam_grid:
            w2_std, b2_std = _projected_gd_ridge(X, y, lam=lam, w_floor=w_floor_std)
            w2_raw = w2_std / stds
            b2_raw = float(b2_std - np.sum((w2_std * means) / stds))
            y2_pred = X @ w2_std + b2_std
            m2 = _metrics(y, y2_pred)
            candidates.append(
                {
                    "lam": float(lam),
                    "w_std": w2_std,
                    "b_std": float(b2_std),
                    "w_raw": w2_raw,
                    "b_raw": float(b2_raw),
                    "r2": float(m2["r2"]),
                    "mae": float(m2["mae"]),
                    "rmse": float(m2["rmse"]),
                }
            )

        # Choose the best-fit alternative subject to staying close to base R2
        ok = [c for c in candidates if c["r2"] >= base_r2 - 0.02]
        chosen = max(ok, key=lambda c: c["r2"]) if ok else max(candidates, key=lambda c: c["r2"])

        w_alt_raw = chosen["w_raw"]
        lam_chosen = float(chosen["lam"])

        # (D) NetRisk correlation + focal ordering check (across all mechanism occupations)
        mech_all = pd.read_csv(MECH_PATH)
        X_all = []
        for f, sgn in FEATURES:
            X_all.append(mech_all[f].astype(float).to_numpy() * sgn)
        X_all = np.column_stack(X_all)
        net_base = X_all @ w_raw
        net_alt = X_all @ w_alt_raw
        rho = _spearmanr(net_base, net_alt)
        r_pear = float(pd.Series(net_base).corr(pd.Series(net_alt), method="pearson"))

        # Focal career ranks (percentile ranks within mechanism set)
        occ_codes = mech_all["occ_code"].astype(str).str.strip()
        pr_base = pd.Series(net_base).rank(pct=True)
        pr_alt = pd.Series(net_alt).rank(pct=True)
        focal_rows = []
        for occ, label in FOCAL_CAREERS.items():
            msk = (occ_codes == occ)
            if not bool(msk.any()):
                continue
            idx = int(np.flatnonzero(msk.to_numpy())[0])
            focal_rows.append(
                (label, float(net_base[idx]), float(pr_base.iloc[idx]), float(net_alt[idx]), float(pr_alt.iloc[idx]))
            )

        # Build LaTeX table
        dim_label = {
            "writing_intensity": "Writing",
            "tool_technology": "Tool/Tech",
            "physical_manual": "Physical",
            "social_perceptiveness": "Social",
            "creativity_originality": "Creativity",
        }

        lines2 = []
        lines2.append(r"\begin{table}[H]")
        lines2.append(r"\centering")
        lines2.append(r"\caption{Why zero weights can occur (collinearity) and an interpretable alternative calibration. Panel A reports Pearson correlations among the five mechanism percentiles. Panel B reports the original nonnegative calibration weights, bootstrap stability (5th/50th/95th percentiles; $n=200$), the fraction of bootstrap fits with weight effectively zero, and an alternative nonnegative fit that enforces small minimum weights for Writing/Creativity (to prevent sparse-weight over-interpretation).}")
        lines2.append(r"\label{tab:calibration_interpretability}")
        lines2.append(r"\resizebox{\textwidth}{!}{%")

        # Panel A: correlation matrix
        lines2.append(r"\begin{tabular}{lccccc}")
        lines2.append(r"\toprule")
        lines2.append(r"\multicolumn{6}{l}{\textbf{Panel A: Feature correlations (Pearson $r$)}} \\")
        lines2.append(r"\midrule")
        header = [""] + [dim_label.get(c, c) for c in corr.columns.tolist()]
        lines2.append(" & ".join(header) + r" \\")
        for row_name in corr.index.tolist():
            row = [dim_label.get(row_name, row_name)] + [f"{float(corr.loc[row_name, c]):.2f}" for c in corr.columns.tolist()]
            lines2.append(" & ".join(row) + r" \\")
        lines2.append(r"\midrule")
        lines2.append(r"\multicolumn{6}{l}{\textbf{Panel B: Weight robustness (raw-scale weights)}} \\")
        lines2.append(r"\midrule")
        lines2.append(r"Dimension & Calibrated & Alt (min-weight) & Boot P50 & Boot [P05,P95] & $P(w\approx 0)$ \\")
        lines2.append(r"\midrule")

        # Load point-estimate calibrated weights from disk if available (matches paper)
        weights_path = DATA_DIR / "calibration_weights.csv"
        w_point = None
        if weights_path.exists():
            ww = pd.read_csv(weights_path).set_index("feature")
            w_point = ww["weight_raw"]

        for f in feat_names:
            dim = dim_label.get(f, f)
            cal_w = float(w_point.loc[f]) if w_point is not None and f in w_point.index else float("nan")
            alt_w = float(w_alt_raw[feat_names.index(f)])
            b50 = float(q50[f])
            b05 = float(q05[f])
            b95 = float(q95[f])
            pz = float(p_zero[f])
            lines2.append(
                f"{dim} & {cal_w:.3f} & {alt_w:.3f} & {b50:.3f} & [{b05:.3f},{b95:.3f}] & {pz:.2f} \\\\"
            )

        lines2.append(r"\midrule")
        # Compact “does it change?” line
        lines2.append(
            r"\multicolumn{6}{l}{\footnotesize Alternative choice: $\lambda="
            + f"{lam_chosen:.2f}"
            + r"$; minimum weights enforced: Writing $\ge 0.005$, Creativity $\ge 0.005$. NetRisk correlation (base vs alternative): Pearson $r="
            + f"{r_pear:.3f}"
            + r"$, Spearman $\rho="
            + f"{rho:.3f}"
            + r"$. Focal percentile ranks (base $\rightarrow$ alternative): "
            + "; ".join([f"{lbl} {pb:.2f}$\\to${pr:.2f}" for (lbl, _, pb, _, pr) in focal_rows])
            + r".} \\"
        )
        lines2.append(r"\bottomrule")
        lines2.append(r"\end{tabular}%")
        lines2.append(r"}")
        lines2.append(r"\end{table}")

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        (TABLES_DIR / "calibration_interpretability.tex").write_text("\n".join(lines2), encoding="utf-8")
        print("Wrote calibration_interpretability.tex")
    except Exception as e:
        print(f"Skipping calibration_interpretability artifacts due to error: {e}")

if __name__ == "__main__":
    main()
