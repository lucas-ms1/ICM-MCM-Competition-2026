"""
Build LaTeX-ready report artifacts (tables + figures) for Problem F.

Outputs (written under ./reports/):
  - reports/tables/scenario_summary.tex
  - reports/tables/sensitivity_grid.tex
  - reports/tables/top_exposed_sheltered.tex
  - reports/tables/openings_summary.tex
  - reports/tables/mechanism_coverage.tex
  - reports/tables/netrisk_summary.tex
  - reports/tables/netrisk_interpretation.tex
  - reports/tables/weight_sensitivity.tex
  - reports/figures/scenario_bar.png
  - reports/figures/netrisk_hist.png

Design goal: every numeric claim in the PDF is reproducible from the CSV artifacts
in ./data (scenario_summary.csv, mechanism_risk_scored.csv, oews/ep tables).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

# Complementarity uplift bound (m_max) used in scenario mapping.
M_MAX_DEFAULT = 0.20


def _latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _fmt_int(x: float | int) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{int(round(float(x))):,}"


def _fmt_int_latex(x: float | int) -> str:
    """LaTeX thousands separator: 2075300 -> 2{,}075{,}300 (for inline math)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    s = f"{int(round(float(x))):,}"
    return s.replace(",", "{,}")


def _fmt_float(x: float | int, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{float(x):.{nd}f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def get_g_adj(g_base: float, risk: float, shock_factor: float, m_comp: float = M_MAX_DEFAULT) -> float:
    """
    Compute adjusted growth using piecewise mapping.
    g_adj = g_base - s_sub * max(risk, 0) + s_comp * max(-risk, 0)
    s_sub = shock_factor
    s_comp = shock_factor * m_comp
    """
    s_sub = shock_factor
    s_comp = shock_factor * float(m_comp)
    if risk >= 0:
        return g_base - (s_sub * risk)
    else:
        return g_base + (s_comp * (-risk))

def build_scenario_summary_table() -> None:
    df = _load_required_csv(DATA_DIR / "scenario_summary.csv")
    df = df.copy()

    career_label = {
        "software_engineer": "Software Developers (STEM)",
        "electrician": "Electricians (Trade)",
        "writer": "Writers and Authors (Arts)",
    }
    df["career_label"] = df["career"].map(career_label).fillna(df["career"])

    # Stable order
    order = ["software_engineer", "electrician", "writer"]
    df["__ord"] = df["career"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__ord").drop(columns="__ord")

    lines: list[str] = []
    has_range = "net_risk_min" in df.columns and "net_risk_max" in df.columns
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    cap = "National 2034 employment under immediate vs. ramped GenAI disruption scenarios (from \\texttt{data/scenario\\_summary.csv})."
    if has_range:
        cap += " Careers are SOC bundles; employment-weighted outcomes; $\\NetRisk$ range is min--max across the bundle."
    # Clarify which index drives scenarios (important for judge auditability).
    cap += " $\\NetRisk$ values use the calibrated index when available; otherwise the uncalibrated mechanism index (see Definitions \\& Provenance)."
    lines.append("\\caption{" + cap + "}")
    lines.append("\\label{tab:scenario_summary}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    if has_range:
        lines.append("\\begin{tabular}{lrrlrrrrr}")
        lines.append("\\toprule")
        lines.append(
            "Career & $\\NetRisk$ & $\\NetRisk$ range & $E_{2024}$ & $E_{2034}$ (Baseline) & $E_{2034}$ (Moderate) & $E_{2034}$ (Ramp Mod.) & $E_{2034}$ (High) & $E_{2034}$ (Ramp High)\\\\"
        )
    else:
        lines.append("\\begin{tabular}{lrrrrrrr}")
        lines.append("\\toprule")
        lines.append(
            "Career & $\\NetRisk$ & $E_{2024}$ & $E_{2034}$ (Baseline) & $E_{2034}$ (Moderate) & $E_{2034}$ (Ramp Mod.) & $E_{2034}$ (High) & $E_{2034}$ (Ramp High)\\\\"
        )
    lines.append("\\midrule")

    for _, r in df.iterrows():
        emp24 = r.get("emp_2024")
        emp34_base = r.get("emp_2034_No_GenAI_Baseline")
        emp34_mod = r.get("emp_2034_Moderate_Substitution")
        emp34_ramp_mod = r.get("emp_2034_Ramp_Moderate")
        emp34_high = r.get("emp_2034_High_Disruption")
        emp34_ramp_high = r.get("emp_2034_Ramp_High")
        row_cells = [
            _latex_escape(r["career_label"]),
            _fmt_float(r.get("net_risk"), 3),
        ]
        if has_range:
            nmin = r.get("net_risk_min")
            nmax = r.get("net_risk_max")
            range_str = f"[{_fmt_float(nmin, 2)}, {_fmt_float(nmax, 2)}]" if nmin is not None and nmax is not None else "---"
            row_cells.append(range_str)
        row_cells.extend([
            _fmt_int(emp24),
            _fmt_int(emp34_base),
            _fmt_int(emp34_mod),
            _fmt_int(emp34_ramp_mod),
            _fmt_int(emp34_high),
            _fmt_int(emp34_ramp_high),
        ])
        lines.append(" & ".join(row_cells) + "\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "scenario_summary.tex", "\n".join(lines))


def build_summary_headline_fragment() -> None:
    """
    Write Summary Sheet headline numbers from scenario_summary.csv so they
    never drift from the scenario table. check_paper_consistency.py validates
    that main.tex (via this fragment) matches the CSV.
    """
    df = _load_required_csv(DATA_DIR / "scenario_summary.csv")
    df = df.set_index("career")

    # Order and display names expected by consistency checker
    careers = [
        ("software_engineer", "Software Developers"),
        ("electrician", "Electricians"),
        ("writer", "Writers and Authors"),
    ]

    def row(career_key: str) -> dict:
        r = df.loc[career_key]
        base = int(round(float(r["emp_2034_No_GenAI_Baseline"])))
        high = int(round(float(r["emp_2034_High_Disruption"])))
        ramp_high = int(round(float(r["emp_2034_Ramp_High"])))
        return {
            "base": base,
            "high": high,
            "ramp_high": ramp_high,
            "d_high": high - base,
            "d_ramp": ramp_high - base,
        }

    parts_immediate: list[str] = []
    parts_ramp: list[str] = []
    for key, label in careers:
        v = row(key)
        b = _fmt_int_latex(v["base"])
        h = _fmt_int_latex(v["high"])
        rh = _fmt_int_latex(v["ramp_high"])
        d_high = v["d_high"]
        d_ramp = v["d_ramp"]
        sign_high = "+" if d_high >= 0 else ""
        sign_ramp = "+" if d_ramp >= 0 else ""
        d_high_str = f"{sign_high}{d_high:,}".replace(",", "{,}")
        d_ramp_str = f"{sign_ramp}{d_ramp:,}".replace(",", "{,}")
        parts_immediate.append(f"{label} \\({b} \\rightarrow {h}\\) (\\({d_high_str}\\))")
        parts_ramp.append(f"{label} \\({b} \\rightarrow {rh}\\) (\\({d_ramp_str}\\))")

    immediate_str = ", ".join(parts_immediate)
    ramp_str = ", ".join(parts_ramp)
    # Format delta with LaTeX thousands: -585091 -> -585{,}091
    content = (
        " Under immediate High disruption vs.\\ baseline, employment shifts by: "
        + immediate_str
        + ". Ramp adoption reduces the magnitude of disruption: "
        + ramp_str
        + " (Table~\\ref{tab:scenario_summary})."
    )
    _write_text(TABLES_DIR / "summary_headline_fragment.tex", content)


def _load_occ_title_map() -> dict[str, str]:
    """
    Best-effort SOC -> title map for report tables.
    Primary source: data/occ_key.csv (built from OEWS).
    Fallback: O*NET Occupation Data.txt (if present).
    """
    title_map: dict[str, str] = {}
    occ_path = DATA_DIR / "occ_key.csv"
    if occ_path.exists():
        try:
            occ = pd.read_csv(occ_path)
            # occ_key schema: soc_code, soc_title
            if "soc_code" in occ.columns and "soc_title" in occ.columns:
                title_map.update(
                    occ.dropna(subset=["soc_code", "soc_title"])
                    .drop_duplicates(subset=["soc_code"])
                    .set_index("soc_code")["soc_title"]
                    .astype(str)
                    .to_dict()
                )
        except Exception:
            pass

    # Fallback: O*NET Occupation Data.txt (SOC slice)
    onet_occ_path = DATA_DIR / "onet" / "Occupation Data.txt"
    if onet_occ_path.exists():
        try:
            onet_occ = pd.read_csv(
                onet_occ_path,
                sep="\t",
                header=None,
                usecols=[0, 1],
                names=["onet_soc", "onet_title"],
                on_bad_lines="skip",
            )
            onet_occ["occ_code"] = onet_occ["onet_soc"].astype(str).str.slice(0, 7)
            title_map.update(
                onet_occ.dropna(subset=["occ_code", "onet_title"])
                .drop_duplicates(subset=["occ_code"])
                .set_index("occ_code")["onet_title"]
                .astype(str)
                .to_dict()
            )
        except Exception:
            pass

    return title_map


def build_netrisk_index_compare_table() -> None:
    """
    Compare the interpretability (uncalibrated) NetRisk to the calibrated (predictive) NetRisk.

    Uses data/mechanism_risk_scored.csv which is written by run_scenarios.py and already
    includes net_risk_uncalibrated and (when available) net_risk_calibrated + net_risk_source.
    """
    path = DATA_DIR / "mechanism_risk_scored.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    x_col = "net_risk_uncalibrated"
    y_col = "net_risk_calibrated" if "net_risk_calibrated" in df.columns else "net_risk"
    if x_col not in df.columns or y_col not in df.columns:
        return

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) == 0:
        return

    x0 = x[m]
    y0 = y[m]
    pearson = float(x0.corr(y0, method="pearson"))
    spearman = float(x0.corr(y0, method="spearman"))

    eps = 1e-6
    sx = np.sign(x0.to_numpy())
    sy = np.sign(y0.to_numpy())
    # Treat near-zero values as 0 for sign comparisons.
    sx[np.abs(x0.to_numpy()) < eps] = 0.0
    sy[np.abs(y0.to_numpy()) < eps] = 0.0
    sign_agree = float(np.mean(sx == sy))
    n_flip = int(np.sum((sx != sy) & (sx != 0.0) & (sy != 0.0)))

    # Basic dispersion of differences (scale differs, but this helps audit "where disagree")
    abs_diff = np.abs(x0.to_numpy() - y0.to_numpy())
    med_abs_diff = float(np.median(abs_diff))
    p90_abs_diff = float(np.quantile(abs_diff, 0.9))

    # How many occupations used the calibrated index in the pipeline?
    n_cal = 0
    if "net_risk_source" in df.columns:
        n_cal = int((df.loc[m, "net_risk_source"].astype(str) == "calibrated").sum())

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Comparison of the interpretability (uncalibrated) NetRisk index vs. the calibrated (predictive) NetRisk index across occupations (from \\texttt{data/mechanism\\_risk\\_scored.csv}).}"
    )
    lines.append("\\label{tab:netrisk_index_compare}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Quantity & Value\\\\")
    lines.append("\\midrule")
    lines.append(f"Occupations with both indices & {int(m.sum()):,}\\\\")
    if n_cal:
        lines.append(f"Occupations using calibrated NetRisk in pipeline & {n_cal:,}\\\\")
    lines.append(f"Pearson correlation ($r$) & {_fmt_float(pearson, 3)}\\\\")
    lines.append(f"Spearman rank correlation ($\\rho$) & {_fmt_float(spearman, 3)}\\\\")
    lines.append(f"Sign agreement rate & {_fmt_float(100.0 * sign_agree, 1)}\\%\\\\")
    lines.append(f"Sign disagreements (nonzero vs nonzero) & {n_flip:,}\\\\")
    lines.append(f"Median $|\\Delta|$ & {_fmt_float(med_abs_diff, 3)}\\\\")
    lines.append(f"90th percentile $|\\Delta|$ & {_fmt_float(p90_abs_diff, 3)}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "netrisk_index_compare.tex", "\n".join(lines))


def build_netrisk_index_disagreement_table(top_n: int = 10) -> None:
    """
    Show concrete examples of where the calibrated and uncalibrated indices disagree.
    Primary: rows with opposite sign (excluding near-zero), ranked by |difference|.
    Fallback: overall largest |difference|.
    """
    path = DATA_DIR / "mechanism_risk_scored.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    x_col = "net_risk_uncalibrated"
    y_col = "net_risk_calibrated" if "net_risk_calibrated" in df.columns else "net_risk"
    if "occ_code" not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return

    df = df.copy()
    df["x"] = pd.to_numeric(df[x_col], errors="coerce")
    df["y"] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["x", "y", "occ_code"]).copy()
    if df.empty:
        return

    eps = 1e-6
    sx = np.sign(df["x"].to_numpy())
    sy = np.sign(df["y"].to_numpy())
    sx[np.abs(df["x"].to_numpy()) < eps] = 0.0
    sy[np.abs(df["y"].to_numpy()) < eps] = 0.0
    df["sign_flip"] = (sx != sy) & (sx != 0.0) & (sy != 0.0)
    df["abs_diff"] = np.abs(df["x"] - df["y"])

    title_map = _load_occ_title_map()
    df["occ_title"] = df["occ_code"].astype(str).map(title_map).fillna("")

    pick = df.loc[df["sign_flip"]].sort_values("abs_diff", ascending=False).head(top_n)
    if pick.empty:
        pick = df.sort_values("abs_diff", ascending=False).head(top_n)

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Examples where the calibrated vs. uncalibrated NetRisk indices disagree most (ranked by $|\\Delta|$; primary filter is opposite sign when available).}"
    )
    lines.append("\\label{tab:netrisk_index_disagree}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("SOC & Title & NetRisk (uncal.) & NetRisk (cal.) & $\\Delta$ & $|\\Delta|$\\\\")
    lines.append("\\midrule")
    for _, r in pick.iterrows():
        title = str(r.get("occ_title", "")).strip()
        if not title:
            title = "Unknown"
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(r["occ_code"])),
                    _latex_escape(title),
                    _fmt_float(r["x"], 3),
                    _fmt_float(r["y"], 3),
                    _fmt_float(float(r["x"] - r["y"]), 3),
                    _fmt_float(r["abs_diff"], 3),
                ]
            )
            + "\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "netrisk_index_disagree.tex", "\n".join(lines))


def build_bundle_contents_fragment() -> None:
    """
    Build a small LaTeX fragment (no floating table env) listing which SOC codes
    are included in each career bundle, and their employment weights (EP E2024 shares).
    """
    careers = [
        ("Software Developers (STEM)", DATA_DIR / "careers" / "software_engineer.csv"),
        ("Electricians (Trade)", DATA_DIR / "careers" / "electrician.csv"),
        ("Writers and Authors (Arts)", DATA_DIR / "careers" / "writer.csv"),
    ]

    rows: list[dict] = []
    for label, p in careers:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        nat = df.loc[df.get("area_type") == 1].copy()
        if nat.empty:
            continue
        # EP employment is in thousands in occupation.xlsx; keep as jobs for readability.
        nat["emp_2024_jobs"] = pd.to_numeric(nat.get("emp_2024"), errors="coerce") * 1000.0
        nat = nat.dropna(subset=["occ_code", "occ_title", "emp_2024_jobs"]).copy()
        tot = float(nat["emp_2024_jobs"].sum()) if float(nat["emp_2024_jobs"].sum()) > 0 else 1.0
        nat["share_pct"] = 100.0 * (nat["emp_2024_jobs"] / tot)
        for _, r in nat.iterrows():
            rows.append(
                {
                    "bundle": label,
                    "occ_code": str(r["occ_code"]),
                    "occ_title": str(r["occ_title"]),
                    "emp_2024_jobs": float(r["emp_2024_jobs"]),
                    "share_pct": float(r["share_pct"]),
                }
            )

    if not rows:
        return
    out = pd.DataFrame(rows)

    # Stable order: bundle label then descending share
    bundle_order = {b: i for i, b in enumerate([c[0] for c in careers])}
    out["__b"] = out["bundle"].map(bundle_order).fillna(999)
    out = out.sort_values(["__b", "share_pct"], ascending=[True, False]).drop(columns="__b")

    lines: list[str] = []
    lines.append("% LaTeX fragment (no floating environment)")
    lines.append("\\begin{tabular}{lllr}")
    lines.append("\\toprule")
    lines.append("Bundle & SOC & Occupation & Share of bundle $E_{2024}$\\\\")
    lines.append("\\midrule")
    for _, r in out.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["bundle"]),
                    _latex_escape(r["occ_code"]),
                    _latex_escape(r["occ_title"]),
                    _fmt_float(r["share_pct"], 1) + "\\%",
                ]
            )
            + "\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    _write_text(TABLES_DIR / "bundle_contents_fragment.tex", "\n".join(lines))


def build_scenario_parameter_table() -> None:
    """
    Build a small table documenting the scenario shock parameters actually used.
    This supports auditability of the claim that 'Moderate/High' are scaled to a
    reference point (e.g., p90 of positive NetRisk) when calibration outputs exist.
    """
    path = DATA_DIR / "scenario_parameters.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    # Focus on the key scenarios referenced in the text; keep ramp rows too for completeness.
    order = ["No_GenAI_Baseline", "Moderate_Substitution", "High_Disruption", "Ramp_Moderate", "Ramp_High"]
    df["__ord"] = df["scenario"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__ord").drop(columns="__ord")

    # Optional calibration metadata columns
    has_p90 = "p90_positive_netrisk" in df.columns and df["p90_positive_netrisk"].notna().any()
    has_target = "target_delta_g_at_p90" in df.columns and df["target_delta_g_at_p90"].notna().any()

    cols = ["scenario", "s_value", "source"]
    if has_target:
        cols.insert(2, "target_delta_g_at_p90")
    if has_p90:
        cols.insert(3 if has_target else 2, "p90_positive_netrisk")

    def _label(s: str) -> str:
        return str(s).replace("_", " ")

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Scenario strength parameters used by the pipeline (from \\texttt{data/scenario\\_parameters.csv}).}")
    lines.append("\\label{tab:scenario_params}")

    if has_target and has_p90:
        lines.append("\\begin{tabular}{lrrrr}")
        lines.append("\\toprule")
        lines.append("Scenario & $s$ & Target $\\Delta g$ at $p90$ & $p90(\\NetRisk_+)$ & Source\\\\")
        lines.append("\\midrule")
        for _, r in df.iterrows():
            lines.append(
                " & ".join(
                    [
                        _latex_escape(_label(r["scenario"])),
                        _fmt_float(r.get("s_value"), 4),
                        _fmt_float(r.get("target_delta_g_at_p90"), 3) if r.get("target_delta_g_at_p90") is not None else "",
                        _fmt_float(r.get("p90_positive_netrisk"), 3) if r.get("p90_positive_netrisk") is not None else "",
                        _latex_escape(str(r.get("source", ""))),
                    ]
                )
                + "\\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
    else:
        lines.append("\\begin{tabular}{lrl}")
        lines.append("\\toprule")
        lines.append("Scenario & $s$ & Source\\\\")
        lines.append("\\midrule")
        for _, r in df.iterrows():
            lines.append(
                " & ".join(
                    [
                        _latex_escape(_label(r["scenario"])),
                        _fmt_float(r.get("s_value"), 4),
                        _latex_escape(str(r.get("source", ""))),
                    ]
                )
                + "\\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "scenario_params.tex", "\n".join(lines))


def build_sensitivity_grid_table(shocks: Iterable[float] = (0.01, 0.015, 0.02, 0.03)) -> None:
    base = _load_required_csv(DATA_DIR / "scenario_summary.csv").copy()
    base["shock_dummy"] = 0.0

    career_label = {
        "software_engineer": "Software Dev",
        "electrician": "Electrician",
        "writer": "Writer",
    }
    base["career_label"] = base["career"].map(career_label).fillna(base["career"])
    base = base.sort_values(["career_label"])

    # Compute 2034 employment at each shock, using the same equation as run_scenarios.py
    rows: list[dict] = []
    for _, r in base.iterrows():
        m_comp = float(r.get("m_comp", M_MAX_DEFAULT))
        for s in shocks:
            g_adj = get_g_adj(float(r["g_baseline"]), float(r["net_risk"]), float(s), m_comp=m_comp)
            emp_2034 = float(r["emp_2024"]) * ((1.0 + g_adj) ** 10)
            rows.append(
                {
                    "career": r["career_label"],
                    "shock": s,
                    "emp_2034": emp_2034,
                }
            )
    grid = pd.DataFrame(rows)
    piv = grid.pivot(index="career", columns="shock", values="emp_2034")

    shock_cols = list(piv.columns)

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Sensitivity of 2034 employment to the scenario parameter $s$ (using piecewise mapping: $s_{sub}=s, s_{comp}=m\\cdot s$ with $m\\le m_{\\max}=0.2$; $m$ varies by occupation bundle based on bottlenecks and coarse demand elasticity).}"
    )
    lines.append("\\label{tab:sensitivity_grid}")
    col_spec = "l" + "r" * len(shock_cols)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = ["Career"] + [f"$s={s:.3f}$" for s in shock_cols]
    lines.append(" & ".join(header) + "\\\\")
    lines.append("\\midrule")
    for career in piv.index:
        vals = [_fmt_int(piv.loc[career, s]) for s in shock_cols]
        lines.append(" & ".join([_latex_escape(career)] + vals) + "\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "sensitivity_grid.tex", "\n".join(lines))


def build_complementarity_sensitivity_table(
    m_max_values: Iterable[float] = (0.10, 0.20, 0.30),
) -> None:
    """
    Sensitivity to the complementarity *bound* m_max in the scenario mapping.
    We recompute 2034 employment under Moderate and High scenarios while varying the
    upper bound applied to the occupation-level complementarity multiplier:

      g_adj = g_base - s * NetRisk                    if NetRisk >= 0
      g_adj = g_base + (m_i*s)*(-NetRisk)             if NetRisk < 0
      m_i = min(m_max, m_raw),  m_raw computed from bottleneck × elasticity
    """
    df = _load_required_csv(DATA_DIR / "scenario_summary.csv").copy()

    # Pull scenario strengths actually used by the pipeline (calibrated if available).
    s_mod = 0.015
    s_high = 0.03
    scen_param_path = DATA_DIR / "scenario_parameters.csv"
    if scen_param_path.exists():
        try:
            sp = pd.read_csv(scen_param_path)
            mod_row = sp.loc[sp["scenario"] == "Moderate_Substitution"]
            high_row = sp.loc[sp["scenario"] == "High_Disruption"]
            if not mod_row.empty:
                s_mod = float(mod_row.iloc[0].get("s_value", s_mod))
            if not high_row.empty:
                s_high = float(high_row.iloc[0].get("s_value", s_high))
        except Exception:
            pass

    career_label = {
        "software_engineer": "Software Developers",
        "electrician": "Electricians",
        "writer": "Writers and Authors",
    }
    df["career_label"] = df["career"].map(career_label).fillna(df["career"])
    order = ["software_engineer", "electrician", "writer"]
    df["__ord"] = df["career"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__ord").drop(columns="__ord")

    def _g_adj(g_base: float, risk: float, s: float, m_comp: float) -> float:
        if risk >= 0:
            return g_base - (s * risk)
        return g_base + (m_comp * s * (-risk))

    rows: list[dict] = []
    for _, r in df.iterrows():
        g_base = float(r["g_baseline"])
        risk = float(r["net_risk"])
        emp_2024 = float(r["emp_2024"])
        m_raw = float(r.get("m_comp_raw", r.get("m_comp", M_MAX_DEFAULT)))
        m_raw = max(0.0, m_raw)
        for m_max in m_max_values:
            m_eff = min(float(m_max), m_raw)
            g_mod = _g_adj(g_base, risk, s_mod, m_eff)
            g_hi = _g_adj(g_base, risk, s_high, m_eff)
            rows.append(
                {
                    "career": str(r["career_label"]),
                    "m_max": float(m_max),
                    "m_eff": float(m_eff),
                    "emp_2034_mod": emp_2024 * ((1.0 + g_mod) ** 10),
                    "emp_2034_high": emp_2024 * ((1.0 + g_hi) ** 10),
                }
            )

    out = pd.DataFrame(rows)

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Sensitivity to the complementarity cap in the scenario mapping (varies the bound $m_{\\max}$ applied to the occupation-level uplift multiplier; baseline uses $m_{\\max}=0.2$).}"
    )
    lines.append("\\label{tab:comp_factor_sensitivity}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Career & $m_{\\max}$ & $m_{\\mathrm{eff}}$ & $E_{2034}$ (Moderate) & $E_{2034}$ (High)\\\\")
    lines.append("\\midrule")

    for _, r in out.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["career"]),
                    _fmt_float(r["m_max"], 2),
                    _fmt_float(r["m_eff"], 2),
                    _fmt_int(r["emp_2034_mod"]),
                    _fmt_int(r["emp_2034_high"]),
                ]
            )
            + "\\\\"
        )

    lines.append("\\bottomrule")
    lines.append(
        "\\multicolumn{5}{l}{\\footnotesize Note: if a bundle has $\\NetRisk\\ge 0$, the complementarity branch is inactive, so varying $m_{\\max}$ has no effect (rows shown for completeness).}\\\\"
    )
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "comp_factor_sensitivity.tex", "\n".join(lines))


def build_comp_cap_components_table() -> None:
    """
    Auditability helper: show bottleneck B, normalized demand-responsiveness proxy eps~,
    and resulting m_eff for the three focal career bundles.
    """
    mech_path = DATA_DIR / "mechanism_risk_scored.csv"
    if not mech_path.exists():
        return
    mech = pd.read_csv(mech_path)
    if mech.empty or "occ_code" not in mech.columns:
        return

    need = ["occ_code", "bottleneck_B", "demand_elasticity_eps", "m_comp"]
    for c in need:
        if c not in mech.columns:
            return
    mech = mech[need].copy()

    bundles = [
        ("Software Developers (STEM)", DATA_DIR / "careers" / "software_engineer.csv"),
        ("Electricians (Trade)", DATA_DIR / "careers" / "electrician.csv"),
        ("Writers and Authors (Arts)", DATA_DIR / "careers" / "writer.csv"),
    ]

    rows: list[dict] = []
    for label, p in bundles:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        nat = df.loc[df.get("area_type") == 1].copy()
        if nat.empty:
            continue
        nat["emp_2024_jobs"] = pd.to_numeric(nat.get("emp_2024"), errors="coerce") * 1000.0
        nat = nat.dropna(subset=["occ_code", "emp_2024_jobs"]).copy()
        if nat.empty or float(nat["emp_2024_jobs"].sum()) <= 0:
            continue
        nat = nat.merge(mech, on="occ_code", how="left")
        w = nat["emp_2024_jobs"] / float(nat["emp_2024_jobs"].sum())

        B = float((pd.to_numeric(nat["bottleneck_B"], errors="coerce").fillna(0.0) * w).sum())
        eps_tilde = float((pd.to_numeric(nat["demand_elasticity_eps"], errors="coerce").fillna(0.0) * w).sum())
        m_eff = float((pd.to_numeric(nat["m_comp"], errors="coerce").fillna(0.0) * w).sum())
        rows.append({"bundle": label, "B": B, "eps_tilde": eps_tilde, "m_eff": m_eff})

    if not rows:
        return

    out = pd.DataFrame(rows)

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Components of the occupation-level complementarity cap for the three focal career bundles (employment-weighted over SOCs using EP $E_{2024}$ shares). $B$ is a bottleneck index; $\\tilde{\\varepsilon}$ is a normalized demand-responsiveness proxy (not a literal elasticity); $m_{\\mathrm{eff}}=\\min(m_{\\max},(1-B)\\tilde{\\varepsilon})$ with baseline $m_{\\max}=0.2$. Reported for auditability; used only when $\\NetRisk<0$.}"
    )
    lines.append("\\label{tab:comp_cap_components}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Bundle & \\(B\\) & \\(\\tilde{\\varepsilon}\\) & \\(m_{\\mathrm{eff}}\\)\\\\")
    lines.append("\\midrule")
    for _, r in out.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(r["bundle"])),
                    _fmt_float(r["B"], 3),
                    _fmt_float(r["eps_tilde"], 3),
                    _fmt_float(r["m_eff"], 3),
                ]
            )
            + "\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "comp_cap_components.tex", "\n".join(lines))


def build_top_exposed_sheltered_table(top_n: int = 10) -> None:
    mech = _load_required_csv(DATA_DIR / "mechanism_risk_scored.csv")
    occ = _load_required_csv(DATA_DIR / "occ_key.csv")
    occ = occ.rename(columns={"soc_code": "occ_code", "soc_title": "occ_title"})
    occ = occ[["occ_code", "occ_title"]].drop_duplicates()

    mech = mech.merge(occ, on="occ_code", how="left")
    # Fill any missing titles using O*NET occupation titles (prevents "nan" in the report table).
    missing = mech["occ_title"].isna() if "occ_title" in mech.columns else pd.Series(False, index=mech.index)
    if bool(missing.any()):
        onet_occ_path = DATA_DIR / "onet" / "Occupation Data.txt"
        if onet_occ_path.exists():
            try:
                onet_occ = pd.read_csv(
                    onet_occ_path,
                    sep="\t",
                    header=None,
                    usecols=[0, 1],
                    names=["onet_soc", "onet_title"],
                    on_bad_lines="skip",
                )
                onet_occ["occ_code"] = onet_occ["onet_soc"].astype(str).str.slice(0, 7)
                title_map = (
                    onet_occ.dropna(subset=["occ_code", "onet_title"])
                    .drop_duplicates(subset=["occ_code"])
                    .set_index("occ_code")["onet_title"]
                    .to_dict()
                )
                mech.loc[missing, "occ_title"] = mech.loc[missing, "occ_code"].map(title_map)
            except Exception:
                # Safe fallback: leave as NaN, we'll render as "Unknown" below.
                pass

    top = mech.sort_values("net_risk", ascending=False).head(top_n)
    bot = mech.sort_values("net_risk", ascending=True).head(top_n)

    def _rows(df: pd.DataFrame) -> list[str]:
        out = []
        for _, r in df.iterrows():
            title = r.get("occ_title", "")
            if title is None or (isinstance(title, float) and math.isnan(title)) or str(title).strip().lower() == "nan" or str(title).strip() == "":
                title = "Unknown"
            out.append(
                " & ".join(
                    [
                        _latex_escape(r.get("occ_code", "")),
                        _latex_escape(title),
                        _fmt_float(r.get("net_risk"), 3),
                    ]
                )
                + "\\\\"
            )
        return out

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Sanity check: most exposed vs. most sheltered occupations by Net Risk (computed from \\texttt{data/mechanism\\_risk\\_scored.csv}).}")
    lines.append("\\label{tab:top_exposed_sheltered}")
    lines.append("\\begin{tabular}{llr}")
    lines.append("\\toprule")
    lines.append("\\multicolumn{3}{c}{Top exposed (highest $\\NetRisk$)}\\\\")
    lines.append("\\midrule")
    lines.append("SOC & Title & $\\NetRisk$\\\\")
    lines.extend(_rows(top))
    lines.append("\\midrule")
    lines.append("\\multicolumn{3}{c}{Top sheltered (lowest $\\NetRisk$)}\\\\")
    lines.append("\\midrule")
    lines.append("SOC & Title & $\\NetRisk$\\\\")
    lines.extend(_rows(bot))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "top_exposed_sheltered.tex", "\n".join(lines))


def build_openings_summary_table() -> None:
    """
    Build a table showing annual openings, training demand proxy, and program sizing rules
    for the three focus careers.
    """
    ep = _load_required_csv(DATA_DIR / "ep_baseline.csv")
    
    # SOC codes for the three careers
    career_soc = {
        "15-1252": "Software Developers (STEM)",
        "47-2111": "Electricians (Trade)",
        "27-3043": "Writers and Authors (Arts)",
    }
    
    # Filter to our three careers
    ep_filtered = ep[ep["occ_code"].isin(career_soc.keys())].copy()
    ep_filtered["career_label"] = ep_filtered["occ_code"].map(career_soc)
    
    # Stable order matching other tables
    order = ["15-1252", "47-2111", "27-3043"]
    ep_filtered["__ord"] = ep_filtered["occ_code"].apply(
        lambda x: order.index(x) if x in order else 999
    )
    ep_filtered = ep_filtered.sort_values("__ord").drop(columns="__ord")
    
    def _get_program_sizing_rule(openings: float) -> str:
        """Interpret openings level into a program sizing recommendation."""
        if openings >= 80:
            return "High openings: maintain/grow capacity even if net growth slows"
        elif openings >= 30:
            return "Moderate openings: maintain capacity, monitor trends"
        else:
            return "Lower openings: consolidate/specialize toward high-value niches"
    
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Annual openings and program sizing implications (from \\texttt{data/ep\\_baseline.csv}).}")
    lines.append("\\label{tab:openings_summary}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lrlp{5.5cm}}")
    lines.append("\\toprule")
    lines.append("Career & Annual Openings (thousands) & Training Demand Proxy & Program Sizing Rule\\\\")
    lines.append("\\midrule")
    
    for _, r in ep_filtered.iterrows():
        openings = r.get("annual_openings", 0)
        # Training demand proxy: convert from thousands to actual annual openings
        training_demand = openings * 1000 if not pd.isna(openings) else 0
        sizing_rule = _get_program_sizing_rule(openings)
        
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["career_label"]),
                    _fmt_float(openings, 1),
                    _fmt_int(training_demand),
                    _latex_escape(sizing_rule),
                ]
            )
            + "\\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    
    _write_text(TABLES_DIR / "openings_summary.tex", "\n".join(lines))


def build_mechanism_coverage_table() -> None:
    """
    Compute coverage counts to support a transparent mapping story:
    O*NET text files -> SOC slice -> relevant elements -> scored occupations -> joins.
    """
    onet_dir = DATA_DIR / "onet"
    files = ["Work Activities.txt", "Abilities.txt", "Skills.txt"]

    frames = []
    for fn in files:
        p = onet_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing O*NET file: {p}")
        df = pd.read_csv(p, sep="\t", on_bad_lines="skip")
        if "Scale ID" in df.columns:
            df = df[df["Scale ID"] == "IM"].copy()
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    # Keep only columns we need; names match build_mechanism_layer_expanded.py
    full = full[["O*NET-SOC Code", "Element Name", "Data Value"]].copy()
    full["soc_code"] = full["O*NET-SOC Code"].astype(str).str.slice(0, 7)

    # Import the exact descriptor map from the mechanism build to avoid mismatch.
    from build_mechanism_layer_expanded import DIMENSION_DESCRIPTORS  # type: ignore

    keywords = [(dim, k.lower()) for dim, ks in DIMENSION_DESCRIPTORS.items() for k in ks]

    def _is_relevant(elem_name: str) -> bool:
        e = str(elem_name).lower()
        for _, k in keywords:
            if k in e or e in k:
                return True
        return False

    # Relevant elements: those that match any descriptor keyword (same matching rule as build_mechanism_layer_expanded)
    full["is_relevant"] = full["Element Name"].map(_is_relevant)
    relevant = full.loc[full["is_relevant"]].copy()

    # Loads of downstream artifacts
    mech_all = _load_required_csv(DATA_DIR / "mechanism_layer_all.csv")
    oews_nat = _load_required_csv(DATA_DIR / "oews_national.csv")
    ep = _load_required_csv(DATA_DIR / "ep_baseline.csv")

    # Unique counts
    n_onet_soc = int(full["O*NET-SOC Code"].nunique())
    n_soc_slice = int(full["soc_code"].nunique())
    n_soc_with_relevant = int(relevant["soc_code"].nunique())
    n_mech_scored = int(mech_all["occ_code"].nunique())

    n_oews_nat = int(oews_nat["occ_code"].astype(str).str.strip().nunique())
    n_ep = int(ep["occ_code"].astype(str).str.strip().nunique()) if "occ_code" in ep.columns else 0

    mech_codes = set(mech_all["occ_code"].astype(str).str.strip())
    oews_codes = set(oews_nat["occ_code"].astype(str).str.strip())
    ep_codes = set(ep["occ_code"].astype(str).str.strip()) if "occ_code" in ep.columns else set()

    n_mech_and_oews = len(mech_codes & oews_codes)
    n_mech_and_ep = len(mech_codes & ep_codes)
    n_mech_and_oews_and_ep = len(mech_codes & oews_codes & ep_codes)

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Coverage and mapping audit: O*NET \\textrightarrow{} SOC \\textrightarrow{} scored mechanism layer \\textrightarrow{} BLS tables.}")
    lines.append("\\label{tab:mechanism_coverage}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Quantity & Count\\\\")
    lines.append("\\midrule")
    lines.append(f"O*NET-SOC codes in loaded O*NET files (IM scale) & {n_onet_soc:,}\\\\")
    lines.append(f"Unique SOC (7-char slice) in loaded O*NET files & {n_soc_slice:,}\\\\")
    lines.append(f"SOC with at least one relevant element for our 5 dimensions & {n_soc_with_relevant:,}\\\\")
    lines.append(f"SOC with mechanism scores written to \\texttt{{mechanism\\_layer\\_all.csv}} & {n_mech_scored:,}\\\\")
    lines.append("\\midrule")
    lines.append(f"Unique occupations in OEWS national table & {n_oews_nat:,}\\\\")
    lines.append(f"Unique occupations in EP baseline table & {n_ep:,}\\\\")
    lines.append(f"SOC present in both mechanism layer and OEWS national & {n_mech_and_oews:,}\\\\")
    lines.append(f"SOC present in both mechanism layer and EP baseline & {n_mech_and_ep:,}\\\\")
    lines.append(f"SOC present in mechanism layer, OEWS, and EP & {n_mech_and_oews_and_ep:,}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "mechanism_coverage.tex", "\n".join(lines))


def build_local_context_table() -> None:
    """
    Build a table showing local labor market context for each institution:
    - Local metro employment and wages
    - Wage premium vs national
    - Attractiveness score
    """
    CAREERS_DIR = DATA_DIR / "careers"
    oews_local = _load_required_csv(DATA_DIR / "oews_institution_local.csv")
    oews_nat = _load_required_csv(DATA_DIR / "oews_national.csv")
    oews_local["area_code"] = oews_local["area_code"].astype(str).str.strip()
    oews_nat["occ_code"] = oews_nat["occ_code"].astype(str).str.strip()
    oews_local["occ_code"] = oews_local["occ_code"].astype(str).str.strip()

    nat_total_row = oews_nat[oews_nat["occ_code"] == "00-0000"]
    nat_total_emp = float(nat_total_row.iloc[0]["emp"]) if not nat_total_row.empty else np.nan
    
    # Institution definitions: career file, area_code, institution name
    institutions = [
        {"career_file": "software_engineer.csv", "area_code": "41740", "state_code": "6", "inst": "SDSU", "career": "Software Developers"},
        {"career_file": "electrician.csv", "area_code": "31080", "state_code": "6", "inst": "LATTC", "career": "Electricians"},
        {"career_file": "writer.csv", "area_code": "41860", "state_code": "6", "inst": "Academy of Art", "career": "Writers and Authors"},
    ]
    
    rows: list[dict] = []
    
    for inst_info in institutions:
        career_path = CAREERS_DIR / inst_info["career_file"]
        if not career_path.exists():
            raise FileNotFoundError(f"Missing career file: {career_path}")
        
        df = pd.read_csv(career_path)
        
        # Get national data first (area_code=99 or 1)
        national_row = df[df["area_code"].astype(str).isin(["99", "1"])]
        if national_row.empty:
            raise ValueError(f"No national data found for {inst_info['career']}")
        national = national_row.iloc[0]
        national_wage = float(national["a_mean"])

        # Prefer metro; fallback to state; then fallback to national.
        local_row = df[df["area_code"].astype(str) == inst_info["area_code"]]
        geo_used = "metro"
        if local_row.empty and inst_info.get("state_code") is not None:
            local_row = df[df["area_code"].astype(str) == str(inst_info["state_code"])]
            geo_used = "state" if not local_row.empty else geo_used
        if local_row.empty:
            local = national
            geo_used = "national"
            metro_name = "U.S. (national fallback)"
            local_emp = float(national["emp"])
            local_wage = national_wage
        else:
            local = local_row.iloc[0]
            local_emp = float(local["emp"])
            local_wage = float(local["a_mean"]) if not pd.isna(local.get("a_mean")) else national_wage
            metro_name = str(local["area_title"])
            if geo_used == "state":
                metro_name = f"{metro_name} (state)"
        
        # Compute wage premium (1.0 when using national fallback)
        wage_premium = local_wage / national_wage if national_wage > 0 else 1.0

        # Location quotient (LQ): (local share of occ) / (national share of occ)
        nat_occ_emp = float(national["emp"])
        local_total_row = oews_local[
            (oews_local["area_code"].astype(str) == str(local["area_code"]))
            & (oews_local["occ_code"] == "00-0000")
        ]
        local_total_emp = float(local_total_row.iloc[0]["emp"]) if not local_total_row.empty else np.nan
        if not np.isnan(local_total_emp) and not np.isnan(nat_total_emp) and local_total_emp > 0 and nat_total_emp > 0 and nat_occ_emp > 0:
            lq = (local_emp / local_total_emp) / (nat_occ_emp / nat_total_emp)
        else:
            lq = 1.0
        
        # Normalize local employment (min-max normalization across all three institutions)
        # We'll compute this after collecting all local_emp values
        rows.append({
            "institution": inst_info["inst"],
            "metro": metro_name,
            "local_emp": local_emp,
            "local_wage": local_wage,
            "national_wage": national_wage,
            "wage_premium": wage_premium,
            "lq": lq,
        })
    
    # Normalize employment across all institutions for scoring
    emp_values = [r["local_emp"] for r in rows]
    emp_min = min(emp_values)
    emp_max = max(emp_values)
    emp_range = emp_max - emp_min if emp_max > emp_min else 1.0
    
    # Compute attractiveness scores
    lq_values = [r["lq"] for r in rows]
    lq_min = min(lq_values)
    lq_max = max(lq_values)
    lq_range = lq_max - lq_min if lq_max > lq_min else 1.0

    for r in rows:
        normalized_emp = (r["local_emp"] - emp_min) / emp_range if emp_range > 0 else 0.5
        normalized_lq = (r["lq"] - lq_min) / lq_range if lq_range > 0 else 0.5
        attractiveness_score = (r["wage_premium"] * 0.4) + (normalized_emp * 0.3) + (normalized_lq * 0.3)
        r["normalized_emp"] = normalized_emp
        r["normalized_lq"] = normalized_lq
        r["attractiveness_score"] = attractiveness_score
    
    # Build LaTeX table
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Local labor market context for each institution (from \\texttt{data/careers/*.csv}).}")
    lines.append("\\label{tab:local_context}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("Institution & Metro & Local Emp & Wage Premium & LQ & Attractiveness Score\\\\")
    lines.append("\\midrule")
    
    for r in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["institution"]),
                    _latex_escape(r["metro"]),
                    _fmt_int(r["local_emp"]),
                    _fmt_float(r["wage_premium"], 3),
                    _fmt_float(r["lq"], 3),
                    _fmt_float(r["attractiveness_score"], 3),
                ]
            )
            + "\\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    
    _write_text(TABLES_DIR / "local_context.tex", "\n".join(lines))


def build_program_sizing_table() -> None:
    """
    Estimate local annual openings and recommended program seats (intake) based on target market share.
    Assumption: seats = (local_openings * share) / (completion_rate * placement_rate)
    
    UPDATED: Computes intervals based on sensitivity analysis of efficiency and LQ.
    """
    CAREERS_DIR = DATA_DIR / "careers"
    ep = _load_required_csv(DATA_DIR / "ep_baseline.csv")
    oews_local = _load_required_csv(DATA_DIR / "oews_institution_local.csv")
    oews_nat = _load_required_csv(DATA_DIR / "oews_national.csv")
    oews_local["area_code"] = oews_local["area_code"].astype(str).str.strip()
    oews_nat["occ_code"] = oews_nat["occ_code"].astype(str).str.strip()
    oews_local["occ_code"] = oews_local["occ_code"].astype(str).str.strip()

    nat_total_row = oews_nat[oews_nat["occ_code"] == "00-0000"]
    nat_total_emp = float(nat_total_row.iloc[0]["emp"]) if not nat_total_row.empty else np.nan
    
    # Using the primary SOC for each career to lookup openings (since EP data is per-SOC)
    # Ideally we'd sum openings for the bundle, but for sizing we often focus on the core role.
    # Let's stick to the primary SOC for simplicity or sum if possible.
    # Given build_tables.py saves the bundle, let's use the primary SOC defined here.
    institutions = [
        {"career_file": "software_engineer.csv", "area_code": "41740", "state_code": "6", "inst": "SDSU", "career_soc": "15-1252"},
        {"career_file": "electrician.csv", "area_code": "31080", "state_code": "6", "inst": "LATTC", "career_soc": "47-2111"},
        {"career_file": "writer.csv", "area_code": "41860", "state_code": "6", "inst": "Academy of Art", "career_soc": "27-3043"},
    ]
    
    # Sensitivity Ranges
    # Use rounded endpoints for consistency with the report note and table rounding.
    # (The underlying multiplicative examples are 0.65*0.60=0.39 and 0.85*0.85=0.7225.)
    efficiency_low = 0.39
    efficiency_high = 0.72
    
    target_shares = [0.05, 0.10, 0.15]
    
    # LQ Clipping range for openings estimation
    lq_clip_low = 0.5
    lq_clip_high = 1.5
    
    rows = []
    robustness_rows = []
    
    for info in institutions:
        # 1. Get national openings for this SOC
        ep_row = ep[ep["occ_code"] == info["career_soc"]]
        if ep_row.empty:
            nat_openings = 0
        else:
            nat_openings = float(ep_row.iloc[0]["annual_openings"]) * 1000
        
        # 2. Get local vs national OEWS employment to scale national openings to local openings.
        career_path = CAREERS_DIR / info["career_file"]
        if not career_path.exists():
            continue
        df = pd.read_csv(career_path)
        
        # Determine local share and LQ
        nat_row = df[df["area_type"].isin([1, 99])].iloc[0] if not df[df["area_type"].isin([1, 99])].empty else None
        nat_emp = float(nat_row["emp"]) if nat_row is not None else 0
        
        local_row = df[df["area_code"].astype(str) == info["area_code"]]
        if local_row.empty and info.get("state_code"):
             local_row = df[df["area_code"].astype(str) == str(info["state_code"])]
        
        if local_row.empty:
            local_emp = nat_emp
            area_for_lq = None
        else:
            local_emp = float(local_row.iloc[0]["emp"])
            area_for_lq = str(local_row.iloc[0]["area_code"])
            
        local_share_of_nat = (local_emp / nat_emp) if nat_emp > 0 else 0.0
        
        local_total_emp = np.nan
        if area_for_lq is not None:
            local_total_row = oews_local[(oews_local["area_code"].astype(str) == area_for_lq) & (oews_local["occ_code"] == "00-0000")]
            local_total_emp = float(local_total_row.iloc[0]["emp"]) if not local_total_row.empty else np.nan
            if nat_emp > 0 and not np.isnan(local_total_emp) and not np.isnan(nat_total_emp):
                lq_raw = (local_emp / local_total_emp) / (nat_emp / nat_total_emp)
            else:
                lq_raw = 1.0
        else:
            lq_raw = 1.0
            
        # ---- Local openings scaling (robustness-aware) ----
        # EP provides national annual openings; we scale them to a metro/state using OEWS employment structure.
        # Using both (local_occ/nat_occ) and LQ can double-count local concentration, since LQ already encodes
        # relative concentration. We therefore use a standard decomposition as the default:
        #
        #   local_openings ≈ nat_openings * (local_total/nat_total) * LQ
        #
        # and clip LQ to [0.5, 1.5] to avoid extreme leverage in this demo application.
        total_share = (
            (local_total_emp / nat_total_emp)
            if (not np.isnan(local_total_emp) and not np.isnan(nat_total_emp) and nat_total_emp > 0)
            else np.nan
        )
        lq_adj = min(max(lq_raw, lq_clip_low), lq_clip_high)

        openings_share_only = nat_openings * local_share_of_nat
        openings_totalshare_lq_raw = nat_openings * (total_share if not np.isnan(total_share) else 0.0) * lq_raw
        openings_legacy = nat_openings * local_share_of_nat * lq_adj

        openings_used = nat_openings * (total_share if not np.isnan(total_share) else local_share_of_nat) * lq_adj
        if openings_used <= 0:
            openings_used = openings_share_only

        robustness_rows.append(
            {
                "inst": info["inst"],
                "nat_openings": nat_openings,
                "local_share_occ": local_share_of_nat,
                "total_share_all": total_share,
                "lq_raw": lq_raw,
                "openings_share_only": openings_share_only,
                "openings_totalshare_lq": openings_totalshare_lq_raw,
                "openings_legacy": openings_legacy,
                "openings_used": openings_used,
            }
        )
        
        # Calculate Seat Ranges for 10% share
        # Seats = (Openings * Share) / Efficiency
        # Min Seats = (Openings * Share) / Max_Efficiency
        # Max Seats = (Openings * Share) / Min_Efficiency
        
        def get_range(share):
            s_min = (openings_used * share) / efficiency_high
            s_max = (openings_used * share) / efficiency_low
            return s_min, s_max

        s5_min, s5_max = get_range(0.05)
        s10_min, s10_max = get_range(0.10)
        s15_min, s15_max = get_range(0.15)
        
        rows.append({
            "inst": info["inst"],
            "openings": openings_used,
            "s5": (s5_min, s5_max),
            "s10": (s10_min, s10_max),
            "s15": (s15_min, s15_max)
        })
        
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Recommended annual program intake ranges (seats) accounting for efficiency uncertainty. Local openings scale national openings using (local total employment share) $\\times$ (clipped LQ); see Table~\\ref{tab:openings_scaling_robustness}.}")
    lines.append("\\label{tab:program_sizing}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Institution & Est. Openings & 5\\% Share & 10\\% Share & 15\\% Share\\\\")
    lines.append("\\midrule")
    
    for r in rows:
        def fmt_range(t):
            return f"{int(t[0])}--{int(t[1])}"
            
        lines.append(
            f"{_latex_escape(r['inst'])} & {_fmt_int(r['openings'])} & {fmt_range(r['s5'])} & {fmt_range(r['s10'])} & {fmt_range(r['s15'])} \\\\"
        )
        
    lines.append("\\bottomrule")
    lines.append("\\multicolumn{5}{l}{\\footnotesize Ranges reflect uncertainty in program efficiency (completion $\\times$ placement) from 0.39 to 0.72.}\\\\")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    
    _write_text(TABLES_DIR / "program_sizing.tex", "\n".join(lines))

    # Also write the local openings scaling robustness check table.
    if robustness_rows:
        rob = pd.DataFrame(robustness_rows)
        inst_order = {"SDSU": 0, "LATTC": 1, "Academy of Art": 2}
        rob["__ord"] = rob["inst"].map(inst_order).fillna(999)
        rob = rob.sort_values("__ord").drop(columns="__ord")

        rlines: list[str] = []
        rlines.append("\\begin{table}[H]")
        rlines.append("\\centering")
        rlines.append(
            "\\caption{Robustness check for scaling national annual openings (EP) to local markets (OEWS). "
            "We compare (i) openings scaled by local occupation share only, (ii) openings scaled by local total employment share $\\times$ LQ (standard decomposition), "
            "and (iii) a legacy method multiplying occupation share by clipped LQ (shown for comparison; can double-count concentration). "
            "Note that (i) and (ii) are algebraically equivalent when LQ is computed from the same OEWS employment counts; differences arise only from clipping or inconsistent sources. "
            "The seats in Table~\\ref{tab:program_sizing} use method (ii) with clipped LQ.}"
        )
        rlines.append("\\label{tab:openings_scaling_robustness}")
        rlines.append("\\resizebox{\\textwidth}{!}{%")
        rlines.append("\\begin{tabular}{lrrrrrrrr}")
        rlines.append("\\toprule")
        rlines.append(
            "Institution & Nat. openings & Local occ share & Local total share & LQ & Openings (share only) & Openings (total share $\\times$ LQ) & Openings (legacy) & Openings (used)\\\\"
        )
        rlines.append("\\midrule")
        for _, r in rob.iterrows():
            rlines.append(
                " & ".join(
                    [
                        _latex_escape(str(r["inst"])),
                        _fmt_int(r["nat_openings"]),
                        _fmt_float(r["local_share_occ"], 4),
                        _fmt_float(r["total_share_all"], 4) if not (isinstance(r["total_share_all"], float) and math.isnan(r["total_share_all"])) else "",
                        _fmt_float(r["lq_raw"], 3),
                        _fmt_int(r["openings_share_only"]),
                        _fmt_int(r["openings_totalshare_lq"]),
                        _fmt_int(r["openings_legacy"]),
                        _fmt_int(r["openings_used"]),
                    ]
                )
                + "\\\\"
            )
        rlines.append("\\bottomrule")
        rlines.append("\\end{tabular}%")
        rlines.append("}")
        rlines.append("\\end{table}")
        rlines.append("")

        _write_text(TABLES_DIR / "openings_scaling_robustness.tex", "\n".join(rlines))


def build_scenario_bar_figure() -> None:
    df = _load_required_csv(DATA_DIR / "scenario_summary.csv").copy()
    df["career"] = df["career"].astype(str)
    label = {
        "software_engineer": "Software Dev",
        "electrician": "Electrician",
        "writer": "Writer",
    }
    df["label"] = df["career"].map(label).fillna(df["career"])

    df = df.sort_values("label")

    x = range(len(df))
    base = df["emp_2034_No_GenAI_Baseline"].astype(float)
    mod = df["emp_2034_Moderate_Substitution"].astype(float)
    high = df["emp_2034_High_Disruption"].astype(float)

    plt.figure(figsize=(7.0, 3.0))
    w = 0.25
    plt.bar([i - w for i in x], base, width=w, label="Baseline")
    plt.bar([i for i in x], mod, width=w, label="Moderate")
    plt.bar([i + w for i in x], high, width=w, label="High")
    plt.xticks(list(x), df["label"].tolist())
    plt.ylabel("2034 employment (jobs)")
    plt.title("2034 employment by scenario")
    plt.ticklabel_format(axis="y", style="plain")
    plt.legend(frameon=False, ncol=3, fontsize=8)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "scenario_bar.png"
    plt.savefig(out, dpi=200)
    plt.close()


def build_weight_sensitivity_table() -> None:
    """
    Build a sensitivity table checking if the High Disruption scenario flips the sign of
    employment change when NetRisk = a * Sub - b * Def, with a and b varied in
    {0.8, 1.0, 1.2}.
    """
    df = _load_required_csv(DATA_DIR / "scenario_summary.csv").copy()
    
    career_label = {
        "software_engineer": "Software Developers",
        "electrician": "Electricians",
        "writer": "Writers and Authors",
    }
    df["career_label"] = df["career"].map(career_label).fillna(df["career"])
    
    # Stable order
    order = ["software_engineer", "electrician", "writer"]
    df["__ord"] = df["career"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("__ord").drop(columns="__ord")
    
    # Weight combinations to test (focus on extreme weights, excluding base case)
    weights = [(0.8, 0.8), (0.8, 1.0), (0.8, 1.2), 
               (1.0, 0.8), (1.0, 1.2),
               (1.2, 0.8), (1.2, 1.0), (1.2, 1.2)]
    
    # Base case: (1.0, 1.0)
    base_a, base_b = 1.0, 1.0

    # Use the actual High Disruption scenario strength used by the pipeline (calibrated if available).
    s_high = 0.03
    s_source = "default"
    scen_param_path = DATA_DIR / "scenario_parameters.csv"
    if scen_param_path.exists():
        try:
            sp = pd.read_csv(scen_param_path)
            high_row = sp.loc[sp["scenario"] == "High_Disruption"]
            if not high_row.empty:
                s_high = float(high_row.iloc[0].get("s_value", s_high))
                s_source = str(high_row.iloc[0].get("source", s_source))
        except Exception:
            pass
    
    # Compute base case employment changes for reference
    base_changes = {}
    for _, r in df.iterrows():
        sub = float(r["substitution"])
        defense = float(r["defense"])
        m_comp = float(r.get("m_comp", M_MAX_DEFAULT))
        net_risk_base = base_a * sub - base_b * defense
        g_base = float(r["g_baseline"])
        g_adj_base = get_g_adj(g_base, net_risk_base, s_high, m_comp=m_comp)
        emp_2024 = float(r["emp_2024"])
        emp_2034_base = emp_2024 * ((1.0 + g_adj_base) ** 10)
        chg_base = emp_2034_base - emp_2024
        base_changes[r["career"]] = chg_base
    
    # Build results matrix
    results = []
    for _, r in df.iterrows():
        career = r["career"]
        career_name = r["career_label"]
        sub = float(r["substitution"])
        defense = float(r["defense"])
        g_base = float(r["g_baseline"])
        emp_2024 = float(r["emp_2024"])
        m_comp = float(r.get("m_comp", M_MAX_DEFAULT))
        
        row_result = {"career": career_name}
        
        for a, b in weights:
            # Compute NetRisk with modified weights
            net_risk = a * sub - b * defense
            
            # Compute High Disruption scenario
            g_adj = get_g_adj(g_base, net_risk, s_high, m_comp=m_comp)
            emp_2034 = emp_2024 * ((1.0 + g_adj) ** 10)
            chg = emp_2034 - emp_2024
            
            # Check if sign flipped compared to base case
            base_chg = base_changes[career]
            if (base_chg >= 0 and chg < 0) or (base_chg < 0 and chg >= 0):
                status = "Flips"
            else:
                status = "Stable"
            
            row_result[f"({a:.1f},{b:.1f})"] = status
        
        results.append(row_result)
    
    # Build LaTeX table
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    s_note = f"$s={s_high:.4f}$"
    if s_source:
        s_note = s_note + f" ({_latex_escape(s_source)})"
    lines.append(
        "\\caption{Sensitivity of High Disruption scenario ("
        + s_note
        + ") to weight parameters in $\\NetRisk = a \\cdot \\SubScore - b \\cdot \\DefScore$ (with piecewise mapping). "
        + "Shows whether employment change sign flips compared to base case ($a=1.0$, $b=1.0$).}"
    )
    lines.append("\\label{tab:weight_sensitivity}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    # Column specification: one for career, then one for each weight combination
    col_spec = "l" + "c" * len(weights)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header row
    header = ["Career"] + [f"$({a:.1f},{b:.1f})$" for a, b in weights]
    lines.append(" & ".join(header) + "\\\\")
    lines.append("\\midrule")
    
    # Data rows
    for result in results:
        career = result["career"]
        vals = [result[f"({a:.1f},{b:.1f})"] for a, b in weights]
        lines.append(" & ".join([_latex_escape(career)] + vals) + "\\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    
    _write_text(TABLES_DIR / "weight_sensitivity.tex", "\n".join(lines))


def build_policy_regimes_table() -> None:
    """Build table showing policy shifts under alternative objective function weights."""
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Recommended policy shifts under alternative objective function weights.}")
    lines.append("\\label{tab:policy_regimes}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lp{2.2cm}p{2.2cm}p{1.8cm}}")
    lines.append("\\toprule")
    lines.append("Regime (Weight Dominance) & Policy Stance & Assessment Changes & Tool Restrictions\\\\")
    lines.append("\\midrule")
    
    rows = [
        ("Balanced ($w_E \\approx w_I$)", "Permit with disclosure", "Verify outputs, oral defense of code/text", "Standard commercial models"),
        ("Integrity-First ($w_I \\gg w_E$)", "Strict audit \\& provenance", "In-person blue book exams; full edit history required", "Local-only or logged enterprise instances"),
        ("Sustainability-First ($w_S \\gg w_E$)", "Minimal compute", "Focus on logic/structure; limit GenAI for drafting", "Small SLMs only; quota on token usage"),
    ]
    
    for r in rows:
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\")
        
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")
    
    _write_text(TABLES_DIR / "policy_regimes.tex", "\n".join(lines))


def build_netrisk_hist_figure() -> None:
    """Generate histogram of NetRisk scores across all occupations."""
    df = _load_required_csv(DATA_DIR / "mechanism_risk_scored.csv")
    net_risk = df["net_risk"].dropna()

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(net_risk, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("NetRisk")
    plt.ylabel("Frequency")
    plt.title("Distribution of NetRisk Scores Across Occupations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "netrisk_hist.png"
    plt.savefig(out, dpi=200)
    plt.close()


def build_netrisk_summary_table() -> None:
    """Calculate summary statistics for NetRisk scores."""
    df = _load_required_csv(DATA_DIR / "mechanism_risk_scored.csv")
    net_risk = df["net_risk"].dropna()

    mean_val = float(net_risk.mean())
    std_val = float(net_risk.std())
    p10 = float(net_risk.quantile(0.10))
    p50 = float(net_risk.quantile(0.50))
    p90 = float(net_risk.quantile(0.90))

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Summary statistics for NetRisk scores (from \\texttt{data/mechanism\\_risk\\_scored.csv}).}")
    lines.append("\\label{tab:netrisk_summary}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("Statistic & Value\\\\")
    lines.append("\\midrule")
    lines.append(f"Mean & {_fmt_float(mean_val, 3)}\\\\")
    lines.append(f"Std. Dev. & {_fmt_float(std_val, 3)}\\\\")
    lines.append(f"10th percentile & {_fmt_float(p10, 3)}\\\\")
    lines.append(f"50th percentile (median) & {_fmt_float(p50, 3)}\\\\")
    lines.append(f"90th percentile & {_fmt_float(p90, 3)}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "netrisk_summary.tex", "\n".join(lines))


def build_netrisk_interpretation() -> None:
    """Create interpretation file with extreme occupations to help interpret the distribution."""
    mech = _load_required_csv(DATA_DIR / "mechanism_risk_scored.csv")
    occ = _load_required_csv(DATA_DIR / "occ_key.csv")
    occ = occ.rename(columns={"soc_code": "occ_code", "soc_title": "occ_title"})
    occ = occ[["occ_code", "occ_title"]].drop_duplicates()

    mech = mech.merge(occ, on="occ_code", how="left")

    # Get top 3 highest (most exposed) and bottom 3 lowest (most sheltered)
    top_exposed = mech.sort_values("net_risk", ascending=False).head(3)
    top_sheltered = mech.sort_values("net_risk", ascending=True).head(3)

    lines: list[str] = []
    lines.append("\\textbf{Extreme Positive Tail (Most Exposed):}")
    lines.append("")
    lines.append("\\begin{itemize}")
    for _, r in top_exposed.iterrows():
        occ_title = r.get("occ_title", r.get("occ_code", "Unknown"))
        net_risk_val = r.get("net_risk", 0)
        lines.append(f"\\item {_latex_escape(occ_title)} ({r.get('occ_code', 'N/A')}): NetRisk = {_fmt_float(net_risk_val, 3)}")
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("\\textbf{Extreme Negative Tail (Most Sheltered):}")
    lines.append("")
    lines.append("\\begin{itemize}")
    for _, r in top_sheltered.iterrows():
        occ_title = r.get("occ_title", r.get("occ_code", "Unknown"))
        net_risk_val = r.get("net_risk", 0)
        lines.append(f"\\item {_latex_escape(occ_title)} ({r.get('occ_code', 'N/A')}): NetRisk = {_fmt_float(net_risk_val, 3)}")
    lines.append("\\end{itemize}")

    _write_text(TABLES_DIR / "netrisk_interpretation.tex", "\n".join(lines))


def build_onet_elements_appendix() -> None:
    """Build appendix table listing O*NET elements used for each dimension."""
    manifest_path = DATA_DIR / "mechanism_element_map.csv"
    if not manifest_path.exists():
        print(f"Warning: {manifest_path} not found. Run build_mechanism_layer_expanded.py first.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # Sort by dimension, then domain, then ID
    df = df.sort_values(["dimension", "domain_file", "element_id"])
    
    lines: list[str] = []
    lines.append("\\begin{longtable}{llp{2cm}p{6cm}}")
    lines.append("\\caption{O*NET Elements mapped to Mechanism Dimensions (using Importance scale).}\\\\")
    lines.append("\\label{tab:onet_elements_appendix}\\\\")
    lines.append("\\toprule")
    lines.append("Dimension & Domain & Element ID & Element Name\\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\caption[]{O*NET Elements mapped to Mechanism Dimensions (continued).}\\\\")
    lines.append("\\toprule")
    lines.append("Dimension & Domain & Element ID & Element Name\\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")
    
    current_dim = ""
    for _, r in df.iterrows():
        dim = r["dimension"].replace("_", " ").title()
        domain = r["domain_file"]
        e_id = r["element_id"]
        e_name = r["element_name"]
        
        # Only show dimension name on change
        dim_cell = _latex_escape(dim) if dim != current_dim else ""
        current_dim = dim
        
        lines.append(
            f"{dim_cell} & {_latex_escape(domain)} & {_latex_escape(e_id)} & {_latex_escape(e_name)} \\\\"
        )
        
    lines.append("\\end{longtable}")
    lines.append("")
    
    _write_text(TABLES_DIR / "onet_elements_appendix.tex", "\n".join(lines))


def build_calibration_summary_table() -> None:
    """Build table summarizing calibrated weights and fit metrics."""
    results_path = DATA_DIR / "calibration_results.csv"
    weights_path = DATA_DIR / "calibration_weights.csv"
    if not results_path.exists() or not weights_path.exists():
        return

    results = pd.read_csv(results_path)
    weights = pd.read_csv(weights_path)

    dim_label = {
        "writing_intensity": "Writing",
        "tool_technology": "Tool/Tech",
        "physical_manual": "Physical",
        "social_perceptiveness": "Social",
        "creativity_originality": "Creativity",
    }

    def _metric(name: str) -> str:
        row = results[results["metric"] == name]
        if row.empty:
            return ""
        return _fmt_float(row.iloc[0]["value"], 3)

    n_samples = results[results["metric"] == "n_samples"]["value"]
    n_samples = int(n_samples.iloc[0]) if not n_samples.empty else 0

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Calibration of mechanism weights to external AI applicability scores (Tomlinson et al., 2025).}")
    lines.append("\\label{tab:calibration_summary}")
    lines.append("\\begin{tabular}{lcr}")
    lines.append("\\toprule")
    lines.append("Dimension & Sign & Weight\\\\")
    lines.append("\\midrule")
    for _, r in weights.iterrows():
        dim = dim_label.get(r["feature"], r["feature"])
        sign = "+" if float(r["sign_convention"]) > 0 else "-"
        weight = _fmt_float(r["weight_raw"], 3)
        lines.append(f"{_latex_escape(dim)} & {sign} & {weight}\\\\")
    lines.append("\\midrule")
    lines.append(
        f"\\multicolumn{{3}}{{l}}{{\\footnotesize Fit: $R^2$={_metric('r2')}, MAE={_metric('mae')}, RMSE={_metric('rmse')}, $n$={n_samples}}}\\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "calibration_summary.tex", "\n".join(lines))


def build_calibration_scatter_figure() -> None:
    """Scatter plot of observed vs predicted AI applicability."""
    fit_path = DATA_DIR / "calibration_fit.csv"
    if not fit_path.exists():
        return
    df = pd.read_csv(fit_path)
    if df.empty:
        return

    x = df["ai_applicability"].astype(float)
    y = df["ai_applicability_pred"].astype(float)

    plt.figure(figsize=(4.2, 4.0))
    plt.scatter(x, y, s=10, alpha=0.5, edgecolor="none")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("Observed AI applicability")
    plt.ylabel("Predicted AI applicability")
    plt.title("Calibration fit")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "calibration_scatter.png"
    plt.savefig(out, dpi=200)
    plt.close()


def build_uncertainty_summary_table() -> None:
    """Build a table of uncertainty intervals for scenario employment."""
    path = DATA_DIR / "uncertainty_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    career_label = {
        "software_engineer": "Software Developers",
        "electrician": "Electricians",
        "writer": "Writers and Authors",
    }
    df["career_label"] = df["career"].map(career_label).fillna(df["career"])

    order = ["software_engineer", "electrician", "writer"]
    df["__ord"] = df["career"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values(["__ord", "scenario"]).drop(columns="__ord")

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Uncertainty intervals for 2034 employment under Moderate and High disruption (Monte Carlo).}")
    lines.append("\\label{tab:uncertainty_summary}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append("Career & Scenario & $E_{2034}$ P5 & P50 & P95\\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        scen = str(r["scenario"]).replace("_", " ")
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["career_label"]),
                    _latex_escape(scen),
                    _fmt_int(r["emp_p05"]),
                    _fmt_int(r["emp_p50"]),
                    _fmt_int(r["emp_p95"]),
                ]
            )
            + "\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "uncertainty_summary.tex", "\n".join(lines))


def build_policy_decision_table() -> None:
    """Build a table of recommended policy regimes by weight regime."""
    path = DATA_DIR / "policy_decision_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Recommended policy regime by institution under alternative objective weights.}")
    lines.append("\\label{tab:policy_decision}")
    lines.append("\\begin{tabular}{lll}")
    lines.append("\\toprule")
    lines.append("Institution & Weight Regime & Recommended Policy\\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["institution"]),
                    _latex_escape(r["weight_regime"].replace("_", " ")),
                    _latex_escape(r["policy_regime"].replace("_", " ")),
                ]
            )
            + "\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "policy_decision.tex", "\n".join(lines))


def build_policy_sensitivity_table() -> None:
    """
    Summarize robustness of policy recommendations.
    Reads data/policy_sensitivity.csv. 
    If columns match the new simplified format (institution, weight_regime, baseline_policy, robustness),
    it just formats it.
    """
    path = DATA_DIR / "policy_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    # Check format
    if "robustness" in df.columns and "recommended_policy" not in df.columns:
        # Pre-summarized format (from new build_policy_model.py)
        out = df.sort_values(["institution", "weight_regime"])
    else:
        # Legacy format (raw sensitivity rows) - keep fallback logic just in case
        grp_cols = ["institution", "weight_regime"]
        rows = []
        for (inst, w), g in df.groupby(grp_cols):
            baseline = str(g["baseline_policy"].dropna().iloc[0]) if g["baseline_policy"].notna().any() else ""
            n = int(len(g))
            # matches_baseline column might not exist if we used new logic, but if we are here we assume legacy columns
            # actually if we are here, we probably have 'recommended_policy'
            n_match = 0
            policies = []
            if "recommended_policy" in g.columns:
                policies = sorted(set(str(x) for x in g["recommended_policy"].dropna().tolist()))
                # If baseline column exists, use it
                matches = g["recommended_policy"] == g["baseline_policy"]
                n_match = int(matches.sum())
            
            if n > 0 and n_match == n:
                robustness = f"Stable ({n_match}/{n})"
            else:
                policy_list = ", ".join(p.replace("_", " ") for p in policies) if policies else ""
                robustness = f"Flips ({n_match}/{n}); seen: {policy_list}"

            rows.append(
                {
                    "institution": str(inst),
                    "weight_regime": str(w).replace("_", " "),
                    "baseline_policy": baseline.replace("_", " "),
                    "robustness": robustness,
                }
            )
        out = pd.DataFrame(rows).sort_values(["institution", "weight_regime"])

    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Robustness of policy recommendations to modest perturbations of institution capacity parameters (audit and sustainability varied by $\\pm 0.1$; 9 combinations per cell).}"
    )
    lines.append("\\label{tab:policy_sensitivity}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llll}")
    lines.append("\\toprule")
    lines.append("Institution & Weight Regime & Baseline Policy & Robustness\\\\")
    lines.append("\\midrule")

    for _, r in out.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(r["institution"]),
                    _latex_escape(r["weight_regime"].replace("_", " ")),
                    _latex_escape(r["baseline_policy"].replace("_", " ")),
                    _latex_escape(r["robustness"]),
                ]
            )
            + "\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    _write_text(TABLES_DIR / "policy_sensitivity.tex", "\n".join(lines))


def build_policy_tradeoff_figure() -> None:
    """Bar chart of policy scores under Balanced weights."""
    path = DATA_DIR / "policy_decision_scores.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df[df["weight_regime"] == "Balanced"].copy()
    if df.empty:
        return

    institutions = df["institution"].unique().tolist()
    policies = ["Ban", "Allow_with_Audit", "Require"]
    x = np.arange(len(institutions))
    w = 0.25

    plt.figure(figsize=(7.0, 3.2))
    for i, policy in enumerate(policies):
        vals = [df[(df["institution"] == inst) & (df["policy_regime"] == policy)]["score"].mean() for inst in institutions]
        plt.bar(x + (i - 1) * w, vals, width=w, label=policy.replace("_", " "))

    plt.xticks(x, institutions)
    plt.ylabel("Objective score (Balanced)")
    plt.title("Policy trade-offs by institution")
    plt.legend(frameon=False, ncol=3, fontsize=8)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "policy_tradeoff.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    build_scenario_summary_table()
    build_summary_headline_fragment()
    build_scenario_parameter_table()
    build_sensitivity_grid_table()
    build_complementarity_sensitivity_table()
    build_comp_cap_components_table()
    build_top_exposed_sheltered_table()
    build_openings_summary_table()
    build_mechanism_coverage_table()
    build_local_context_table()
    build_program_sizing_table()
    build_weight_sensitivity_table()
    build_policy_regimes_table()
    build_scenario_bar_figure()
    build_netrisk_hist_figure()
    build_netrisk_summary_table()
    build_netrisk_interpretation()
    build_onet_elements_appendix()
    build_calibration_summary_table()
    build_calibration_scatter_figure()
    build_netrisk_index_compare_table()
    build_netrisk_index_disagreement_table()
    build_bundle_contents_fragment()
    build_uncertainty_summary_table()
    build_policy_decision_table()
    build_policy_sensitivity_table()
    build_policy_tradeoff_figure()
    print("Wrote report artifacts to", REPORTS_DIR)


if __name__ == "__main__":
    main()

