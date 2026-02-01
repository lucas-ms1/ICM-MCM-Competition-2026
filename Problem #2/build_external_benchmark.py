"""
External benchmark reality check: compare NetRisk to Felten/Raj/Seamans AIOE.

Outputs:
  - data/netrisk_vs_aioe.csv
  - reports/tables/external_benchmark.tex
  - reports/tables/external_benchmark_disagree.tex

Data source (public):
  - Felten, Raj, Seamans (2021) AIOE dataset, GitHub repo: https://github.com/AIOE-Data/AIOE
  - Default download target: AIOE_DataAppendix.xlsx (Data Appendix A: AIOE by occupation)
    raw file: https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx

Design goal: provide a judge-friendly independent cross-check (correlation + examples),
without changing the scenario pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
EXTERNAL_DIR = DATA_DIR / "external"


AIOE_REPO_RAW_XLSX = "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx"
AIOE_DEFAULT_XLSX = EXTERNAL_DIR / "AIOE_DataAppendix.xlsx"


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


def _fmt_float(x: float | int, nd: int = 3) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return ""
    if math.isnan(xf) or math.isinf(xf):
        return ""
    return f"{xf:.{nd}f}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _soc6_to_soc7(soc6: str) -> str | None:
    """
    Convert 6-digit SOC like '111011' to 7-char SOC stem like '11-1011'.
    """
    if soc6 is None:
        return None
    s = re.sub(r"\D", "", str(soc6))
    if len(s) != 6:
        return None
    return f"{s[:2]}-{s[2:]}"


def _download_if_missing(path: Path) -> None:
    if path.exists():
        return
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AIOE benchmark file to: {path}")
    urllib.request.urlretrieve(AIOE_REPO_RAW_XLSX, path)  # nosec - public dataset


@dataclass(frozen=True)
class AIOEExtract:
    sheet: str
    soc_col: str
    score_col: str


def _find_aioe_sheet(xls: dict[str, pd.DataFrame]) -> AIOEExtract:
    """
    Best-effort heuristic to find the sheet/columns containing:
      - 6-digit SOC occupation code
      - AIOE score
    """
    # Look for obvious candidates first.
    candidates: list[AIOEExtract] = []
    for sheet, df in xls.items():
        if df is None or df.empty:
            continue

        cols = [str(c) for c in df.columns]
        cols_l = [c.lower() for c in cols]

        soc_like = [cols[i] for i, c in enumerate(cols_l) if ("soc" in c and "code" in c) or c.strip() in {"soc", "soc6", "soc_6", "soc6digit"}]
        score_like = [cols[i] for i, c in enumerate(cols_l) if "aioe" in c and "industry" not in c and "geog" not in c]

        for sc in soc_like:
            for yc in score_like:
                candidates.append(AIOEExtract(sheet=sheet, soc_col=sc, score_col=yc))

    if candidates:
        return candidates[0]

    # Fallback: scan columns by content.
    for sheet, df in xls.items():
        if df is None or df.empty:
            continue

        cols = [str(c) for c in df.columns]
        for soc_col in cols:
            s = df[soc_col]
            # SOC6 tends to be numeric; allow strings too.
            soc6 = s.astype(str).str.replace(r"\D", "", regex=True)
            soc6_ok = soc6.str.len().eq(6).mean()
            if soc6_ok < 0.5:
                continue

            # Now find a numeric score column with variation.
            for score_col in cols:
                if score_col == soc_col:
                    continue
                y = pd.to_numeric(df[score_col], errors="coerce")
                if y.notna().mean() < 0.5:
                    continue
                if float(y.std(skipna=True)) <= 0:
                    continue
                # Prefer columns containing 'aioe' in the name.
                if "aioe" in score_col.lower():
                    return AIOEExtract(sheet=sheet, soc_col=soc_col, score_col=score_col)
                # Keep as a fallback candidate.
                candidates.append(AIOEExtract(sheet=sheet, soc_col=soc_col, score_col=score_col))

    if candidates:
        return candidates[0]

    raise ValueError("Could not locate an AIOE occupation-level sheet/columns in the Excel file.")


def _load_occ_titles() -> pd.DataFrame:
    occ_key = DATA_DIR / "occ_key.csv"
    if not occ_key.exists():
        return pd.DataFrame(columns=["occ_code", "occ_title"])
    occ = pd.read_csv(occ_key)
    if "soc_code" in occ.columns and "soc_title" in occ.columns:
        occ = occ.rename(columns={"soc_code": "occ_code", "soc_title": "occ_title"})
    occ["occ_code"] = occ["occ_code"].astype(str).str.strip()
    occ["occ_title"] = occ["occ_title"].astype(str)
    return occ[["occ_code", "occ_title"]].drop_duplicates(subset=["occ_code"])


def build_external_benchmark() -> None:
    mech_path = DATA_DIR / "mechanism_risk_scored.csv"
    if not mech_path.exists():
        raise FileNotFoundError(f"Missing required file: {mech_path}. Run run_scenarios.py first.")

    # Ensure the benchmark file exists (download if possible).
    _download_if_missing(AIOE_DEFAULT_XLSX)

    # Load all sheets.
    xls = pd.read_excel(AIOE_DEFAULT_XLSX, sheet_name=None)
    extract = _find_aioe_sheet(xls)

    raw = xls[extract.sheet].copy()
    raw_soc7 = raw[extract.soc_col].astype(str).map(_soc6_to_soc7)
    raw_score = pd.to_numeric(raw[extract.score_col], errors="coerce")

    aioe = pd.DataFrame({"occ_code": raw_soc7, "aioe": raw_score}).dropna(subset=["occ_code", "aioe"])
    aioe["occ_code"] = aioe["occ_code"].astype(str).str.strip()
    aioe = aioe.drop_duplicates(subset=["occ_code"], keep="first")

    mech = pd.read_csv(mech_path)
    mech["occ_code"] = mech["occ_code"].astype(str).str.strip()

    # Join
    joined = mech.merge(aioe, on="occ_code", how="inner")
    joined = joined.copy()

    # Add titles for disagreement examples
    occ = _load_occ_titles()
    if not occ.empty:
        joined = joined.merge(occ, on="occ_code", how="left")
    if "occ_title" not in joined.columns:
        joined["occ_title"] = ""

    # Decide which NetRisk columns are available
    cols = {
        "uncal": "net_risk_uncalibrated" if "net_risk_uncalibrated" in joined.columns else None,
        "cal": "net_risk_calibrated" if "net_risk_calibrated" in joined.columns else None,
        "used": "net_risk" if "net_risk" in joined.columns else None,
    }

    def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) == 0:
            return float("nan")
        return float(x[m].corr(y[m], method=method))

    def _summary_for(col: str) -> dict[str, float]:
        x = joined[col]
        y = joined["aioe"]
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        pearson = float("nan")
        spearman = float("nan")
        if n:
            pearson = float(x[m].corr(y[m], method="pearson"))
            spearman = float(x[m].corr(y[m], method="spearman"))
        return {"n": n, "pearson": pearson, "spearman": spearman}

    summaries: dict[str, dict[str, float]] = {}
    for key, col in cols.items():
        if col is None:
            continue
        if col not in joined.columns:
            continue
        summaries[key] = _summary_for(col)

    # Write joined audit CSV
    out_csv = DATA_DIR / "netrisk_vs_aioe.csv"
    joined.to_csv(out_csv, index=False)

    # LaTeX summary table
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{External reality check: correlation between our NetRisk indices and the Felten--Raj--Seamans AI Occupational Exposure (AIOE) score (from \\texttt{data/netrisk\\_vs\\_aioe.csv}).}"
    )
    lines.append("\\label{tab:external_benchmark}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Index & Matched occupations ($n$) & Pearson $r$ & Spearman $\\rho$\\\\")
    lines.append("\\midrule")

    row_labels = {
        "uncal": "NetRisk (uncalibrated mechanism)",
        "cal": "NetRisk (calibrated predictive)",
        "used": "NetRisk (pipeline-used)",
    }
    for key in ["uncal", "cal", "used"]:
        if key not in summaries:
            continue
        s = summaries[key]
        lines.append(
            " & ".join(
                [
                    _latex_escape(row_labels.get(key, key)),
                    f"{int(s['n']):,}",
                    _fmt_float(s["pearson"], 3),
                    _fmt_float(s["spearman"], 3),
                ]
            )
            + "\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    _write_text(TABLES_DIR / "external_benchmark.tex", "\n".join(lines))

    # Disagreement examples (rank differences)
    # Use calibrated if available, otherwise pipeline-used, otherwise uncal.
    pick_col = cols["cal"] or cols["used"] or cols["uncal"]
    if pick_col and pick_col in joined.columns:
        tmp = joined.copy()
        tmp["netrisk_for_rank"] = pd.to_numeric(tmp[pick_col], errors="coerce")
        tmp["aioe_for_rank"] = pd.to_numeric(tmp["aioe"], errors="coerce")
        tmp = tmp.dropna(subset=["netrisk_for_rank", "aioe_for_rank"]).copy()
        if not tmp.empty:
            tmp["rank_netrisk"] = tmp["netrisk_for_rank"].rank(ascending=False, method="average")
            tmp["rank_aioe"] = tmp["aioe_for_rank"].rank(ascending=False, method="average")
            tmp["rank_gap"] = (tmp["rank_netrisk"] - tmp["rank_aioe"]).abs()
            show = tmp.sort_values("rank_gap", ascending=False).head(12)

            dlines: list[str] = []
            dlines.append("\\begin{table}[H]")
            dlines.append("\\centering")
            dlines.append(
                "\\caption{Examples of disagreement between NetRisk ranking and AIOE ranking (largest absolute rank gaps; NetRisk uses "
                + _latex_escape(pick_col)
                + ").}"
            )
            dlines.append("\\label{tab:external_benchmark_disagree}")
            dlines.append("\\resizebox{\\textwidth}{!}{%")
            dlines.append("\\begin{tabular}{llrrr}")
            dlines.append("\\toprule")
            dlines.append("SOC & Title & NetRisk & AIOE & $|\\Delta \\mathrm{rank}|$\\\\")
            dlines.append("\\midrule")
            for _, r in show.iterrows():
                title = str(r.get("occ_title", "")).strip()
                if not title or title.lower() == "nan":
                    title = "Unknown"
                dlines.append(
                    " & ".join(
                        [
                            _latex_escape(str(r["occ_code"])),
                            _latex_escape(title),
                            _fmt_float(r["netrisk_for_rank"], 3),
                            _fmt_float(r["aioe_for_rank"], 3),
                            _fmt_float(r["rank_gap"], 0),
                        ]
                    )
                    + "\\\\"
                )
            dlines.append("\\bottomrule")
            dlines.append("\\end{tabular}%")
            dlines.append("}")
            dlines.append("\\end{table}")
            dlines.append("")
            _write_text(TABLES_DIR / "external_benchmark_disagree.tex", "\n".join(dlines))

    print("Wrote:")
    print(" -", out_csv)
    print(" -", TABLES_DIR / "external_benchmark.tex")
    print(" -", TABLES_DIR / "external_benchmark_disagree.tex (if applicable)")


if __name__ == "__main__":
    build_external_benchmark()

