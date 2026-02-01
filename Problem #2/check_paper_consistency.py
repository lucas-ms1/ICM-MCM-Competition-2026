"""
Consistency checks for Problem F paper artifacts.

This script checks:
  1) All \\input'ed table fragments referenced in reports/main.tex exist.
  2) All \\includegraphics files referenced in reports/main.tex exist.
  3) Summary Sheet headline numeric claims match data/scenario_summary.csv.
  4) reports/tables/scenario_summary.tex matches data/scenario_summary.csv (key fields).
  5) Program sizing table matches openings_used and efficiency/share formulas.
  6) External benchmark table correlations match recomputation from data/netrisk_vs_aioe.csv.

Outputs:
  - data/validation/paper_consistency_check.txt

Non-goal: pixel-perfect figure validation (we validate their data sources and existence instead).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_DIR = ROOT / "data"
VALID_DIR = DATA_DIR / "validation"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _parse_int_commas(s: str) -> int:
    s2 = re.sub(r"[^0-9\-]", "", s)
    return int(s2)


def _parse_float(s: str) -> float:
    return float(s.strip())


def _almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, float) and (math.isnan(a) or math.isinf(a)):
        return False
    if isinstance(b, float) and (math.isnan(b) or math.isinf(b)):
        return False
    return abs(float(a) - float(b)) <= tol


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str = ""


def check_inputs_and_figures(main_tex: Path) -> list[CheckResult]:
    text = _read_text(main_tex)
    results: list[CheckResult] = []

    # \input{tables/foo.tex} inside \IfFileExists {...}{\input{...}}{}
    input_paths = re.findall(r"\\input\{([^}]+)\}", text)
    missing_inputs = []
    for rel in input_paths:
        p = (main_tex.parent / rel).resolve()
        if not p.exists():
            missing_inputs.append(str(p))
    results.append(
        CheckResult(
            name="table_inputs_exist",
            ok=(len(missing_inputs) == 0),
            details=("Missing:\n" + "\n".join(missing_inputs)) if missing_inputs else f"Found {len(input_paths)} inputs; all exist.",
        )
    )

    # \includegraphics{figures/foo.png}
    fig_paths = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", text) + re.findall(
        r"\\includegraphics\{([^}]+)\}", text
    )
    missing_figs = []
    for rel in fig_paths:
        p = (main_tex.parent / rel).resolve()
        if not p.exists():
            missing_figs.append(str(p))
    results.append(
        CheckResult(
            name="figures_exist",
            ok=(len(missing_figs) == 0),
            details=("Missing:\n" + "\n".join(missing_figs)) if missing_figs else f"Found {len(fig_paths)} figures; all exist.",
        )
    )
    return results


def check_summary_sheet_headlines(main_tex: Path, scenario_csv: Path) -> list[CheckResult]:
    """
    Validates that the Summary Sheet headline findings for baseline/high/ramp high
    match data/scenario_summary.csv exactly (as integers).
    """
    text = _read_text(main_tex)
    df = pd.read_csv(scenario_csv)
    df = df.set_index("career")

    # Pull the Summary Sheet line (we look for the exact anchor)
    m = re.search(r"Headline findings \(national 2034; SOC bundles\)\.(.+?)\\par", text, flags=re.DOTALL)
    if not m:
        return [CheckResult("summary_sheet_headline_parse", False, "Could not locate the Summary Sheet headline findings block.")]

    block = m.group(1).strip()
    # If the block is an \input of the headline fragment, expand it so we validate the fragment content
    frag_path = main_tex.parent / "tables" / "summary_headline_fragment.tex"
    if frag_path.exists() and ("summary_headline_fragment.tex" in block and "\\input" in block):
        block = _read_text(frag_path)

    # The headline findings block has two distinct parts:
    #  - immediate High vs baseline (each career appears once)
    #  - Ramp adoption reduces... (each career appears again)
    # We split on the Ramp sentence so we don't cross-match the wrong occupation.
    parts = block.split("Ramp adoption reduces", 1)
    if len(parts) != 2:
        return [CheckResult("summary_sheet_headline_split", False, "Could not split headline block on 'Ramp adoption reduces'.")]
    immediate_part = parts[0]
    ramp_part = parts[1]

    def _grab(label: str) -> tuple[int, int, int]:
        # baseline, high, ramp_high
        pat_immediate = (
            re.escape(label)
            + r"\s*\\\((?P<base>[0-9\{\},]+)\s*\\rightarrow\s*(?P<high>[0-9\{\},]+)\\\)"
        )
        mi = re.search(pat_immediate, immediate_part, flags=re.DOTALL)
        if not mi:
            raise ValueError(f"Could not parse immediate High numbers for {label}")

        pat_ramp = (
            re.escape(label)
            + r"\s*\\\((?P<base2>[0-9\{\},]+)\s*\\rightarrow\s*(?P<rh>[0-9\{\},]+)\\\)"
        )
        mr = re.search(pat_ramp, ramp_part, flags=re.DOTALL)
        if not mr:
            raise ValueError(f"Could not parse Ramp High numbers for {label}")

        base = _parse_int_commas(mi.group("base"))
        high = _parse_int_commas(mi.group("high"))
        base2 = _parse_int_commas(mr.group("base2"))
        rh = _parse_int_commas(mr.group("rh"))
        if base != base2:
            raise ValueError(f"Baseline mismatch within Summary Sheet for {label}: {base} vs {base2}")
        return base, high, rh

    def _expect(career_key: str) -> tuple[int, int, int]:
        r = df.loc[career_key]
        base = int(round(float(r["emp_2034_No_GenAI_Baseline"])))
        high = int(round(float(r["emp_2034_High_Disruption"])))
        rh = int(round(float(r["emp_2034_Ramp_High"])))
        return base, high, rh

    checks = []
    mapping = [
        ("Software Developers", "software_engineer", "Software Developers"),
        ("Electricians", "electrician", "Electricians"),
        ("Writers and Authors", "writer", "Writers and Authors"),
    ]

    for label, key, short in mapping:
        try:
            got = _grab(label)
            exp = _expect(key)
            ok = got == exp
            checks.append(
                CheckResult(
                    name=f"summary_sheet_{short.replace(' ', '_').lower()}",
                    ok=ok,
                    details=f"got={got}, expected={exp}",
                )
            )
        except Exception as e:
            checks.append(CheckResult(name=f"summary_sheet_{short.replace(' ', '_').lower()}", ok=False, details=str(e)))

    return checks


def check_scenario_table_matches_csv(scenario_csv: Path, scenario_tex: Path) -> list[CheckResult]:
    df = pd.read_csv(scenario_csv).copy()
    df = df.set_index("career")
    tex = _read_text(scenario_tex)

    # Parse the 3 data rows from the latex table.
    # Expected row format (has range): Label & netrisk & [min,max] & emp_2024 & emp_2034... \\
    rows = []
    for line in tex.splitlines():
        if line.strip().endswith(r"\\") and "&" in line and "Career" not in line:
            rows.append(line.strip().rstrip(r"\\").strip())

    # There are also lines like \midrule or \bottomrule; filter those out.
    rows = [r for r in rows if not r.startswith("\\") and not r.startswith("%")]

    if len(rows) != 3:
        return [CheckResult("scenario_table_rowcount", False, f"Expected 3 data rows, found {len(rows)}")]

    # Map from displayed label to career key
    label_to_key = {
        "Software Developers (STEM)": "software_engineer",
        "Electricians (Trade)": "electrician",
        "Writers and Authors (Arts)": "writer",
    }

    def _cell_int(s: str) -> int:
        return _parse_int_commas(s)

    def _cell_float(s: str) -> float:
        return float(s.strip())

    checks: list[CheckResult] = []
    for r in rows:
        cells = [c.strip() for c in r.split("&")]
        label = cells[0]
        key = label_to_key.get(label)
        if key is None:
            checks.append(CheckResult("scenario_table_label", False, f"Unexpected label: {label}"))
            continue
        row = df.loc[key]

        # cells: label, netrisk, range, emp24, base, mod, ramp_mod, high, ramp_high
        got_netrisk = _cell_float(cells[1])
        exp_netrisk = float(row["net_risk"])
        ok_netrisk = abs(got_netrisk - exp_netrisk) <= 1e-3
        checks.append(CheckResult(f"scenario_table_netrisk_{key}", ok_netrisk, f"got={got_netrisk}, expected={exp_netrisk}"))

        got_emp24 = _cell_int(cells[3])
        exp_emp24 = int(round(float(row["emp_2024"])))
        checks.append(CheckResult(f"scenario_table_emp2024_{key}", got_emp24 == exp_emp24, f"got={got_emp24}, expected={exp_emp24}"))

        colmap = {
            4: "emp_2034_No_GenAI_Baseline",
            5: "emp_2034_Moderate_Substitution",
            6: "emp_2034_Ramp_Moderate",
            7: "emp_2034_High_Disruption",
            8: "emp_2034_Ramp_High",
        }
        for idx, col in colmap.items():
            got = _cell_int(cells[idx])
            exp = int(round(float(row[col])))
            checks.append(CheckResult(f"scenario_table_{col}_{key}", got == exp, f"got={got}, expected={exp}"))

    return checks


def check_program_sizing(program_tex: Path, openings_robust_tex: Path) -> list[CheckResult]:
    """
    Validates:
      - program_sizing "Est. Openings" equals openings_used from openings_scaling_robustness table.
      - seat ranges match openings_used, shares in {0.05,0.10,0.15}, efficiency in [0.39,0.72]
        with integer truncation.
    """
    checks: list[CheckResult] = []

    # Parse openings_used from robustness table
    rob = _read_text(openings_robust_tex).splitlines()
    rob_rows_all = [l.strip() for l in rob if l.strip().endswith(r"\\") and "&" in l and not l.strip().startswith("\\")]
    # Keep only the known institution rows (avoid the header line "Institution & ...")
    known_insts = {"SDSU", "LATTC", "Academy of Art"}
    rob_rows = []
    for l in rob_rows_all:
        first = l.split("&", 1)[0].strip()
        if first in known_insts:
            rob_rows.append(l)
    if len(rob_rows) < 3:
        return [
            CheckResult(
                "openings_robust_parse",
                False,
                f"Expected >=3 institution rows, got {len(rob_rows)} (parsed {len(rob_rows_all)} candidate rows)",
            )
        ]

    inst_to_openings_used: dict[str, int] = {}
    for rr in rob_rows:
        cells = [c.strip() for c in rr.rstrip(r"\\").split("&")]
        inst = cells[0]
        openings_used = _parse_int_commas(cells[-1])
        inst_to_openings_used[inst] = openings_used

    # Parse program sizing table
    txt = _read_text(program_tex).splitlines()
    rows_all = [l.strip() for l in txt if l.strip().endswith(r"\\") and "&" in l and not l.strip().startswith("\\")]
    rows = []
    for l in rows_all:
        first = l.split("&", 1)[0].strip()
        if first in known_insts:
            rows.append(l)
    # Expect 3 institutions
    if len(rows) < 3:
        return [
            CheckResult(
                "program_sizing_rowcount",
                False,
                f"Expected >=3 institution rows, got {len(rows)} (parsed {len(rows_all)} candidate rows)",
            )
        ]

    eff_low = 0.39
    eff_high = 0.72
    shares = {"5\\% Share": 0.05, "10\\% Share": 0.10, "15\\% Share": 0.15}

    def _parse_range(s: str) -> tuple[int, int]:
        # e.g. "118--220"
        mm = re.search(r"(\d+)\s*--\s*(\d+)", s)
        if not mm:
            raise ValueError(f"Bad range: {s}")
        return int(mm.group(1)), int(mm.group(2))

    for rr in rows:
        cells = [c.strip() for c in rr.rstrip(r"\\").split("&")]
        inst = cells[0]
        est_openings = _parse_int_commas(cells[1])
        exp_openings = inst_to_openings_used.get(inst)
        checks.append(
            CheckResult(
                f"program_sizing_openings_{inst}",
                exp_openings is not None and est_openings == exp_openings,
                f"got={est_openings}, expected={exp_openings}",
            )
        )

        for share, cell in [(0.05, cells[2]), (0.10, cells[3]), (0.15, cells[4])]:
            got_lo, got_hi = _parse_range(cell)
            # Seats = openings*share/efficiency
            exp_lo = int((est_openings * share) / eff_high)  # min seats uses max efficiency
            exp_hi = int((est_openings * share) / eff_low)   # max seats uses min efficiency
            ok = (got_lo == exp_lo) and (got_hi == exp_hi)
            checks.append(
                CheckResult(
                    f"program_sizing_seats_{inst}_{int(100*share)}pct",
                    ok,
                    f"got=({got_lo},{got_hi}), expected=({exp_lo},{exp_hi})",
                )
            )

    return checks


def check_external_benchmark(bench_tex: Path, joined_csv: Path) -> list[CheckResult]:
    df = pd.read_csv(joined_csv)
    checks: list[CheckResult] = []

    # Determine columns available
    pairs = [
        ("NetRisk (uncalibrated mechanism)", "net_risk_uncalibrated"),
        ("NetRisk (calibrated predictive)", "net_risk_calibrated"),
        ("NetRisk (pipeline-used)", "net_risk"),
    ]

    # Parse table rows from tex
    tex = _read_text(bench_tex).splitlines()
    rows_all = [l.strip() for l in tex if l.strip().endswith(r"\\") and "&" in l and not l.strip().startswith("\\")]
    # Keep only data rows (avoid header like "Index & Matched occupations ...")
    rows = []
    for l in rows_all:
        first = l.split("&", 1)[0].strip()
        if first.startswith("NetRisk"):
            rows.append(l)
    # There should be 3 data rows
    if len(rows) < 3:
        return [
            CheckResult(
                "external_benchmark_rowcount",
                False,
                f"Expected >=3 NetRisk rows, got {len(rows)} (parsed {len(rows_all)} candidate rows)",
            )
        ]

    def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) == 0:
            return float("nan")
        return float(x[m].corr(y[m], method=method))

    # Build expected rounded values to 3 decimals
    exp: dict[str, tuple[int, float, float]] = {}
    for label, col in pairs:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df["aioe"], errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        pr = _corr(df[col], df["aioe"], "pearson")
        sr = _corr(df[col], df["aioe"], "spearman")
        exp[label] = (n, round(pr, 3), round(sr, 3))

    for rr in rows[:3]:
        cells = [c.strip() for c in rr.rstrip(r"\\").split("&")]
        label = cells[0]
        got_n = _parse_int_commas(cells[1])
        got_pr = round(float(cells[2]), 3)
        got_sr = round(float(cells[3]), 3)

        e = exp.get(label)
        if e is None:
            checks.append(CheckResult(f"external_benchmark_label_{label}", False, "Label not found in expected map"))
            continue
        ok = (got_n == e[0]) and _almost_equal(got_pr, e[1], 1e-6) and _almost_equal(got_sr, e[2], 1e-6)
        checks.append(CheckResult(f"external_benchmark_{label}", ok, f"got=(n={got_n},r={got_pr},rho={got_sr}), expected=(n={e[0]},r={e[1]},rho={e[2]})"))

    return checks


def main() -> None:
    main_tex = REPORTS_DIR / "main.tex"
    scenario_csv = DATA_DIR / "scenario_summary.csv"
    scenario_tex = TABLES_DIR / "scenario_summary.tex"
    program_tex = TABLES_DIR / "program_sizing.tex"
    openings_rob_tex = TABLES_DIR / "openings_scaling_robustness.tex"
    bench_tex = TABLES_DIR / "external_benchmark.tex"
    joined_csv = DATA_DIR / "netrisk_vs_aioe.csv"

    checks: list[CheckResult] = []

    checks.extend(check_inputs_and_figures(main_tex))
    if scenario_csv.exists():
        checks.extend(check_summary_sheet_headlines(main_tex, scenario_csv))
    else:
        checks.append(CheckResult("scenario_csv_exists", False, f"Missing: {scenario_csv}"))
    if scenario_csv.exists() and scenario_tex.exists():
        checks.extend(check_scenario_table_matches_csv(scenario_csv, scenario_tex))
    else:
        checks.append(CheckResult("scenario_table_inputs", False, f"Missing: {scenario_csv} or {scenario_tex}"))

    if program_tex.exists() and openings_rob_tex.exists():
        checks.extend(check_program_sizing(program_tex, openings_rob_tex))
    else:
        checks.append(CheckResult("program_sizing_inputs", False, f"Missing: {program_tex} or {openings_rob_tex}"))

    if bench_tex.exists() and joined_csv.exists():
        checks.extend(check_external_benchmark(bench_tex, joined_csv))
    else:
        checks.append(CheckResult("external_benchmark_inputs", False, f"Missing: {bench_tex} or {joined_csv}"))

    ok = all(c.ok for c in checks)
    n_ok = sum(1 for c in checks if c.ok)
    n_total = len(checks)

    lines: list[str] = []
    lines.append("Paper consistency check")
    lines.append("=" * 80)
    lines.append(f"Overall: {'PASS' if ok else 'FAIL'} ({n_ok}/{n_total} checks passed)")
    lines.append("")
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        lines.append(f"[{status}] {c.name}")
        if c.details:
            lines.append("  " + c.details.replace("\n", "\n  "))
    lines.append("")

    out = VALID_DIR / "paper_consistency_check.txt"
    _write_text(out, "\n".join(lines))
    print(f"Wrote {out}")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

