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
    Validates that the Summary Sheet headline findings are structurally derived
    from scenario artifacts by:
      1) checking that tables/scenario_macros.tex exists and matches data/scenario_summary.csv
      2) checking that tables/summary_headline_fragment.tex references those macros (no literals)
    """
    df = pd.read_csv(scenario_csv).set_index("career")
    macros_path = main_tex.parent / "tables" / "scenario_macros.tex"
    frag_path = main_tex.parent / "tables" / "summary_headline_fragment.tex"

    if not macros_path.exists():
        return [CheckResult("scenario_macros_exist", False, f"Missing: {macros_path}")]
    if not frag_path.exists():
        return [CheckResult("summary_headline_fragment_exist", False, f"Missing: {frag_path}")]

    macros = _read_text(macros_path)
    frag = _read_text(frag_path)

    def _extract_num_macro(macro_name: str) -> int:
        # expects: \newcommand{\EHighSoftware}{\num{1490209}}
        mm = re.search(
            r"\\newcommand\{\\"
            + re.escape(macro_name)
            + r"\}\{\\num\{(?P<num>-?\d+)\}\}",
            macros,
        )
        if not mm:
            raise ValueError(f"Missing or malformed macro: {macro_name}")
        return int(mm.group("num"))

    mapping = [
        ("software_engineer", "Software"),
        ("electrician", "Electrician"),
        ("writer", "Writer"),
    ]

    checks: list[CheckResult] = []
    for key, suf in mapping:
        r = df.loc[key]
        exp_base = int(round(float(r["emp_2034_No_GenAI_Baseline"])))
        exp_high = int(round(float(r["emp_2034_High_Disruption"])))
        exp_ramp = int(round(float(r["emp_2034_Ramp_High"])))

        try:
            got_base = _extract_num_macro(f"EBase{suf}")
            got_high = _extract_num_macro(f"EHigh{suf}")
            got_ramp = _extract_num_macro(f"ERampHigh{suf}")
            ok = (got_base == exp_base) and (got_high == exp_high) and (got_ramp == exp_ramp)
            checks.append(
                CheckResult(
                    name=f"scenario_macros_{key}",
                    ok=ok,
                    details=f"got=(base={got_base},high={got_high},ramp={got_ramp}), expected=(base={exp_base},high={exp_high},ramp={exp_ramp})",
                )
            )
        except Exception as e:
            checks.append(CheckResult(name=f"scenario_macros_{key}", ok=False, details=str(e)))

    # Ensure headline fragment is macro-driven (no hand-entered digits)
    required_macros = [
        r"\EBaseSoftware",
        r"\EHighSoftware",
        r"\ERampHighSoftware",
        r"\EBaseElectrician",
        r"\EHighElectrician",
        r"\ERampHighElectrician",
        r"\EBaseWriter",
        r"\EHighWriter",
        r"\ERampHighWriter",
    ]
    missing = [m for m in required_macros if m not in frag]
    checks.append(
        CheckResult(
            "summary_headline_fragment_uses_macros",
            ok=(len(missing) == 0),
            details=("Missing macros in fragment: " + ", ".join(missing)) if missing else "OK",
        )
    )

    # Guardrail: fragment shouldn't contain long runs of digits (copy/paste risk)
    has_literal_digits = re.search(r"\d{4,}", frag) is not None
    checks.append(
        CheckResult(
            "summary_headline_fragment_no_literals",
            ok=(not has_literal_digits),
            details="Found literal 4+ digit sequences in headline fragment." if has_literal_digits else "OK",
        )
    )

    return checks


def check_summary_sheet_one_page(main_tex: Path) -> list[CheckResult]:
    """
    Hard-fail if the Summary Sheet spills beyond page 1.
    We check the LaTeX aux labels for page numbers after PDF compilation.
    """
    aux = main_tex.parent / "main.aux"
    if not aux.exists():
        return [
            CheckResult(
                "summary_sheet_one_page_aux_exists",
                False,
                f"Missing {aux}. Build the PDF to enable the one-page Summary Sheet check.",
            )
        ]
    txt = _read_text(aux)

    def _page(label: str) -> int | None:
        # Typical LaTeX aux format:
        # \newlabel{page:summary_end}{{}{2}{Summary Sheet}{section*.1}{}}
        mm = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{\}\{(\d+)\}", txt)
        if not mm:
            return None
        return int(mm.group(1))

    p_end = _page("page:summary_end")
    p_toc = _page("page:toc_start")
    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            "summary_sheet_end_on_page_1",
            ok=(p_end == 1),
            details=f"page:summary_end={p_end} (expected 1)",
        )
    )
    # Optional sanity: TOC should start on page 2
    if p_toc is not None:
        checks.append(
            CheckResult(
                "toc_starts_on_page_2",
                ok=(p_toc == 2),
                details=f"page:toc_start={p_toc} (expected 2)",
            )
        )
    return checks


def check_summary_sheet_refs(main_tex: Path) -> list[CheckResult]:
    """
    Verify that every \\ref{...} used in the Summary Sheet points to an existing \\label{...}
    somewhere in main.tex or any input'd table fragments.
    """
    text = _read_text(main_tex)
    # Extract Summary Sheet block (up to the explicit page break after it).
    m = re.search(r"\\section\*\{Summary Sheet\}(.+?)\\newpage", text, flags=re.DOTALL)
    if not m:
        return [CheckResult("summary_sheet_block_parse", False, "Could not locate Summary Sheet block.")]
    block = m.group(1)
    refs = sorted(set(re.findall(r"\\ref\{([^}]+)\}", block)))
    if not refs:
        return [CheckResult("summary_sheet_refs_present", True, "No refs in Summary Sheet.")]

    # Collect all labels across main.tex + tables/*.tex
    label_texts: list[str] = [text]
    for p in TABLES_DIR.glob("*.tex"):
        try:
            label_texts.append(_read_text(p))
        except Exception:
            pass
    all_text = "\n".join(label_texts)
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_text))

    missing = [r for r in refs if r not in labels]
    return [
        CheckResult(
            "summary_sheet_refs_resolve",
            ok=(len(missing) == 0),
            details=("Missing labels: " + ", ".join(missing)) if missing else f"OK ({len(refs)} refs resolved)",
        )
    ]


def check_model_definition_claims(main_tex: Path) -> list[CheckResult]:
    """
    Light-weight structural check that the Summary Sheet's model-definition claims
    are consistent with the Model section definitions (same components + cap logic).

    This is not a full LaTeX math equivalence prover; it's a guardrail against drift.
    """
    text = _read_text(main_tex)

    # Summary Sheet block (up to the explicit page break after it).
    msum = re.search(r"\\section\*\{Summary Sheet\}(.+?)\\newpage", text, flags=re.DOTALL)
    if not msum:
        return [CheckResult("model_claims_summary_parse", False, "Could not locate Summary Sheet block.")]
    summary = msum.group(1)

    # Model section block
    mmodel = re.search(r"\\section\{Model\}(.+?)\\section\{Results\}", text, flags=re.DOTALL)
    if not mmodel:
        return [CheckResult("model_claims_model_parse", False, "Could not locate Model section block.")]
    model = mmodel.group(1)

    # Check NetRisk components appear in Summary Sheet definition line
    need_summary = [
        r"\NetRisk",
        "Writing",
        "ToolTech",
        "Physical",
        "Social",
        "Creativity",
    ]
    miss_sum = [s for s in need_summary if s not in summary]

    # Check that Model section defines SubScore/DefScore with matching components and divisors.
    # (We look for the specific dimension identifiers used in the narrative.)
    need_model = [
        r"\SubScore_i",
        "Writing",
        "ToolTech",
        r"\DefScore_i",
        "Physical",
        "Social",
        "Creativity",
    ]
    miss_model = [s for s in need_model if s not in model]
    # Divisors: allow either explicit "/2" "/3" or LaTeX \frac{...}{2} / \frac{...}{3}
    # Avoid fragile full-LaTeX parsing; just require the divisor tokens appear.
    has_div2 = ("/2" in model) or (("}{2}" in model) and ("\\frac" in model))
    has_div3 = ("/3" in model) or (("}{3}" in model) and ("\\frac" in model))
    if not has_div2:
        miss_model.append("div2")
    if not has_div3:
        miss_model.append("div3")

    # Check cap logic exists in both places.
    # Summary Sheet can choose to mention either m_max symbol or just the numeric cap.
    cap_need_sum = ["m_i", "0.2"]
    miss_cap_sum = [s for s in cap_need_sum if s not in summary]
    has_mmax_sum = ("m_{\\max}" in summary) or ("m_{\\max}=0.2" in summary) or ("m_{\\max} = 0.2" in summary)
    # Model section should include the symbolic cap definition.
    cap_need_model = ["m_i", "m_{\\max}", "0.2"]
    miss_cap_model = [s for s in cap_need_model if s not in model]

    ok = (len(miss_sum) == 0) and (len(miss_model) == 0) and (len(miss_cap_sum) == 0) and (len(miss_cap_model) == 0)
    details = []
    if miss_sum:
        details.append("Summary missing: " + ", ".join(miss_sum))
    if miss_model:
        details.append("Model missing: " + ", ".join(miss_model))
    if miss_cap_sum:
        details.append("Summary cap missing: " + ", ".join(miss_cap_sum))
    if miss_cap_model:
        details.append("Model cap missing: " + ", ".join(miss_cap_model))

    return [CheckResult("model_definition_claims_consistent", ok=ok, details="; ".join(details) if details else "OK")]


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
    checks.extend(check_summary_sheet_refs(main_tex))
    checks.extend(check_model_definition_claims(main_tex))
    checks.extend(check_summary_sheet_one_page(main_tex))
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

