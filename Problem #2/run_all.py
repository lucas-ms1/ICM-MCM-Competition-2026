import subprocess
import sys
from pathlib import Path
import shutil

def run_script(script_name):
    print(f"\n>>> Running {script_name}...")
    # Run the script in its own directory context if needed, but here they are all in root
    # We pass cwd=base_dir to ensure relative paths work
    result = subprocess.run([sys.executable, str(script_name)], capture_output=True, text=True, cwd=script_name.parent)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print(result.stdout)

def validate_outputs(base_dir):
    print("\n>>> Validating outputs...")
    required = [
        base_dir / "data" / "mechanism_layer_all.csv",
        base_dir / "data" / "scenario_summary.csv",
        base_dir / "data" / "mechanism_element_map.csv",
        base_dir / "data" / "netrisk_vs_aioe.csv",
        base_dir / "reports" / "tables" / "onet_elements_appendix.tex",
        base_dir / "reports" / "tables" / "program_sizing.tex",
        base_dir / "reports" / "tables" / "external_benchmark.tex",
        base_dir / "reports" / "tables" / "policy_regimes.tex",
        base_dir / "data" / "validation" / "calibration_check.txt"
    ]
    
    all_ok = True
    for p in required:
        if not p.exists():
            print(f"FAIL: Missing {p}")
            all_ok = False
        else:
            print(f"OK: {p.relative_to(base_dir)}")
            
    if all_ok:
        print("All required artifacts present.")
    else:
        print("Some artifacts are missing.")

def build_pdf(base_dir: Path) -> None:
    """
    Compile reports/main.tex -> reports/main.pdf.

    We avoid latexmk (requires perl on some MiKTeX installs) and instead use the
    standard pdflatex/bibtex cycle.
    """
    reports_dir = base_dir / "reports"
    tex_path = reports_dir / "main.tex"
    if not tex_path.exists():
        print(f"\n>>> Skipping PDF build: missing {tex_path}")
        return

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not pdflatex or not bibtex:
        print("\n>>> Skipping PDF build: pdflatex/bibtex not found on PATH.")
        print("    Tip: open reports/main.tex and compile in your LaTeX environment, or install MiKTeX.")
        return

    print("\n>>> Building PDF (pdflatex/bibtex)...")
    # 1) First pass: write aux
    subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=reports_dir, check=True)
    # 2) BibTeX: build bibliography from aux
    subprocess.run([bibtex, "main"], cwd=reports_dir, check=True)
    # 3-4) Resolve citations + cross-refs
    subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=reports_dir, check=True)
    subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=reports_dir, check=True)
    print(f">>> Wrote {reports_dir / 'main.pdf'}")

def main():
    base_dir = Path(__file__).resolve().parent
    
    scripts = [
        "build_tables.py",
        "build_mechanism_layer_expanded.py",
        "build_mechanism_sensitivity.py",
        "build_calibration.py",
        "build_calibration_validation.py",
        "run_scenarios.py",
        "build_external_benchmark.py",
        "build_uncertainty.py",
        "build_policy_model.py",
        "build_report_artifacts.py"
    ]
    
    for s in scripts:
        run_script(base_dir / s)
        
    validate_outputs(base_dir)
    build_pdf(base_dir)
    print("\n>>> Pipeline complete.")

if __name__ == "__main__":
    main()
