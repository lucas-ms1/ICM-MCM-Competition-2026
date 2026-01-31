import subprocess
import sys
from pathlib import Path

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
        base_dir / "reports" / "tables" / "onet_elements_appendix.tex",
        base_dir / "reports" / "tables" / "program_sizing.tex",
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

def main():
    base_dir = Path(__file__).resolve().parent
    
    scripts = [
        "build_tables.py",
        "build_mechanism_layer_expanded.py",
        "run_scenarios.py",
        "build_report_artifacts.py"
    ]
    
    for s in scripts:
        run_script(base_dir / s)
        
    validate_outputs(base_dir)
    print("\n>>> Pipeline complete.")

if __name__ == "__main__":
    main()
