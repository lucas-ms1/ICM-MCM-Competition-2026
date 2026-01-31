import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
CAREERS_DIR = DATA_DIR / "careers"

INSTITUTIONS = [
    {"career": "software_engineer", "area_code": "41740", "inst": "SDSU"},
    {"career": "electrician", "area_code": "31080", "inst": "LATTC"},
    {"career": "writer", "area_code": "41860", "inst": "Academy of Art"},
]

def main():
    for item in INSTITUTIONS:
        path = CAREERS_DIR / f"{item['career']}.csv"
        if not path.exists():
            continue
            
        df = pd.read_csv(path, dtype=str)
        # Filter for area
        row = df[df["area_code"] == item["area_code"]]
        
        print(f"\n--- {item['inst']} ({item['career']}) ---")
        if row.empty:
            print("No local data found.")
        else:
            # Print Emp, Mean Wage, Median Wage
            r = row.iloc[0]
            print(f"Area: {r['area_title']}")
            print(f"Employment: {r['emp']}")
            print(f"Mean Wage: {r['a_mean']}")
            print(f"Median Wage: {r['a_median']}")
            print(f"Location Quotient: {r.get('loc_quotient', 'N/A')}") # loc_quotient not in baseline, but maybe useful if I had it. OEWS has it usually? 
            # My clean script kept minimal cols.

if __name__ == "__main__":
    main()
