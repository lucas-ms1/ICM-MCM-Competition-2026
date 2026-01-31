import pandas as pd
from pathlib import Path

path = Path("data/occupation.xlsx")
xl = pd.ExcelFile(path)
print("Sheet names:", xl.sheet_names)

for sheet in xl.sheet_names:
    print(f"\n--- Sheet: {sheet} ---")
    df = pd.read_excel(path, sheet_name=sheet, nrows=5)
    print(df.columns.tolist())
    print(df.head())
