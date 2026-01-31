import os
from pathlib import Path

print(f"CWD: {os.getcwd()}")
p = Path("Problem #2")
if p.exists():
    print(f"'Problem #2' exists. Contents:")
    for x in p.iterdir():
        print(f"  {x.name}")
        if x.name == "Problem #2":
            print("    Found nested 'Problem #2'. Contents:")
            for y in x.iterdir():
                 print(f"      {y.name}")
else:
    print(f"'Problem #2' NOT found in {os.getcwd()}")
