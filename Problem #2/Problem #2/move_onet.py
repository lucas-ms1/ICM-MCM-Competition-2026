import shutil
from pathlib import Path

src = Path("Problem #2/Problem #2/data/onet_db/db_30_1_text")
dst = Path("Problem #2/data/onet")

dst.mkdir(parents=True, exist_ok=True)

if src.exists():
    print(f"Moving content from {src} to {dst}...")
    for item in src.iterdir():
        if item.is_file():
            shutil.move(str(item), dst / item.name)
    print("Move complete.")
    
    # Clean up empty source
    try:
        src.rmdir()
        src.parent.rmdir() # onet_db
        (src.parent.parent / "db_30_1_text.zip").unlink()
        src.parent.parent.rmdir() # data
        src.parent.parent.parent.rmdir() # Problem #2 nested
    except Exception as e:
        print(f"Cleanup warning: {e}")

else:
    print(f"Source {src} not found.")
