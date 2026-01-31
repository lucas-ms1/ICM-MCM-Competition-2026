import shutil
from pathlib import Path

# Relative to workspace root
src = Path("Problem #2/data/onet_db/db_30_1_text")
dst = Path("data/onet")

dst.mkdir(parents=True, exist_ok=True)

if src.exists():
    print(f"Moving content from {src} to {dst}...")
    for item in src.iterdir():
        if item.is_file():
            shutil.move(str(item), dst / item.name)
    print("Move complete.")
    
    # Clean up empty source
    try:
        shutil.rmtree(src) # remove db_30_1_text
        shutil.rmtree(src.parent) # remove onet_db
        (src.parent.parent / "db_30_1_text.zip").unlink() # remove zip
        # src.parent.parent is Problem #2/data
    except Exception as e:
        print(f"Cleanup warning: {e}")

else:
    print(f"Source {src} not found.")
