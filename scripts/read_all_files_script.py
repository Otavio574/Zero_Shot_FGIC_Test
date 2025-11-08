from pathlib import Path

path = Path("datasets")
for p in path.rglob("*.jpg"):
    print("🖼️", p)
