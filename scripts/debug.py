import torch
import json

# 1. Verifica embeddings
data = torch.load("embeddings/Stanford_Dogs.pt", weights_only=False)
print("=== EMBEDDINGS ===")
print(f"Keys: {data.keys() if isinstance(data, dict) else 'tensor only'}")
print(f"Shape: {data['image_embeddings'].shape}")
print(f"Num paths: {len(data['image_paths'])}")
print(f"Sample path: {data['image_paths'][0]}")

# 2. Verifica descriptors
with open("descriptors/Stanford_Dogs_descriptors.json") as f:
    desc = json.load(f)
print(f"\n=== DESCRIPTORS ===")
print(f"Num descriptors: {len(desc)}")
print(f"Sample keys:")
for i, key in enumerate(list(desc.keys())[:3]):
    print(f"  {key}: {desc[key][:80]}...")

# 3. Extrai classes únicas dos paths
from pathlib import Path
classes = set()
for path in data['image_paths']:
    classes.add(Path(path).parts[-2])
print(f"\n=== CLASSES ===")
print(f"Num classes: {len(classes)}")
print(f"Sample classes: {list(classes)[:3]}")