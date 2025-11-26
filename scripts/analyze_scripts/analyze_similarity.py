import torch
import os
import itertools
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

EMB_DIR = Path("embeddings")

datasets = [f for f in os.listdir(EMB_DIR) if f.endswith(".pt")]
results = []

for a, b in itertools.combinations(datasets, 2):
    emb_a = torch.load(EMB_DIR / a)
    emb_b = torch.load(EMB_DIR / b)

    # Garantir formato tensor puro
    if isinstance(emb_a, dict):
        emb_a = emb_a["image_embeddings"]
    if isinstance(emb_b, dict):
        emb_b = emb_b["image_embeddings"]

    sim = cosine_similarity(emb_a[:500].numpy(), emb_b[:500].numpy()).mean()
    results.append((a.replace(".pt", ""), b.replace(".pt", ""), sim))

# Ordenar e exibir top 10 mais semelhantes
results.sort(key=lambda x: x[2], reverse=True)
print("\n🔝 Top 10 pares mais semelhantes:")
for a, b, sim in results[:10]:
    print(f"{a} ↔ {b} → Similaridade média: {sim:.4f}")

torch.save(results, "outputs/dataset_similarity.pt")
