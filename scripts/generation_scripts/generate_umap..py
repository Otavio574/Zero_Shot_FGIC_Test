import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

EMB_DIR = Path("embeddings")

all_embeds = []
all_labels = []

for f in EMB_DIR.glob("*.pt"):
    name = f.stem
    data = torch.load(f)
    if isinstance(data, dict):
        data = data["image_embeddings"]
    sample = data[:300].numpy()
    all_embeds.append(sample)
    all_labels += [name] * len(sample)

X = np.vstack(all_embeds)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(12, 8))
for name in set(all_labels):
    idx = [i for i, lbl in enumerate(all_labels) if lbl == name]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=name, s=10)
plt.legend(fontsize=8)
plt.title("Distribuição dos datasets no espaço CLIP (t-SNE)")
plt.tight_layout()
plt.savefig("outputs/tsne_datasets.png", dpi=300)
plt.show()
