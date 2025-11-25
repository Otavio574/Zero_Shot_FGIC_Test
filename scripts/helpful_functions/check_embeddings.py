import torch
data = torch.load("embeddings_openai/CUB_200_2011.pt")
emb = data["image_embeddings"]

print("std:", emb.std().item())
print("mean:", emb.mean().item())
print("min:", emb.min().item(), "max:", emb.max().item())
