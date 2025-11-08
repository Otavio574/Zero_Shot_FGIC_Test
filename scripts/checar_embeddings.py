import torch

data = torch.load("embeddings/CUB_200_2011.pt")
print("Keys:", data.keys())
print("Embeddings shape:", data["image_embeddings"].shape)
print("First image path:", data["image_paths"][0])


# Adicione no início do código para debug: