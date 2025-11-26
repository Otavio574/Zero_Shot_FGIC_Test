import clip
import torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device="cpu")

img = Image.open("datasets/CUB_200_2011/Acadian_Flycatcher/Acadian_Flycatcher_0003_29094.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)

with torch.no_grad():
    emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)

print("std =", emb.std().item())
print("mean =", emb.mean().item())
print("min/max =", emb.min().item(), emb.max().item())
