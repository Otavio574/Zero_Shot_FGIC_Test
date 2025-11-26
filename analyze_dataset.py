import os
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Carregar config
CONFIG_PATH = Path("datasets_config.json")
config = json.load(open(CONFIG_PATH))
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_images_in_folder(folder, max_images=5000):
    exts = [".jpg", ".jpeg", ".png"]
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(e) for e in exts):
                imgs.append(os.path.join(root, f))
        if len(imgs) >= max_images:
            break
    return imgs

summary = []
for ds in config["datasets"]:
    name = ds["name"]
    path = Path(ds["path"])
    print(f"📂 Analisando {name}...")

    images = get_images_in_folder(path)
    num_imgs = len(images)

    if num_imgs == 0:
        print(f"⚠️ Nenhuma imagem encontrada em {path}")
        continue

    # Estatísticas simples
    sizes = []
    for img_path in random.sample(images, min(100, num_imgs)):
        try:
            with Image.open(img_path) as im:
                sizes.append(im.size)
        except Exception:
            continue

    if sizes:
        w, h = zip(*sizes)
        avg_w, avg_h = sum(w)/len(w), sum(h)/len(h)
    else:
        avg_w = avg_h = 0

    summary.append({
        "dataset": name,
        "num_images": num_imgs,
        "avg_width": avg_w,
        "avg_height": avg_h,
        "path": str(path)
    })

    # Amostra visual
    sample_imgs = random.sample(images, min(5, num_imgs))
    fig, axes = plt.subplots(1, len(sample_imgs), figsize=(15, 3))
    for ax, img in zip(axes, sample_imgs):
        try:
            ax.imshow(Image.open(img))
        except Exception:
            continue
        ax.axis("off")
    plt.suptitle(f"{name} - {len(images)} imagens")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_samples.jpg")
    plt.close()

# Salvar resumo JSON e tabela
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Tabela simples
print("\n📊 Resumo dos datasets:")
for s in summary:
    print(f"- {s['dataset']}: {s['num_images']} imagens | {s['avg_width']:.0f}x{s['avg_height']:.0f}")

print(f"\n✅ Relatório salvo em: {OUTPUT_DIR}")
