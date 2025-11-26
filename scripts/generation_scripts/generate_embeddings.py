import os
import json
import pathlib

import torch
import clip
from PIL import Image
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sobe dois níveis (..) para chegar na raiz e encontrar a pasta 'outputs'
SUMMARY_PATH = pathlib.Path("..") / ".." / "outputs" / "analysis" / "summary.json"
OUT_DIR = pathlib.Path("..") / ".." / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets_from_summary(summary_path: pathlib.Path) -> dict:
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar summary: {e}")
        return {}

    datasets = {}
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get("dataset")
            dataset_path = item.get("path")
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    elif isinstance(summary, dict) and "datasets" in summary:
        for name, path in summary["datasets"].items():
            datasets[name] = path

    return datasets


def main():
    print(f"🚀 Gerando embeddings com CLIP oficial")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")

    # 🔑 CORREÇÃO: Definir a Raiz do Projeto (subida dupla) para resolver o caminho dos Datasets
    # Isso garante que ele volte de 'scripts/generation_scripts/' para a raiz do projeto.
    PROJECT_ROOT = pathlib.Path("..") / ".."
    print(f"📁 Raiz do Projeto (Assumida): {PROJECT_ROOT.resolve()}")

    datasets = load_datasets_from_summary(SUMMARY_PATH)
    if not datasets:
        print("❌ Nenhum dataset encontrado no summary.json")
        return

    # Verificar quais datasets já têm embeddings
    datasets_to_process = {}
    datasets_skipped = []
    
    for dataset_name, dataset_path in datasets.items():
        save_path = OUT_DIR / f"{dataset_name}.pt"
        if save_path.exists():
            print(f"⏭️  Pulando {dataset_name} (embedding já existe)")
            datasets_skipped.append(dataset_name)
        else:
            datasets_to_process[dataset_name] = dataset_path
    
    if datasets_skipped:
        print(f"\n📋 Datasets pulados: {len(datasets_skipped)}")
        print(f"🔨 Datasets a processar: {len(datasets_to_process)}\n")
    
    if not datasets_to_process:
        print("✅ Todos os embeddings já foram gerados!")
        return

    print("🔄 Carregando CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ Modelo carregado!\n")

    for dataset_name, dataset_path in datasets_to_process.items():
        # 🔑 CORREÇÃO APLICADA: Junta a Raiz do Projeto com o caminho relativo (datasets/...)
        root = PROJECT_ROOT / pathlib.Path(dataset_path) 
        
        if not root.exists():
            print(f"⚠️ Dataset não encontrado: {root}")
            continue

        save_path = OUT_DIR / f"{dataset_name}.pt"
        print(f"\n📘 Dataset: {dataset_name}")
        print(f"📁 Raiz: {root}")
        print(f"💾 Saída: {save_path}")

        image_embeds_list = []
        image_paths = []

        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        total_imgs = sum(
            len([p for p in cls.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
            for cls in class_dirs
        )

        pbar = tqdm(total=total_imgs, desc=f"📷 {dataset_name}")

        for cls in class_dirs:
            imgs = sorted([
                p for p in cls.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ])

            for img_path in imgs:
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    pbar.update(1)
                    continue

                img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feat = model.encode_image(img_tensor)
                    feat = feat / feat.norm(dim=-1, keepdim=True)  # ✅ NORMALIZAÇÃO
                    feat = feat.float().cpu()

                image_embeds_list.append(feat)
                image_paths.append(str(img_path))
                pbar.update(1)

        pbar.close()

        if not image_embeds_list:
            print("⚠️ Nenhuma imagem encontrada, pulando...")
            continue

        image_embeds = torch.cat(image_embeds_list, dim=0)  # [N, 512]

        torch.save(
            {
                "image_embeddings": image_embeds,
                "image_paths": image_paths,
                "clip_model": MODEL_NAME,
                "clip_lib": "openai/clip_official",
            },
            save_path,
        )

        print(f"✅ Salvo: {save_path}")
        print(f"   Embeddings: {image_embeds.shape}")
        print(f"   Imagens:    {len(image_paths)}")

    print("\n🎉 Embeddings gerados corretamente!")


if __name__ == "__main__":
    main()