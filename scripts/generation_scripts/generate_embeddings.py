"""
Script de geração de embeddings de imagem para múltiplos modelos CLIP.
Gera um arquivo .pt (tensor) por dataset e por modelo.

Necessário para avaliação zero-shot, pois os embeddings de imagem (mat1)
e de texto (mat2) devem ter a mesma dimensão (D).
"""

import os
import json
import pathlib
import torch
import clip
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

# ==========================
# CONFIG
# ==========================

# Lista de modelos para iterar (mantida em sincronia com o script de avaliação)
ALL_MODELS: List[str] = [
    'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
    'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Assumindo que o script está em scripts/generation_scripts
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

SUMMARY_PATH = PROJECT_ROOT / "outputs" / "analysis" / "summary.json"
OUT_DIR = PROJECT_ROOT / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets_from_summary(summary_path: pathlib.Path) -> Dict[str, str]:
    """
    Carrega a lista de datasets e seus caminhos a partir do arquivo summary.json.
    """
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar summary.json: {e}")
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


def generate_embeddings_for_model(model_name: str, datasets: Dict[str, str], project_root: pathlib.Path):
    """
    Gera embeddings para todos os datasets usando um único modelo CLIP.
    """
    model_safe_name = model_name.replace('/', '-')
    print("\n" + "=" * 80)
    print(f"🚀 INICIANDO GERAÇÃO DE EMBEDDINGS PARA O MODELO: {model_name}")
    print("=" * 80)
    
    # Verificar quais datasets já têm embeddings para este modelo
    datasets_to_process = {}
    datasets_skipped = []
    
    for dataset_name, dataset_path in datasets.items():
        # O nome do arquivo de saída deve incluir o modelo para evitar conflitos
        save_path = OUT_DIR / f"{dataset_name}_{model_safe_name}.pt"
        if save_path.exists():
            datasets_skipped.append(dataset_name)
        else:
            datasets_to_process[dataset_name] = dataset_path
    
    if datasets_skipped:
        print(f"⏭️  {len(datasets_skipped)} datasets já possuem embeddings para este modelo. Pulando: {', '.join(datasets_skipped)}")

    if not datasets_to_process:
        print(f"✅ Todos os embeddings já foram gerados para {model_name}. Passando para o próximo.")
        return

    print("🔄 Carregando CLIP...")
    try:
        model, preprocess = clip.load(model_name, device=DEVICE)
        model.eval()
        print(f"✅ Modelo carregado! Dimensão do embedding: {model.visual.output_dim}\n")
    except Exception as e:
        print(f"❌ Não foi possível carregar o modelo {model_name}: {e}")
        return

    for dataset_name, dataset_path in datasets_to_process.items():
        # Define o caminho completo para a pasta do dataset
        root = project_root / pathlib.Path(dataset_path) 
        
        # O nome do arquivo de saída deve incluir o modelo para evitar conflitos
        save_path = OUT_DIR / f"{dataset_name}_{model_safe_name}.pt"
        
        if not root.exists():
            print(f"⚠️ Dataset não encontrado: {root}")
            continue

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

        pbar = tqdm(total=total_imgs, desc=f"📷 {dataset_name} ({model_name})")

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

        image_embeds = torch.cat(image_embeds_list, dim=0)

        torch.save(
            {
                "image_embeddings": image_embeds,
                "image_paths": image_paths,
                "clip_model": model_name,
                "clip_lib": "openai/clip_official",
            },
            save_path,
        )

        print(f"✅ Salvo: {save_path}")
        print(f"   Embeddings: {image_embeds.shape}")
        print(f"   Imagens:    {len(image_paths)}")


def main():
    print(f"🖥️  Gerador de Embeddings CLIP Multi-Modelo")
    print(f"💻 Device: {DEVICE}\n")
    print(f"📁 Raiz do Projeto (Assumida): {PROJECT_ROOT.resolve()}")

    datasets = load_datasets_from_summary(SUMMARY_PATH)
    if not datasets:
        print("❌ Nenhum dataset encontrado no summary.json")
        return

    # Iterar sobre a lista completa de modelos
    for model_name in ALL_MODELS:
        generate_embeddings_for_model(model_name, datasets, PROJECT_ROOT)

    print("\n\n🎉 Geração de Embeddings para todos os modelos concluída!")


if __name__ == "__main__":
    main()