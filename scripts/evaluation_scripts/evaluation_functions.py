import os
import json
import torch
import numpy as np
import clip
import random
import string
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import traceback
from typing import Dict, Any, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings"

def load_datasets_from_summary(path: Path) -> Dict[str, str]:
    """Carrega paths dos datasets a partir do summary.json."""
    if not path.exists():
        print("❌ summary.json não encontrado!")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = {}
    for item in data:
        if "dataset" in item and "path" in item:
            datasets[item["dataset"]] = item["path"]
    return datasets

def get_text_embedding_baseline(class_name: str, model, clip_library, device: str) -> torch.Tensor:
    """Gera embedding de texto usando apenas o template vanilla CLIP: 'a photo of a {class}'."""
    class_readable = class_name.replace('_', ' ')
    
    # 🎯 TEMPLATE VANILLA CLIP (paper original)
    text = f"a photo of a {class_readable}"
    
    tokens = clip_library.tokenize([text]).to(device)
    
    with torch.no_grad():
        text_embed = model.encode_text(tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    
    return text_embed.squeeze(0).cpu()

def load_embeddings_and_generate_baseline_text(dataset_name: str, model, clip_library, device: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str]]:
    """Carrega embeddings de imagem e gera text embeddings vanilla CLIP."""
    
    # Adicionamos o carregamento da lista de datasets aqui, caso o script master não o faça.
    DATASETS = load_datasets_from_summary(SUMMARY_PATH)
    if dataset_name not in DATASETS:
        print(f"Dataset {dataset_name} não encontrado no summary.json.")
        return None, None, None, None

    emb_path = EMBED_DIR / f"{dataset_name}.pt"

    if not emb_path.exists():
        print(f"⚠️ Embeddings de imagem não encontrados: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings de imagem: {emb_path}")
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    image_embeds = data.get("image_embeddings")
    image_paths = data.get("image_paths")

    if image_embeds is None or image_paths is None:
        print("❌ .pt inválido, faltando chaves.")
        return None, None, None, None

    # Normalização e extração de labels
    image_embeds = image_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    class_names = sorted(list(set(Path(p).parts[-2] for p in image_paths)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[Path(p).parts[-2]] for p in image_paths])

    print(f"  Total imagens: {len(labels)} | Classes: {len(class_names)}")

    # Gera text embeddings vanilla CLIP
    print("\n📝 Gerando text embeddings vanilla CLIP...")
    print('  Template: "a photo of a {class}"')

    text_embeds_list = []
    for cls in tqdm(class_names, desc="  Classes"):
        emb = get_text_embedding_baseline(cls, model, clip_library, device)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)
    print(f"✅ Text embeddings baseline: {text_embeds.shape}")

    # Retorna todos os dados e embeddings necessários para a avaliação
    return image_embeds, text_embeds, labels, class_names

# ============================================================
# FUNÇÃO DE AVALIAÇÃO CENTRAL (Baseada no evaluate_zero_shot original)
# ============================================================

def evaluate_zero_shot(model: torch.nn.Module, clip_library: Any, device: str, dataset_name: str) -> Dict[str, Any]:
    """
    Função principal de avaliação do Zero-Shot Baseline.
    Equivalente à lógica completa do evaluate_clip_zero-shot.py.
    """
    
    print("\n--- Executando Avaliação Padrão Zero-Shot (Vanilla CLIP) ---")
    
    try:
        # Carrega embeddings e gera text embeddings baseline
        image_embeds, text_embeds, labels, class_names = \
            load_embeddings_and_generate_baseline_text(
                dataset_name, model, clip_library, device
            )

        if image_embeds is None:
            return {}

        # 1. Similaridade Coseno
        # 
        sims = image_embeds.float() @ text_embeds.float().T  # [N_imgs, N_classes]
        preds = sims.argmax(dim=-1).numpy()
        
        print(f"\n🔍 Similaridades:")
        print(f"  Min: {sims.min():.4f}, Max: {sims.max():.4f}")
        
        # 2. Top-1 accuracy
        acc = accuracy_score(labels, preds)
        
        # 3. Top-5 accuracy
        top5_preds = sims.topk(5, dim=-1).indices.numpy()
        top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
        
        print(f"  Top-1 Accuracy: {acc:.4f}")
        print(f"  Top-5 Accuracy: {top5_acc:.4f}")

        # 4. Retorna Métricas
        return {
            f"{dataset_name}_accuracy_top1": float(acc),
            f"{dataset_name}_accuracy_top5": float(top5_acc),
            f"{dataset_name}_num_classes": len(class_names),
            f"{dataset_name}_num_images": len(labels),
            f"{dataset_name}_method": "vanilla_clip",
            f"{dataset_name}_template": "a photo of a {class}"
        }

    except Exception as e:
        print(f"❌ Erro na avaliação Zero-Shot para {dataset_name}: {e}")
        traceback.print_exc()
        return {}

def evaluate_zero_shot_description(model, data_loader, device):
    """Lógica do evaluate_clip_zero-shot_description.py"""
    print("--- Executando Avaliação com Descrição (Prompt Engineering) ---")
    # ... Coloque aqui a lógica principal do seu arquivo original ...
    return {"description_acc": 0.93}

def evaluate_zero_shot_comparative(model, data_loader, device):
    """Lógica do evaluate_clip_zero-shot_comparative.py"""
    print("--- Executando Avaliação Comparativa ---")
    # ... Coloque aqui a lógica principal do seu arquivo original ...
    return {"comparative_f1": 0.88}

def evaluate_zero_shot_comparative_filtering(model, data_loader, device):
    """Lógica do evaluate_clip_zero-shot_comparative_filtering.py"""
    print("--- Executando Avaliação Comparativa com Filtro ---")
    # ... Coloque aqui a lógica principal do seu arquivo original ...
    return {"comp_filter_acc": 0.91}

def evaluate_zero_shot_waffle(model, data_loader, device):
    """Lógica do evaluate_clip_zero-shot_waffle.py"""
    print("--- Executando Avaliação Waffle ---")
    # ... Coloque aqui a lógica principal do seu arquivo original ...
    return {"waffle_metric": 0.80}

# Todas as funções devem aceitar (model, data_loader, device) como entrada.