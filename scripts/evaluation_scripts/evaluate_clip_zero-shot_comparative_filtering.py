"""
Avaliação Zero-Shot com CLIP usando Comparative Filtering.
Usa embeddings pré-calculados e filtra classes muito similares.
Método: Remove classes cujos text embeddings são muito similares entre si.
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from collections import Counter

# ============================
# CONFIGURAÇÕES
# ============================

def load_datasets_from_summary(summary_path: Path) -> dict:
    """Carrega configuração de datasets do summary.json"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    datasets = {}
    
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    
    return datasets


SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS = load_datasets_from_summary(SUMMARY_PATH)

MODEL_NAME = "ViT-B/32"  # Modelo CLIP do OpenAI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_comparative_filtering"

# Parâmetro de filtragem
SIMILARITY_THRESHOLD = 0.7 # Remove classes com similaridade média > 0.7

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptions(dataset_name):
    """Carrega comparative descriptors do dataset"""
    path = os.path.join("descriptors", f"{dataset_name}_descriptors_comparative.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️  Nenhum descriptor COMPARATIVO encontrado para {dataset_name}, usando genéricos.")
        return {}


def load_embeddings(dataset_name):
    """Carrega embeddings de imagem"""
    emb_path = os.path.join("embeddings", f"{dataset_name}.pt")
    if not os.path.exists(emb_path):
        print(f"❌ Embeddings não encontrados: {emb_path}")
        return None, None
    
    data = torch.load(emb_path, weights_only=False)
    if isinstance(data, dict):
        return data["image_embeddings"], data.get("image_paths", None)
    return data, None


def infer_classes_from_paths(image_paths):
    """Extrai nomes de classes dos caminhos"""
    class_names = []
    labels = []
    class_to_idx = {}

    for path in image_paths:
        parts = Path(path).parts
        class_name = parts[-2] if len(parts) >= 2 else "unknown"
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_names)
            class_names.append(class_name)
        labels.append(class_to_idx[class_name])

    return np.array(labels), class_names


def apply_comparative_filtering_safe(text_embeds: torch.Tensor,
                                     class_names: list,
                                     base_threshold: float = 0.7,
                                     strategy: str = "max",   # "max" ou "mean"
                                     min_keep: int = 2):
    """
    Filtragem robusta e determinística.
    - text_embeds: torch.Tensor [C, D], NÃO precisa estar em device específico.
    - strategy: "max" (remove se max_sim >= thresh) ou "mean" (remove se avg_sim >= thresh)
    - min_keep: garante pelo menos N classes mantidas (fallback seguro).
    Retorna: filtered_text_embeds (cpu float numpy), filtered_class_names, keep_indices (sorted)
    """
    if not torch.is_tensor(text_embeds):
        raise ValueError("text_embeds deve ser torch.Tensor")
    te = text_embeds.detach().cpu().float()
    C = te.shape[0]
    if C == 0:
        return te, [], []

    # normaliza por segurança
    te = te / (te.norm(dim=-1, keepdim=True) + 1e-12)

    sims = (te @ te.T).numpy()  # [C, C]
    np.fill_diagonal(sims, 0.0)
    upper = sims[np.triu_indices_from(sims, k=1)]
    mean_sim = float(np.mean(upper)) if upper.size > 0 else 0.0
    std_sim = float(np.std(upper)) if upper.size > 0 else 0.0

    # threshold adaptativo (opcional) — você pode simplesmente usar base_threshold
    threshold = min(base_threshold + 0.1 * std_sim, 0.85)

    print(f"   Filtering stats: mean={mean_sim:.4f} std={std_sim:.4f} base={base_threshold:.3f} -> threshold={threshold:.4f}")

    keep = []
    for i in range(C):
        row = sims[i]
        if strategy == "max":
            score = float(np.max(row))
        else:
            score = float(np.mean(row))
        if score < threshold:
            keep.append(i)

    # fallback: evita manter 0 classes
    if len(keep) < min_keep:
        print(f"   ⚠️ keep={len(keep)} < min_keep ({min_keep}). Mantendo top-{min_keep} menos similares (fallback).")
        # escolhe as min_keep classes com menor média de similaridade
        mean_per = sims.mean(axis=1)
        keep = list(np.argsort(mean_per)[:min_keep])

    keep = sorted(keep)
    filtered_text_embeds = te[keep].numpy()  # retorna numpy para checagens fáceis
    filtered_class_names = [class_names[i] for i in keep]

    print(f"   ✅ Mantidas {len(keep)}/{C} classes: {filtered_class_names[:10]}{'...' if len(keep)>10 else ''}")
    return filtered_text_embeds, filtered_class_names, keep



def evaluate_with_filtering_safe(dataset_name, image_embeds, image_paths, descriptions, model,
                                 base_threshold=0.7, strategy="max"):
    # infer classes
    labels, class_names = infer_classes_from_paths(image_paths)
    labels = np.array(labels, dtype=np.int64)
    C_original = len(class_names)
    N_images = len(labels)
    print(f"\n🔍 {dataset_name}: classes={C_original}, images={N_images}")

    # build class_texts aggregated (mantendo flatten como no seu código)
    class_descriptors = {}
    for cn in class_names:
        class_code = cn.split('-')[0] if '-' in cn else cn
        class_descriptors[cn] = []
        if descriptions:
            for k, v in descriptions.items():
                if class_code in k or cn in k:
                    class_descriptors[cn].append(v)
    # flatten each
    all_texts = []
    text_to_class_idx = []
    for idx, cn in enumerate(class_names):
        descs = class_descriptors.get(cn, [])
        flat = []
        for d in descs:
            if isinstance(d, list):
                flat.extend(d)
            else:
                flat.append(d)
        if not flat:
            flat = [f"a photo of a {cn.replace('_',' ')}"]
        all_texts.extend(flat)
        text_to_class_idx.extend([idx]*len(flat))

    # tokeniza e gera text embeddings (garantir same dtype/device)
    text_tokens = clip.tokenize(all_texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        all_text_embeds = model.encode_text(text_tokens)
        all_text_embeds = all_text_embeds / (all_text_embeds.norm(dim=-1, keepdim=True) + 1e-12)

    # agrupa por classe (média)
    final_text_embeds = []
    for i in range(len(class_names)):
        ids = [j for j, c in enumerate(text_to_class_idx) if c == i]
        emb = all_text_embeds[ids].mean(dim=0)
        emb = emb / (emb.norm() + 1e-12)
        final_text_embeds.append(emb)
    text_embeds = torch.stack(final_text_embeds)  # [C, D]

    # DEBUG básico
    print("   DEBUG shapes: image_embeds", image_embeds.shape, "text_embeds", text_embeds.shape)

    # aplica filtro seguro
    filtered_text_embeds_np, filtered_class_names, keep_indices = apply_comparative_filtering_safe(
        text_embeds, class_names, 
        base_threshold=0.7,
        strategy="mean", 
        min_keep=max(5, len(class_names)//3)
    )

    # checagens de integridade
    if len(keep_indices) == 0:
        print("   ⚠️ nenhuma classe mantida após filtro (improvável). Abortando.")
        return None, None, None, None

    # filtra imagens
    valid_mask = np.isin(labels, keep_indices)
    filtered_image_embeds = image_embeds[valid_mask]
    filtered_labels = labels[valid_mask]

    # remapeia labels de forma segura e verifica
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    try:
        filtered_labels_mapped = np.array([old_to_new[int(l)] for l in filtered_labels], dtype=np.int64)
    except KeyError as e:
        print("   ❌ Erro no remapeamento de labels:", e)
        return None, None, None, None

    # sanity checks
    uniq = np.unique(filtered_labels_mapped)
    if len(uniq) != len(filtered_class_names):
        print(f"   ⚠️ Atenção: classes mantidas={len(filtered_class_names)} mas labels únicos após remapeamento={len(uniq)}")
    print("   label dist (top 5):", Counter(filtered_labels_mapped).most_common(5))

    if filtered_image_embeds.shape[0] == 0:
        print("   ⚠️ 0 imagens após filtragem — abortando.")
        return None, None, None, None

    # move tensors to device and float
    X = filtered_image_embeds.to(DEVICE).float()
    Y = torch.from_numpy(filtered_text_embeds_np).to(DEVICE).float()

    sims = X @ Y.T
    preds = sims.argmax(dim=-1).cpu().numpy()

    # quick leak check: pred distribution vs label distribution
    print("   preds dist (top 5):", Counter(preds).most_common(5))

    acc = accuracy_score(filtered_labels_mapped, preds)
    print(f"   ✅ accuracy (filtered) = {acc:.4f}  (images kept: {X.shape[0]})")

    # debug sample
    sample_idx = np.random.choice(range(X.shape[0]), size=min(6, X.shape[0]), replace=False)
    print("   Exemplos: (true -> pred)")
    for i in sample_idx:
        t = filtered_class_names[filtered_labels_mapped[i]]
        p = filtered_class_names[preds[i]]
        ok = "✓" if t == p else "✗"
        print(f"     {ok} {t[:30]:30s} -> {p[:30]:30s}")

    return acc, C_original, len(filtered_class_names), X.shape[0]


# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Avaliação Zero-Shot com Comparative Filtering")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🔍 Threshold de similaridade: {SIMILARITY_THRESHOLD}\n")
    
    # Carrega modelo CLIP
    print("🔄 Carregando modelo CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "comparative filtering",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "total_datasets": len(DATASETS),
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            image_embeds, image_paths = load_embeddings(dataset_name)
            if image_embeds is None:
                print(f"⚠️  Pulando {dataset_name} (sem embeddings)")
                continue
            
            descriptions = load_descriptions(dataset_name)
            
            acc, num_orig, num_filtered, num_images = evaluate_with_filtering_safe(
                dataset_name, image_embeds, image_paths, descriptions, model
            )
            
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes_original": num_orig,
                "num_classes_filtered": num_filtered,
                "num_images": num_images
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_comparative_filtering.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ Avaliação finalizada com sucesso.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()