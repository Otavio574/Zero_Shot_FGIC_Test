"""
Avaliação Zero-Shot com WaffleCLIP (sem geração de embeddings).
Agora os datasets são detectados automaticamente em datasets/.
"""

import os
import json
import torch
import numpy as np
import clip

from pathlib import Path
from sklearn.metrics import accuracy_score


# ======================================
# DETECÇÃO AUTOMÁTICA DE DATASETS
# ======================================

def detect_datasets(base_path="datasets"):
    """
    Cada subpasta em datasets/ é considerada um dataset.
    Exemplo:
        datasets/cifar10/
        datasets/aircraft/
    """
    base = Path(base_path)

    if not base.exists():
        raise RuntimeError(f"Pasta '{base_path}' não existe!")

    datasets = {}

    for item in base.iterdir():
        if item.is_dir():
            datasets[item.name] = str(item)

    print(f"📁 Detectados {len(datasets)} datasets:")
    for ds in datasets:
        print(f"  - {ds}")

    return datasets


DATASETS = detect_datasets()

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_waffle_test"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================
# FUNÇÕES AUXILIARES
# ======================================

def load_descriptions(dataset_name):
    """Carrega descriptions/waffle descriptors do dataset"""
    path = os.path.join("descriptors_waffle_clip_random", f"{dataset_name}_waffle.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"⚠️ Nenhum descriptor encontrado para {dataset_name}, usando fallback.")
    return {}


def load_embeddings(dataset_name):
    """Carrega embeddings pré-calculados (não gera novos!)"""
    emb_path = os.path.join("embeddings", f"{dataset_name}.pt")

    if not os.path.exists(emb_path):
        print(f"❌ Embeddings não encontrados: {emb_path}")
        return None, None

    data = torch.load(emb_path, weights_only=False)

    if isinstance(data, dict):
        return data["image_embeddings"], data.get("image_paths", None)

    return data, None


def infer_classes_from_paths(image_paths):
    class_names, labels = [], []
    class_to_idx = {}

    for path in image_paths:
        parts = Path(path).parts
        class_name = parts[-2] if len(parts) >= 2 else "unknown"

        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_names)
            class_names.append(class_name)

        labels.append(class_to_idx[class_name])

    return np.array(labels), class_names


# ======================================
# AVALIAÇÃO COM WAFFLECLIP
# ======================================

def evaluate_waffle(dataset_name, image_embeds, image_paths, descriptions, model):

    print(f"\n🔍 Iniciando avaliação WaffleCLIP: {dataset_name}")

    labels, class_names = infer_classes_from_paths(image_paths)
    num_classes = len(class_names)

    print(f"   Classes detectadas: {num_classes}")
    print(f"   Total de imagens: {len(image_embeds)}")

    # Agrupar descriptors
    class_descriptors = {c: [] for c in class_names}

    for class_name in class_names:
        class_code = class_name.split("-")[0]
        for key, value in descriptions.items():
            if class_code in key or class_name in key:
                class_descriptors[class_name].append(value)

    # Normalizar textos
    all_texts = []
    text_to_class = []

    for idx, class_name in enumerate(class_names):
        descs = class_descriptors[class_name]
        if not descs:
            descs = [f"a photo of a {class_name.replace('_', ' ')}"]

        normalized = []
        for d in descs:
            if isinstance(d, list):
                normalized.extend([str(x) for x in d])
            else:
                normalized.append(str(d))

        all_texts.extend(normalized)
        text_to_class.extend([idx] * len(normalized))

    # Tokenização
    text_tokens = clip.tokenize(all_texts, truncate=True).to(DEVICE)

    with torch.no_grad():
        all_text_embeds = model.encode_text(text_tokens)
        all_text_embeds /= all_text_embeds.norm(dim=-1, keepdim=True)

    # Média por classe
    final_embeds = []
    for idx in range(num_classes):
        inds = [i for i, c in enumerate(text_to_class) if c == idx]
        avg = all_text_embeds[inds].mean(dim=0)
        avg /= avg.norm()
        final_embeds.append(avg)

    text_embeds = torch.stack(final_embeds)

    # Similaridade
    image_embeds = image_embeds.to(DEVICE).float()
    text_embeds = text_embeds.to(DEVICE).float()

    sims = image_embeds @ text_embeds.T

    preds = sims.argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(labels, preds)

    print(f"✅ Acurácia: {acc:.4f}")

    return acc, num_classes, len(image_embeds)


# ======================================
# MAIN
# ======================================

def main():
    print(f"\n🚀 Avaliação Zero-Shot com WaffleCLIP")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")

    print("\n🔄 Carregando modelo CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!")

    results = {
        "model": MODEL_NAME,
        "method": "WaffleCLIP",
        "results": {}
    }

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"📊 Dataset: {ds_name}")
        print(f"{'='*60}")

        image_embeds, image_paths = load_embeddings(ds_name)
        if image_embeds is None:
            continue

        descriptions = load_descriptions(ds_name)

        acc, n_cls, n_imgs = evaluate_waffle(
            ds_name, image_embeds, image_paths, descriptions, model
        )

        results["results"][ds_name] = {
            "accuracy": float(acc),
            "num_classes": n_cls,
            "num_images": n_imgs
        }

    # Salvar
    out = os.path.join(RESULTS_DIR, "zero_shot_results_waffle.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n📈 Resultados salvos em {out}")
    print("✅ Finalizado.")


if __name__ == "__main__":
    main()
