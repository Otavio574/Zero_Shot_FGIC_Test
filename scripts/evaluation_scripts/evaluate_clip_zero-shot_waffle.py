"""
Avaliação Zero-Shot com WaffleCLIP.
Usa embeddings pré-calculados e descriptors com texto "waffle" (aleatório).
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from pathlib import Path

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

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_waffle"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptions(dataset_name):
    """Carrega descriptions/waffle descriptors do dataset"""
    path = os.path.join("descriptors_waffle_clip_random", f"{dataset_name}_waffle.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"⚠️  Nenhum descriptor encontrado para {dataset_name}, usando genéricos.")
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


def evaluate_waffle(dataset_name, image_embeds, image_paths, descriptions, model):
    """Avaliação com WaffleCLIP descriptors (com agregação)"""
    
    print(f"\n🔍 Iniciando avaliação WaffleCLIP: {dataset_name}")
    labels, class_names = infer_classes_from_paths(image_paths)
    num_classes = len(class_names)
    
    print(f"   Classes detectadas: {num_classes}")
    print(f"   Total de imagens: {len(image_embeds)}")
    
    # ===== AGREGAÇÃO DE MÚLTIPLOS DESCRIPTORS =====
    class_descriptors = {}
    
    for class_name in class_names:
        class_code = class_name.split('-')[0] if '-' in class_name else class_name
        class_descriptors[class_name] = []
        
        if descriptions:
            for desc_key, desc_value in descriptions.items():
                if class_code in desc_key or class_name in desc_key:
                    class_descriptors[class_name].append(desc_value)
    
    # Estatísticas
    desc_counts = [len(descs) for descs in class_descriptors.values()]
    fallback_count = sum(1 for count in desc_counts if count == 0)
    
    if desc_counts:
        print(f"📊 Descriptors por classe:")
        print(f"   Total: {sum(desc_counts)} | Média: {np.mean(desc_counts):.1f} | Min: {min(desc_counts)} | Max: {max(desc_counts)}")
        print(f"   Classes com waffle: {num_classes - fallback_count}/{num_classes}")
        print(f"   Classes com fallback: {fallback_count}/{num_classes}")
    
    # Gera lista flat de todos os textos
    # Gera lista flat de todos os textos (normalizando valores list/str)
    all_texts = []
    text_to_class_idx = []

    for idx, class_name in enumerate(class_names):
        descs = class_descriptors.get(class_name, [])
        if not descs:
            descs = [f"a photo of a {class_name.replace('_', ' ')}"]

        # Normaliza: se algum descriptor for lista, estenda com os itens; se for string, adicione
        normalized_descs = []
        for d in descs:
            if isinstance(d, list):
                # estende com cada item da lista (filtrando não-strings)
                for sub in d:
                    if isinstance(sub, str):
                        normalized_descs.append(sub)
                    else:
                        normalized_descs.append(str(sub))
            elif isinstance(d, str):
                normalized_descs.append(d)
            else:
                # fallback: força para string (evita crash no tokenize)
                normalized_descs.append(str(d))

        # adiciona à lista final
        all_texts.extend(normalized_descs)
        text_to_class_idx.extend([idx] * len(normalized_descs))

    # DEBUG: valida se todos os itens são strings
    non_str_items = [(i, type(x), x) for i, x in enumerate(all_texts) if not isinstance(x, str)]
    if non_str_items:
        print("⚠️ WARNING: itens não-string encontrados em all_texts (serão convertidos):")
        for idx, t, val in non_str_items[:10]:
            print(f"   index {idx}: type={t} value={val}")
        # converte todos para string por segurança
        all_texts = [str(x) for x in all_texts]

    print(f"\n📝 Gerando embeddings para {len(all_texts)} descriptors...")

    # Gera embeddings de texto com CLIP
    # Caso haja muitos textos, talvez seja melhor tokenizar em batches (opcional)
    text_tokens = clip.tokenize(all_texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        all_text_embeds = model.encode_text(text_tokens)
        all_text_embeds /= all_text_embeds.norm(dim=-1, keepdim=True)

    
    # Agrupa por classe e faz MÉDIA
    print(f"🔄 Agregando embeddings por classe...")
    final_text_embeds = []
    
    for idx in range(len(class_names)):
        indices = [i for i, c_idx in enumerate(text_to_class_idx) if c_idx == idx]
        class_embeds = all_text_embeds[indices]
        avg_embed = class_embeds.mean(dim=0)
        avg_embed /= avg_embed.norm()
        final_text_embeds.append(avg_embed)
    
    text_embeds = torch.stack(final_text_embeds)
    
    # Avalia
    image_embeds_float = image_embeds.to(DEVICE).float()
    text_embeds_float = text_embeds.to(DEVICE).float()
    
    sims = image_embeds_float @ text_embeds_float.T
    preds = sims.argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(labels, preds)
    
    print(f"✅ Acurácia WaffleCLIP: {acc:.4f}")
    
    # Debug: mostra alguns exemplos de predições
    print(f"\n🔍 DEBUG - Primeiras 5 predições:")
    for i in range(min(5, len(labels))):
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        correct = "✓" if labels[i] == preds[i] else "✗"
        print(f"   {correct} True: {true_label[:30]:30s} | Pred: {pred_label[:30]:30s}")
    
    return acc, num_classes, len(image_embeds)


# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Avaliação Zero-Shot com WaffleCLIP")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")
    
    # Carrega modelo CLIP
    print("🔄 Carregando modelo CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "WaffleCLIP",
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
            
            acc, num_classes, num_images = evaluate_waffle(
                dataset_name, image_embeds, image_paths, descriptions, model
            )
            
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes": num_classes,
                "num_images": num_images
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_waffle.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ Avaliação finalizada com sucesso.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
    

