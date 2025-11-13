"""
Avaliação Zero-Shot com CLIP usando descriptions detalhadas.
IMPLEMENTAÇÃO SINGLE-DESCRIPTOR (PADRÃO): Classifica usando APENAS
uma descrição por classe (string), alinhando-se ao formato do seu JSON atual.
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
import traceback

# ============================
# CONFIGURAÇÕES
# ============================

def load_datasets_from_summary(summary_path: Path) -> dict:
    """Carrega configuração de datasets do summary.json"""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar summary: {e}")
        return {}

    datasets = {}
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    elif isinstance(summary, dict) and 'datasets' in summary:
        for name, path in summary['datasets'].items():
            datasets[name] = path

    return datasets


path_string = "all_zero-shot_results/"
SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS = load_datasets_from_summary(SUMMARY_PATH)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = path_string + "results_zero_shot_description_single_final"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def sanitize_class_name(class_name: str) -> str:
    """Função auxiliar para limpar nomes de classe."""
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()


def load_descriptions(dataset_name):
    """Carrega descrições do dataset."""
    path = os.path.join("descriptors_local_llm", f"{dataset_name}_local_descriptors.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Descritores não encontrados em {path}.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Erro ao ler JSON de descritores em {path}")
        return {}


def load_embeddings_and_generate_text(dataset_name, descriptions, model):
    """
    Carrega embeddings de imagem e gera embeddings de texto de UMA string por classe.
    """
    embedding_path = os.path.join("embeddings", f"{dataset_name}.pt")

    if not os.path.exists(embedding_path):
        print(f"⚠️  Embeddings de imagem não encontrados: {embedding_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings de imagem: {embedding_path}")
    embeddings_data = torch.load(embedding_path, weights_only=False, map_location='cpu')

    image_embeds = embeddings_data.get('image_embeddings')
    image_paths = embeddings_data.get('image_paths')

    if image_embeds is None or image_paths is None:
        print("❌ Chaves 'image_embeddings' ou 'image_paths' não encontradas no arquivo .pt.")
        return None, None, None, None

    labels = []
    unique_class_names_from_paths = sorted(list(set(Path(p).parts[-2] for p in image_paths if len(Path(p).parts) >= 2)))
    class_to_idx_sorted = {name: idx for idx, name in enumerate(unique_class_names_from_paths)}
    class_names = unique_class_names_from_paths

    for path in image_paths:
        class_name = Path(path).parts[-2] if len(Path(path).parts) >= 2 else "unknown"
        labels.append(class_to_idx_sorted.get(class_name, -1))

    labels = np.array(labels)

    print(f"   Total de imagens: {len(labels)} | Classes: {len(class_names)}")

    all_class_texts = []
    for class_name in class_names:
        description_data = descriptions.get(class_name)

        if isinstance(description_data, str):
            text_to_use = description_data
        elif isinstance(description_data, list) and description_data:
            text_to_use = description_data[0]
            print(f"⚠️ Aviso: Classe '{class_name}' tem lista de templates. Usando APENAS o primeiro.")
        else:
            text_to_use = f"a photo of a {sanitize_class_name(class_name)}"
            print(f"❌ Erro de formato para '{class_name}'. Usando template padrão.")

        all_class_texts.append(text_to_use)

    print(f"\n📝 Gerando text embeddings para {len(all_class_texts)} classes...")

    text_tokens = clip.tokenize(all_class_texts, truncate=True).to(DEVICE)

    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    print(f"✅ Text embeddings prontos! Shape: {text_embeds.shape}")

    return image_embeds.cpu(), text_embeds.cpu(), labels, class_names


def evaluate_zero_shot_single_descriptor(image_embeds, text_embeds, labels):
    """Calcula acurácia zero-shot com Single-Descriptor (comparação direta)."""
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()

    final_similarity_matrix = image_embeds @ text_embeds.T
    preds = final_similarity_matrix.argmax(dim=-1).numpy()

    acc = accuracy_score(labels, preds)
    return acc, preds


def plot_confusion_matrix(labels, preds, class_names, output_path):
    """Gera e salva matriz de confusão"""
    try:
        cm = confusion_matrix(labels, preds, normalize='true')
    except ValueError as e:
        print(f"❌ Erro ao calcular matriz de confusão: {e}")
        return

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='viridis', aspect='auto')
    plt.title("CLIP Zero-Shot with Single Description (Final)", fontsize=14)
    plt.colorbar()

    fontsize = max(3, 12 - len(class_names) // 10)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=fontsize)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Avaliação Zero-Shot com CLIP + Descriptions (SINGLE-DESCRIPTOR FINAL)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")

    print("🔄 Carregando modelo CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "CLIP with Single Description (Final Attempt)",
        "total_datasets": len(DATASETS),
        "successful": 0,
        "failed": 0,
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"📊 Avaliando dataset: {dataset_name}")
        print(f"{'='*70}")

        try:
            descriptions = load_descriptions(dataset_name)
            if not descriptions:
                print(f"⏭️  Não há descrições para {dataset_name}. Pulando.")
                summary["failed"] += 1
                continue

            print(f"✅ Descriptions carregadas: {len(descriptions)} classes")

            image_embeds, text_embeds, labels, class_names = load_embeddings_and_generate_text(dataset_name, descriptions, model)

            if image_embeds is None:
                print(f"⏭️  Pulando {dataset_name} devido à falha no carregamento/parsing.")
                summary["failed"] += 1
                continue

            acc, preds = evaluate_zero_shot_single_descriptor(image_embeds, text_embeds, labels)
            print(f"\n✅ Acurácia zero-shot (CLIP Single-Descriptor): {acc:.4f}")

            plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_cm_single_description_final.png")
            plot_confusion_matrix(labels, preds, class_names, output_path=plot_path)

            summary["successful"] += 1
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes": len(text_embeds),
                "num_images": len(labels),
                "confusion_matrix_plot": plot_path
            }

        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            traceback.print_exc()
            summary["failed"] += 1
            continue

    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_single_description_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ {summary['successful']} datasets processados com sucesso.")
    print(f"❌ {summary['failed']} falharam.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
