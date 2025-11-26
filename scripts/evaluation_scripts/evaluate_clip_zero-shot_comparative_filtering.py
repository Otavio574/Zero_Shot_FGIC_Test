"""
Avaliação Zero-Shot com Comparative-CLIP + Filtering Process.
Usa descritores comparativos FILTRADOS (few-shot).

Baseado no paper "Enhancing Visual Classification using Comparative Descriptors" (WACV 2025)
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score
import traceback

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_comparative_clip_filtered"
DESCRIPTOR_DIR = BASE_DIR / "descriptors_comparative_filtering"


MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD SUMMARY
# ============================================================

def load_datasets_from_summary(path: Path):
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


DATASETS = load_datasets_from_summary(SUMMARY_PATH)


# ============================================================
# CARREGAR DESCRITORES FILTRADOS
# ============================================================

def load_filtered_descriptors(dataset_name: str):
    """
    Carrega descritores comparativos FILTRADOS.
    """
    path = DESCRIPTOR_DIR / f"{dataset_name}_comparative_filtered.json"

    if not path.exists():
        print(f"❌ Descritores filtrados não encontrados: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        descriptors = json.load(f)
    
    print(f"📂 Carregados descritores filtrados de: {path}")
    
    # Estatísticas
    total_descs = sum(len(descs) for descs in descriptors.values())
    classes_with_descs = sum(1 for descs in descriptors.values() if len(descs) > 0)
    classes_without_descs = len(descriptors) - classes_with_descs
    avg_per_class = total_descs / len(descriptors) if descriptors else 0
    
    print(f"   Classes: {len(descriptors)}")
    print(f"   Com descritores: {classes_with_descs}")
    print(f"   Sem descritores (usam vanilla): {classes_without_descs}")
    print(f"   Total descritores: {total_descs}")
    print(f"   Média/classe: {avg_per_class:.1f}")
    
    return descriptors


# ============================================================
# EMBEDDING COM DESCRITORES FILTRADOS
# ============================================================

def get_text_embedding_filtered(class_name: str, filtered_descriptors: list, 
                                 model, clip_library, device):
    """
    Gera embeddings de texto usando descritores filtrados.
    
    - Se houver descritores: usa template "a photo of a {class}, {descriptor}"
    - Se lista vazia: usa vanilla prompt "a photo of a {class}"
    """
    
    class_readable = class_name.replace('_', ' ')
    
    # Se não há descritores filtrados, usa vanilla prompt
    if not filtered_descriptors or len(filtered_descriptors) == 0:
        texts = [f"a photo of a {class_readable}"]
    else:
        # Usa descritores filtrados
        texts = [
            f"a photo of a {class_readable}, {desc}" 
            for desc in filtered_descriptors
        ]
    
    # Tokeniza
    tokens = clip_library.tokenize(texts).to(device)
    
    # Encode
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Média dos descritores (ou único embedding se vanilla)
    final = text_embeds.mean(dim=0)
    final = final / final.norm()
    
    return final.cpu()


# ============================================================
# CARREGA EMBEDDINGS + TEXT EMBEDDINGS FILTRADOS
# ============================================================

def load_embeddings_and_generate_filtered_text(dataset_name, descriptors, 
                                                model, clip_library):
    """
    Carrega embeddings de imagem e gera text embeddings com descritores filtrados.
    """
    emb_path = EMBED_DIR / f"{dataset_name}.pt"

    if not emb_path.exists():
        print(f"⚠️  Embeddings de imagem não encontrados: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings de imagem: {emb_path}")
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    image_embeds = data.get("image_embeddings")
    image_paths = data.get("image_paths")

    if image_embeds is None or image_paths is None:
        print("❌ .pt inválido, faltando chaves")
        return None, None, None, None

    # Normalização
    image_embeds = image_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Extrai classes dos paths
    class_names = sorted(list(set(Path(p).parts[-2] for p in image_paths)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[Path(p).parts[-2]] for p in image_paths])

    print(f"   Total imagens: {len(labels)} | Classes: {len(class_names)}")

    # Gera text embeddings com descritores filtrados
    print("\n📝 Gerando text embeddings com descritores filtrados...")

    text_embeds_list = []
    vanilla_count = 0
    
    for cls in tqdm(class_names, desc="   Classes"):
        filtered_descs = descriptors.get(cls, [])
        
        if not filtered_descs:
            vanilla_count += 1
        
        emb = get_text_embedding_filtered(
            cls, filtered_descs, model, clip_library, DEVICE
        )
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)

    print(f"✅ Text embeddings filtrados: {text_embeds.shape}")
    print(f"   Classes usando vanilla prompt: {vanilla_count}/{len(class_names)}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT EVALUATION
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels):
    """
    Avaliação zero-shot via similaridade coseno.
    """
    sims = img_embeds @ text_embeds.T
    preds = sims.argmax(dim=-1).numpy()
    
    # Top-1 accuracy
    acc = accuracy_score(labels, preds)
    
    # Top-5 accuracy
    top5_preds = sims.topk(5, dim=-1).indices.numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc, preds


# ============================================================
# MAIN
# ============================================================

def main():
    print("🎯 Comparative-CLIP + Filtering Process Evaluation (Few-Shot)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")

    print("🔄 Carregando CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ Modelo carregado!\n")

    summary = {}

    for dataset_name, dataset_path in DATASETS.items():

        print("=" * 70)
        print(f"📊 Avaliando {dataset_name} com Comparative-CLIP (Filtered)")
        print("=" * 70)

        try:
            # Carrega descritores filtrados
            filtered_descriptors = load_filtered_descriptors(dataset_name)
            
            if not filtered_descriptors:
                print("⏭️  Sem descritores filtrados → ignorado.")
                continue

            # Carrega embeddings e gera text embeddings
            image_embeds, text_embeds, labels, class_names = \
                load_embeddings_and_generate_filtered_text(
                    dataset_name, filtered_descriptors, model, clip
                )

            if image_embeds is None:
                continue

            # Avaliação zero-shot
            acc, top5_acc, preds = evaluate_zero_shot(
                image_embeds.float(), text_embeds.float(), labels
            )

            print(f"\n🎯 Resultados:")
            print(f"   Top-1 Accuracy: {acc:.4f}")
            print(f"   Top-5 Accuracy: {top5_acc:.4f}")

            summary[dataset_name] = {
                "accuracy_top1": float(acc),
                "accuracy_top5": float(top5_acc),
                "num_classes": len(class_names),
                "num_images": len(labels),
                "method": "comparative_clip_filtered",
            }

        except Exception as e:
            print(f"❌ Erro no dataset {dataset_name}: {e}")
            traceback.print_exc()

    # Salvar resultados
    out_path = RESULTS_DIR / "comparative_clip_filtered_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📈 Resultados salvos em:", out_path)
    print("=" * 70)
    
    # Mostra resumo
    if summary:
        print("\n📊 RESUMO (Comparative-CLIP + Filtering):")
        print(f"{'Dataset':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Classes':<10}")
        print("-" * 70)
        for ds, results in summary.items():
            print(f"{ds:<20} {results['accuracy_top1']:<12.4f} "
                  f"{results['accuracy_top5']:<12.4f} {results['num_classes']:<10}")


if __name__ == "__main__":
    main()