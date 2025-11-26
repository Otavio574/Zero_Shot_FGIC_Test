"""
Avaliação Zero-Shot com Comparative-CLIP.
Usa descritores comparativos que enfatizam diferenças entre classes similares.

Baseado no paper "Enhancing Visual Classification using Comparative Descriptors" (WACV 2025)

Template: "a photo of a {class}, {comparative_descriptor}"
Exemplo: "a photo of a Black-footed Albatross, larger wingspan typically 6-7 feet"
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
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_comparative_clip"
DESCRIPTOR_DIR = BASE_DIR / "descriptors_comparative"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
# CARREGAR DESCRITORES COMPARATIVOS
# ============================================================

def load_comparative_descriptions(dataset_name: str):
    """
    Carrega descritores comparativos do Comparative-CLIP.
    """
    path = DESCRIPTOR_DIR / f"{dataset_name}_comparative.json"

    if not path.exists():
        print(f"❌ Descritores comparativos não encontrados: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        descriptors = json.load(f)
    
    print(f"📂 Carregados descritores comparativos de: {path}")
    
    # Estatísticas
    total_descs = sum(len(descs) for descs in descriptors.values())
    avg_per_class = total_descs / len(descriptors) if descriptors else 0
    print(f"   Classes: {len(descriptors)} | Total descritores: {total_descs} | Média/classe: {avg_per_class:.1f}")
    
    return descriptors


# ============================================================
# EMBEDDING COM DESCRITORES COMPARATIVOS
# ============================================================

def get_text_embedding_comparative(class_name: str, comparative_descriptors: list, 
                                   model, clip_library, device):
    """
    Gera embeddings de texto usando descritores comparativos.
    
    Template Comparative-CLIP: "a photo of a {class}, {comparative_descriptor}"
    
    Os descritores já vêm formatados como comparações, ex:
    - "larger wingspan, typically 6-7 feet"
    - "darker plumage compared to Slaty-backed Gull"
    """
    
    # Nome legível da classe
    class_readable = class_name.replace('_', ' ')
    
    # Validação de entrada
    if comparative_descriptors is None:
        comparative_descriptors = []
    if isinstance(comparative_descriptors, str):
        comparative_descriptors = [comparative_descriptors]
    if not isinstance(comparative_descriptors, list):
        comparative_descriptors = [str(comparative_descriptors)]
    
    # Limpa descritores
    comparative_descriptors = [
        d.strip() for d in comparative_descriptors 
        if isinstance(d, str) and d.strip()
    ]
    
    # Fallback se não houver descritores
    if len(comparative_descriptors) == 0:
        texts = [f"a photo of a {class_readable}"]
    else:
        # 🔥 TEMPLATE COMPARATIVE-CLIP:
        # Os descritores comparativos já trazem a informação de diferença
        texts = [
            f"a photo of a {class_readable}, {desc}" 
            for desc in comparative_descriptors
        ]
    
    # Tokeniza com a biblioteca CLIP
    tokens = clip_library.tokenize(texts).to(device)
    
    # Encode
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Média dos descritores (ensemble)
    final = text_embeds.mean(dim=0)
    final = final / final.norm()
    
    return final.cpu()


# ============================================================
# CARREGA EMBEDDINGS + TEXT EMBEDDINGS COMPARATIVOS
# ============================================================

def load_embeddings_and_generate_comparative_text(dataset_name, descriptions, 
                                                   model, clip_library):
    """
    Carrega embeddings de imagem e gera text embeddings comparativos.
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
    
    # 🔍 DEBUG:
    print(f"\n🔍 DEBUG - Classes do embedding:")
    print(f"   Total: {len(class_names)}")
    print(f"   Primeiras 5: {class_names[:5]}")
    
    print(f"\n🔍 DEBUG - Verificando match com descritores comparativos:")
    matches = sum(1 for cls in class_names if cls in descriptions)
    print(f"   Classes que batem: {matches}/{len(class_names)}")
    
    if matches < len(class_names):
        missing = [cls for cls in class_names if cls not in descriptions]
        print(f"   ⚠️  Classes sem descritores: {missing[:5]}...")

    # Gera text embeddings com descritores comparativos
    print("\n📝 Gerando text embeddings comparativos...")

    text_embeds_list = []
    for cls in class_names:
        comparative_descs = descriptions.get(cls, [])
        emb = get_text_embedding_comparative(
            cls, comparative_descs, model, clip_library, DEVICE
        )
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)  # [num_classes, 512]

    print(f"✅ Text embeddings comparativos: {text_embeds.shape}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT EVALUATION
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels):
    """
    Avaliação zero-shot via similaridade coseno.
    """
    sims = img_embeds @ text_embeds.T  # [N_imgs, N_classes]
    preds = sims.argmax(dim=-1).numpy()
    
    # 🔍 DEBUG:
    print(f"\n🔍 Similaridades:")
    print(f"   Min: {sims.min():.4f}, Max: {sims.max():.4f}")
    print(f"   Mean: {sims.mean():.4f}, Std: {sims.std():.4f}")
    
    # Top-1 accuracy
    acc = accuracy_score(labels, preds)
    
    # Top-5 accuracy
    top5_preds = sims.topk(5, dim=-1).indices.numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    print(f"   Top-5 Accuracy: {top5_acc:.4f}")
    
    return acc, top5_acc, preds


# ============================================================
# MAIN
# ============================================================

def main():
    print("🎯 Comparative-CLIP Zero-Shot Evaluation")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")

    print("🔄 Carregando CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ Modelo carregado!\n")

    summary = {}

    for dataset_name, dataset_path in DATASETS.items():

        print("=" * 70)
        print(f"📊 Avaliando {dataset_name} com Comparative-CLIP")
        print("=" * 70)

        try:
            # Carrega descritores comparativos
            comparative_descriptions = load_comparative_descriptions(dataset_name)
            
            if not comparative_descriptions:
                print("⏭️  Sem descritores comparativos → ignorado.")
                continue

            # Carrega embeddings e gera text embeddings
            image_embeds, text_embeds, labels, class_names = \
                load_embeddings_and_generate_comparative_text(
                    dataset_name, comparative_descriptions, model, clip
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
                "avg_descriptors_per_class": sum(
                    len(comparative_descriptions.get(cls, [])) 
                    for cls in class_names
                ) / len(class_names)
            }

        except Exception as e:
            print(f"❌ Erro no dataset {dataset_name}: {e}")
            traceback.print_exc()

    # Salvar resultados
    out_path = RESULTS_DIR / "comparative_clip_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📈 Resultados salvos em:", out_path)
    print("=" * 70)
    
    # Mostra resumo
    if summary:
        print("\n📊 RESUMO DOS RESULTADOS:")
        print(f"{'Dataset':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Classes':<10}")
        print("-" * 70)
        for ds, results in summary.items():
            print(f"{ds:<20} {results['accuracy_top1']:<12.4f} "
                  f"{results['accuracy_top5']:<12.4f} {results['num_classes']:<10}")


if __name__ == "__main__":
    main()