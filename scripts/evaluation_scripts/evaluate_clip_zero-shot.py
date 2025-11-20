"""
Avaliação Zero-Shot com CLIP Baseline (Vanilla CLIP).
Usa apenas o template padrão: "a photo of a {class}"

Baseado no paper original do CLIP (Radford et al., 2021):
- Sem descritores externos
- Template simples
- Zero-shot puro
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

SUMMARY_PATH = Path("outputs/analysis/summary.json")
EMBED_DIR = Path("embeddings_openai")
RESULTS_DIR = Path("all_zero-shot_results/results_clip_baseline")

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
# EMBEDDING COM TEMPLATE VANILLA CLIP
# ============================================================

def get_text_embedding_baseline(class_name: str, model, clip_library, device):
    """
    Gera embedding de texto usando apenas o template vanilla CLIP.
    
    Template: "a photo of a {class}"
    
    Sem descritores adicionais - zero-shot puro!
    """
    
    # Nome legível da classe
    class_readable = class_name.replace('_', ' ')
    
    # 🎯 TEMPLATE VANILLA CLIP (paper original)
    text = f"a photo of a {class_readable}"
    
    # Tokeniza
    tokens = clip_library.tokenize([text]).to(device)
    
    # Encode
    with torch.no_grad():
        text_embed = model.encode_text(tokens)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    
    return text_embed.squeeze(0).cpu()


# ============================================================
# CARREGA EMBEDDINGS + GERA TEXT EMBEDDINGS BASELINE
# ============================================================

def load_embeddings_and_generate_baseline_text(dataset_name, model, clip_library):
    """
    Carrega embeddings de imagem e gera text embeddings vanilla CLIP.
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

    # Gera text embeddings vanilla CLIP (sem descritores)
    print("\n📝 Gerando text embeddings vanilla CLIP...")
    print('   Template: "a photo of a {class}"')

    text_embeds_list = []
    for cls in tqdm(class_names, desc="   Classes"):
        emb = get_text_embedding_baseline(cls, model, clip_library, DEVICE)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)  # [num_classes, 512]

    print(f"✅ Text embeddings baseline: {text_embeds.shape}")

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
    print("🎯 CLIP Baseline Zero-Shot Evaluation (Vanilla CLIP)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print('📝 Template: "a photo of a {class}"\n')

    print("🔄 Carregando CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ Modelo carregado!\n")

    summary = {}

    for dataset_name, dataset_path in DATASETS.items():

        print("=" * 70)
        print(f"📊 Avaliando {dataset_name} com CLIP Baseline")
        print("=" * 70)

        try:
            # Carrega embeddings e gera text embeddings baseline
            image_embeds, text_embeds, labels, class_names = \
                load_embeddings_and_generate_baseline_text(
                    dataset_name, model, clip
                )

            if image_embeds is None:
                continue

            # Avaliação zero-shot
            acc, top5_acc, preds = evaluate_zero_shot(
                image_embeds.float(), text_embeds.float(), labels
            )

            print(f"\n🎯 Resultados CLIP Baseline:")
            print(f"   Top-1 Accuracy: {acc:.4f}")
            print(f"   Top-5 Accuracy: {top5_acc:.4f}")

            summary[dataset_name] = {
                "accuracy_top1": float(acc),
                "accuracy_top5": float(top5_acc),
                "num_classes": len(class_names),
                "num_images": len(labels),
                "method": "vanilla_clip",
                "template": "a photo of a {class}"
            }

        except Exception as e:
            print(f"❌ Erro no dataset {dataset_name}: {e}")
            traceback.print_exc()

    # Salvar resultados
    out_path = RESULTS_DIR / "clip_baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📈 Resultados salvos em:", out_path)
    print("=" * 70)
    
    # Mostra resumo
    if summary:
        print("\n📊 RESUMO DOS RESULTADOS (CLIP Baseline):")
        print(f"{'Dataset':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Classes':<10}")
        print("-" * 70)
        for ds, results in summary.items():
            print(f"{ds:<20} {results['accuracy_top1']:<12.4f} "
                  f"{results['accuracy_top5']:<12.4f} {results['num_classes']:<10}")
        
        # Média geral
        avg_top1 = np.mean([r['accuracy_top1'] for r in summary.values()])
        avg_top5 = np.mean([r['accuracy_top5'] for r in summary.values()])
        print("-" * 70)
        print(f"{'MÉDIA':<20} {avg_top1:<12.4f} {avg_top5:<12.4f}")


if __name__ == "__main__":
    main()