"""
Avaliação Zero-Shot com CLIP usando MULTI-DESCRIPTORS.
Versão rigorosamente compatível com o paper (DCLIP-style):
- Cada classe pode ter 1 ou vários descritores.
- Embeddings médios por classe (normalizados).
- Similaridade imagem–texto com CLIP ViT-B/32.
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import traceback

# ============================================================
# CONFIG
# ============================================================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
EMBED_DIR = Path("embeddings_openai")
DESCRIPTOR_DIR = Path("descriptors_dclip")
RESULTS_DIR = Path("all_zero-shot_results/results_zero_shot_description")

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
# SANITIZAR NOMES (compatível com embeddings salvos)
# ============================================================

def sanitize_class_name(name: str) -> str:
    parts = name.split(".", 1)
    if len(parts) == 2 and parts[0].isdigit():
        name = parts[1]
    return name.replace("_", " ").replace("-", " ").lower().strip()


# ============================================================
# CARREGAR DESCRITORES
# ============================================================

def load_descriptions(dataset_name: str):
    path = DESCRIPTOR_DIR / f"{dataset_name}_dclip.json"

    if not path.exists():
        print(f"❌ Descriptors não encontrados: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MULTI-DESCRIPTOR → EMBEDDING CLIP COM TEMPLATE
# ============================================================

def get_text_embedding_for_class(class_name, texts, model, clip_library, device):
    """
    Aceita lista de descritores e retorna o embedding médio (normalizado).
    USA TEMPLATE: "a photo of a {class}, which {descriptor}"
    """
    
    # sempre vira lista
    if texts is None:
        texts = []
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = [str(texts)]

    # limpeza
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]

    # nome da classe legível
    class_readable = class_name.replace('_', ' ')

    # fallback ou template
    if len(texts) == 0:
        texts = [f"a photo of a {class_readable}"]
    else:
        # 🔥 TEMPLATE CRÍTICO (DCLIP-style):
        texts = [f"a photo of a {class_readable}, which {desc}" for desc in texts]

    # tokenizar com a LIB clip do OpenAI
    tokens = clip_library.tokenize(texts).to(device)

    # encode
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # média de descritores
    final = text_embeds.mean(dim=0)
    final = final / final.norm()

    return final.cpu()


# ============================================================
# CARREGA EMBEDDINGS + TEXT EMBEDDINGS
# ============================================================

def load_embeddings_and_generate_text(dataset_name, descriptions, model, clip_library):
    emb_path = EMBED_DIR / f"{dataset_name}.pt"

    if not emb_path.exists():
        print(f"⚠️  Embeddings não encontrados: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings: {emb_path}")
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    image_embeds = data.get("image_embeddings")
    image_paths = data.get("image_paths")

    if image_embeds is None or image_paths is None:
        print("❌ .pt inválido, faltando chaves")
        return None, None, None, None

    # garante float32 e normalização
    image_embeds = image_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    class_names = sorted(list(set(Path(p).parts[-2] for p in image_paths)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[Path(p).parts[-2]] for p in image_paths])

    print(f"   Total imagens: {len(labels)} | Classes: {len(class_names)}")
    
    # 🔍 DEBUG:
    print(f"\n🔍 DEBUG - Classes do embedding:")
    print(f"   Total: {len(class_names)}")
    print(f"   Primeiras 5: {class_names[:5]}")
    
    print(f"\n🔍 DEBUG - Verificando match com descritores:")
    matches = sum(1 for cls in class_names if cls in descriptions)
    print(f"   Classes que batem: {matches}/{len(class_names)}")

    # text embeddings multi-descriptor COM TEMPLATE
    print("📝 Gerando text embeddings...")

    text_embeds_list = []
    for cls in class_names:
        texts = descriptions.get(cls, [])
        emb = get_text_embedding_for_class(cls, texts, model, clip_library, DEVICE)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)  # [num_classes, 512]

    print(f"✅ Text embeddings: {text_embeds.shape}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT COM DEBUG
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels):
    sims = img_embeds @ text_embeds.T  # [N_imgs, N_classes]
    preds = sims.argmax(dim=-1).numpy()
    
    # 🔍 DEBUG:
    print(f"\n🔍 Similaridades:")
    print(f"   Min: {sims.min():.4f}, Max: {sims.max():.4f}")
    print(f"   Mean: {sims.mean():.4f}, Std: {sims.std():.4f}")
    
    # Top-5 accuracy
    top5_preds = sims.topk(5, dim=-1).indices.numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    print(f"   Top-5 Accuracy: {top5_acc:.4f}")
    
    acc = accuracy_score(labels, preds)
    return acc, preds


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 Zero-Shot Evaluation — CLIP + Multi Descriptors + Template")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")

    print("🔄 Carregando CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ Modelo carregado!\n")

    summary = {}

    for dataset_name, dataset_path in DATASETS.items():

        print("=" * 70)
        print(f"📊 Avaliando {dataset_name}")
        print("=" * 70)

        try:
            descriptions = load_descriptions(dataset_name)
            if not descriptions:
                print("⏭️  Sem descrições → ignorado.")
                continue

            image_embeds, text_embeds, labels, class_names = \
                load_embeddings_and_generate_text(dataset_name, descriptions, model, clip)

            if image_embeds is None:
                continue

            acc, preds = evaluate_zero_shot(image_embeds.float(), text_embeds.float(), labels)

            print(f"🎯 Accuracy Zero-Shot: {acc:.4f}")

            summary[dataset_name] = {
                "accuracy": float(acc),
                "num_classes": len(class_names),
                "num_images": len(labels),
            }

        except Exception as e:
            print(f"❌ Erro no dataset {dataset_name}: {e}")
            traceback.print_exc()

    # salvar resultados
    out_path = RESULTS_DIR / "zero_shot_results_multi_descriptor_with_template.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n📈 Resultados salvos em:", out_path)


if __name__ == "__main__":
    main()