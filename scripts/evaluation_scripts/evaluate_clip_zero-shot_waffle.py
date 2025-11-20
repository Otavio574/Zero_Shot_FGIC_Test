"""
Avaliação Zero-Shot com WaffleCLIP usando DESCRITORES ALEATÓRIOS.
Implementa o método do paper "Waffling around for Performance" (ICCV 2023):
- Cada classe recebe descritores ALEATÓRIOS (palavras + sequências de caracteres)
- Múltiplas execuções (reps) para calcular média e desvio padrão
- Template: "a photo of a {class}, {random_descriptor}"
"""

import os
import json
import torch
import numpy as np
import clip
import random
import string
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score
import traceback

# ============================================================
# CONFIG
# ============================================================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
EMBED_DIR = Path("embeddings_openai")
RESULTS_DIR = Path("all_zero-shot_results/results_waffle_clip")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parâmetros WaffleCLIP
WAFFLE_COUNT = 15  # número de PARES (palavra + sequência de caracteres)
REPS = 7  # número de repetições para média/desvio

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
# GERADORES DE DESCRITORES ALEATÓRIOS (WAFFLE)
# ============================================================

def generate_random_word_descriptors(count: int, seed: int = None) -> list:
    """
    Gera palavras aleatórias do vocabulário CLIP.
    Baseado no paper: usa palavras comuns do vocabulário.
    """
    if seed is not None:
        random.seed(seed)
    
    # Vocabulário simples de palavras comuns (você pode expandir)
    # No paper original, eles usam o vocabulário do CLIP, mas aqui uso uma lista fixa
    word_vocab = [
        "red", "blue", "green", "yellow", "large", "small", "round", "square",
        "bright", "dark", "smooth", "rough", "soft", "hard", "light", "heavy",
        "fast", "slow", "new", "old", "hot", "cold", "wet", "dry", "clean",
        "dirty", "full", "empty", "strong", "weak", "young", "ancient", "modern",
        "natural", "artificial", "wild", "domestic", "common", "rare", "simple",
        "complex", "quiet", "loud", "sweet", "bitter", "fresh", "stale", "wide",
        "narrow", "deep", "shallow", "high", "low", "thick", "thin", "long", "short"
    ]
    
    return [random.choice(word_vocab) for _ in range(count)]


def generate_random_char_descriptors(count: int, seed: int = None) -> list:
    """
    Gera sequências de caracteres aleatórias (ex: "aaaaa aaa").
    Baseado no paper: sequências de letras repetidas ou aleatórias.
    """
    if seed is not None:
        random.seed(seed)
    
    descriptors = []
    for _ in range(count):
        # Tipo 1: letras repetidas (ex: "aaaaa")
        if random.random() < 0.5:
            char = random.choice(string.ascii_lowercase)
            length = random.randint(4, 8)
            desc = char * length
        # Tipo 2: sequência aleatória (ex: "xkjdf")
        else:
            length = random.randint(4, 8)
            desc = ''.join(random.choices(string.ascii_lowercase, k=length))
        
        # Às vezes adiciona espaço e mais caracteres
        if random.random() < 0.3:
            desc += " " + random.choice(string.ascii_lowercase) * random.randint(2, 4)
        
        descriptors.append(desc)
    
    return descriptors


def generate_waffle_descriptors(count: int, seed: int = None) -> list:
    """
    Gera descritores WaffleCLIP: pares de (palavra aleatória + sequência de caracteres).
    Total = count * 2 descritores.
    """
    words = generate_random_word_descriptors(count, seed)
    chars = generate_random_char_descriptors(count, seed)
    
    # Intercala: palavra, chars, palavra, chars, ...
    all_descriptors = []
    for w, c in zip(words, chars):
        all_descriptors.append(w)
        all_descriptors.append(c)
    
    return all_descriptors


# ============================================================
# EMBEDDING COM DESCRITORES WAFFLE
# ============================================================

def get_text_embedding_waffle(class_name: str, waffle_descriptors: list, 
                               model, clip_library, device):
    """
    Cria embeddings de texto usando descritores WaffleCLIP.
    Template: "a photo of a {class}, {descriptor}"
    """
    class_readable = class_name.replace('_', ' ')
    
    # Constrói prompts com template WaffleCLIP
    texts = [f"a photo of a {class_readable}, {desc}" for desc in waffle_descriptors]
    
    # Tokeniza
    tokens = clip_library.tokenize(texts).to(device)
    
    # Encode
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Média dos descritores
    final = text_embeds.mean(dim=0)
    final = final / final.norm()
    
    return final.cpu()


# ============================================================
# CARREGA EMBEDDINGS + GERA TEXT EMBEDDINGS WAFFLE
# ============================================================

def load_embeddings_and_generate_waffle_text(dataset_name, model, clip_library, seed):
    """
    Carrega embeddings de imagem e gera text embeddings com descritores aleatórios.
    """
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

    # Normalização
    image_embeds = image_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Extrai classes
    class_names = sorted(list(set(Path(p).parts[-2] for p in image_paths)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[Path(p).parts[-2]] for p in image_paths])

    print(f"   Total imagens: {len(labels)} | Classes: {len(class_names)}")

    # 🎲 Gera descritores WAFFLE para cada classe
    print(f"🎲 Gerando descritores WaffleCLIP (seed={seed})...")
    
    text_embeds_list = []
    for cls in class_names:
        # Cada classe recebe SEUS PRÓPRIOS descritores aleatórios
        waffle_descs = generate_waffle_descriptors(WAFFLE_COUNT, seed)
        emb = get_text_embedding_waffle(cls, waffle_descs, model, clip_library, DEVICE)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)

    print(f"✅ Text embeddings WaffleCLIP: {text_embeds.shape}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT EVALUATION
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels):
    """
    Calcula acurácia zero-shot via similaridade coseno.
    """
    sims = img_embeds @ text_embeds.T  # [N_imgs, N_classes]
    preds = sims.argmax(dim=-1).numpy()
    
    acc = accuracy_score(labels, preds)
    
    # Top-5
    top5_preds = sims.topk(5, dim=-1).indices.numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc


# ============================================================
# MAIN COM MÚLTIPLAS REPETIÇÕES
# ============================================================

def main():
    print("🎲 WaffleCLIP Zero-Shot Evaluation - Random Descriptors")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🎲 Waffle Count: {WAFFLE_COUNT} pares (= {WAFFLE_COUNT*2} descritores)")
    print(f"🔁 Repetições: {REPS}\n")

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
            # Múltiplas repetições com seeds diferentes
            accuracies = []
            top5_accuracies = []
            
            for rep in range(REPS):
                print(f"\n🔁 Repetição {rep+1}/{REPS}")
                
                # Usa seed diferente para cada repetição
                seed = rep + 42  
                
                image_embeds, text_embeds, labels, class_names = \
                    load_embeddings_and_generate_waffle_text(
                        dataset_name, model, clip, seed
                    )

                if image_embeds is None:
                    break

                acc, top5_acc = evaluate_zero_shot(
                    image_embeds.float(), text_embeds.float(), labels
                )

                print(f"   🎯 Accuracy: {acc:.4f} | Top-5: {top5_acc:.4f}")
                
                accuracies.append(acc)
                top5_accuracies.append(top5_acc)
            
            if len(accuracies) > 0:
                # Calcula estatísticas
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                mean_top5 = np.mean(top5_accuracies)
                std_top5 = np.std(top5_accuracies)
                
                print(f"\n📊 RESULTADOS FINAIS:")
                print(f"   Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"   Top-5:    {mean_top5:.4f} ± {std_top5:.4f}")

                summary[dataset_name] = {
                    "accuracy_mean": float(mean_acc),
                    "accuracy_std": float(std_acc),
                    "top5_mean": float(mean_top5),
                    "top5_std": float(std_top5),
                    "num_classes": len(class_names),
                    "num_images": len(labels),
                    "waffle_count": WAFFLE_COUNT,
                    "reps": REPS
                }

        except Exception as e:
            print(f"❌ Erro no dataset {dataset_name}: {e}")
            traceback.print_exc()

    # Salvar resultados
    out_path = RESULTS_DIR / f"waffle_clip_results_count{WAFFLE_COUNT}_reps{REPS}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📈 Resultados salvos em:", out_path)
    print("=" * 70)


if __name__ == "__main__":
    main()