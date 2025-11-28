"""
Avaliação Zero-Shot com CLIP usando MULTI-DESCRIPTORS.
Versão rigorosamente compatível com o paper (DCLIP-style):
- Cada classe pode ter 1 ou vários descritores.
- Embeddings médios por classe (normalizados).

Este script foi modificado para rodar automaticamente todos os modelos
especificados em ALL_MODELS, carregando os embeddings de imagem
corretos (que contêm o nome do modelo) para garantir a compatibilidade
de dimensões.
"""

import os
import json
import torch
import numpy as np
import clip
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
import traceback
import sys
from typing import Dict, List, Any

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_description_clip"
DESCRIPTOR_DIR = BASE_DIR / "descriptors_dclip"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Lista de modelos para iterar.
ALL_MODELS: List[str] = [
    'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
    'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD SUMMARY
# ============================================================

def load_datasets_from_summary(path: Path) -> Dict[str, str]:
    """Carrega a lista de datasets do summary.json."""
    if not path.exists():
        print("❌ summary.json não encontrado!")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = {}
    if isinstance(data, list):
        for item in data:
            if "dataset" in item and "path" in item:
                datasets[item["dataset"]] = item["path"]

    return datasets


DATASETS: Dict[str, str] = load_datasets_from_summary(SUMMARY_PATH)


# ============================================================
# SANITIZAR NOMES (compatível com embeddings salvos)
# ============================================================

def sanitize_class_name(name: str) -> str:
    """Função de sanitização original, mantida por compatibilidade."""
    parts = name.split(".", 1)
    if len(parts) == 2 and parts[0].isdigit():
        name = parts[1]
    return name.replace("_", " ").replace("-", " ").lower().strip()


# ============================================================
# CARREGAR DESCRITORES
# ============================================================

def load_descriptions(dataset_name: str) -> Dict[str, Any]:
    """Carrega os descritores de texto para o dataset."""
    path = DESCRIPTOR_DIR / f"{dataset_name}_dclip.json"

    if not path.exists():
        # Apenas um aviso, o loop principal deve lidar com o pulo.
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MULTI-DESCRIPTOR → EMBEDDING CLIP COM TEMPLATE
# ============================================================

def get_text_embedding_for_class(class_name: str, texts: List[str], model, clip_library, device: str) -> torch.Tensor:
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

def load_embeddings_and_generate_text(dataset_name: str, model_name: str, descriptions: Dict[str, Any], model: Any, clip_library: Any, device: str):
    """
    Carrega embeddings de imagem (específicos do modelo) e gera text embeddings multi-descriptor.
    """
    # 📢 CORREÇÃO: Usa o nome do modelo no caminho do arquivo
    model_safe_name = model_name.replace('/', '-')
    emb_path = EMBED_DIR / f"{dataset_name}_{model_safe_name}.pt" 

    if not emb_path.exists():
        print(f"⚠️ Embeddings de imagem não encontrados para {model_name}: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings: {emb_path}")
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    # Verifica integridade
    loaded_clip_model = data.get("clip_model")
    if loaded_clip_model != model_name:
        print(f"❌ Erro de integridade: Arquivo '{emb_path.name}' foi gerado por '{loaded_clip_model}' mas está sendo usado com '{model_name}'.")

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

    print(f"   Total imagens: {len(labels)} | Classes: {len(class_names)}")
    
    # text embeddings multi-descriptor COM TEMPLATE
    print("📝 Gerando text embeddings...")

    text_embeds_list = []
    for cls in tqdm(class_names, desc="   Classes"):
        # Se a classe não tiver descritores, o `get_text_embedding_for_class` usa o fallback vanilla.
        texts = descriptions.get(cls, []) 
        emb = get_text_embedding_for_class(cls, texts, model, clip_library, device)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)

    print(f"✅ Text embeddings: {text_embeds.shape}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT COM DEBUG
# ============================================================

def evaluate_zero_shot(img_embeds: torch.Tensor, text_embeds: torch.Tensor, labels: np.ndarray):
    """
    Avaliação zero-shot via similaridade coseno.
    """
    # Move para o dispositivo antes do produto escalar
    img_embeds = img_embeds.to(DEVICE)
    text_embeds = text_embeds.to(DEVICE)
    
    # Produto escalar (similaridade coseno, pois ambos estão normalizados)
    # [N_imgs, D] @ [D, N_classes] -> [N_imgs, N_classes]
    sims = img_embeds @ text_embeds.T 
    preds = sims.argmax(dim=-1).cpu().numpy() # Move para CPU para usar numpy/sklearn

    # Top-5 accuracy
    top5 = sims.topk(5, dim=-1).indices.cpu().numpy()
    top5_acc = sum(labels[i] in top5[i] for i in range(len(labels))) / len(labels)

    acc = accuracy_score(labels, preds)
    
    # 🔍 DEBUG:
    print(f"\n🔍 Similaridades:")
    print(f"   Min: {sims.min():.4f}, Max: {sims.max():.4f}")
    print(f"   Mean: {sims.mean():.4f}, Std: {sims.std():.4f}")
    print(f"   Top-5 Accuracy: {top5_acc:.4f}")
    
    return acc, top5_acc, preds


# ============================================================
# MAIN LOOP (Alterada para rodar todos os modelos)
# ============================================================

def main():
    print("🚀 Zero-Shot Evaluation — CLIP + Multi Descriptors + Template")
    print(f"💻 Device: {DEVICE}\n")

    # === LOOP PRINCIPAL SOBRE OS MODELOS ===
    for model_name in ALL_MODELS:
        print("\n" + "#" * 70)
        print(f"🚀 INICIANDO AVALIAÇÃO PARA MODELO: {model_name}")
        print("#" * 70)
        
        try:
            # Carregar o modelo específico da iteração
            print(f"🔄 Carregando CLIP ({model_name})...")
            model, _ = clip.load(model_name, device=DEVICE)
            model.eval()
            print("✅ Modelo carregado!\n")

            summary = {}

            for dataset_name, dataset_path in DATASETS.items():

                print("=" * 70)
                print(f"📊 Avaliando {dataset_name} com {model_name}")
                print("=" * 70)

                try:
                    descriptions = load_descriptions(dataset_name)
                    if not descriptions:
                        print("⏭️  Sem descrições → ignorado.")
                        continue

                    # Carrega embeddings (usando o nome do modelo) e gera text embeddings
                    image_embeds, text_embeds, labels, class_names = \
                        load_embeddings_and_generate_text(
                            dataset_name, model_name, descriptions, model, clip, DEVICE
                        )

                    if image_embeds is None:
                        continue

                    acc1, acc5, preds = evaluate_zero_shot(
                        image_embeds.float(), text_embeds.float(), labels
                    )

                    print(f"\n🎯 Resultados Multi-Descriptor para {dataset_name}:")
                    print(f"   Top-1: {acc1:.4f} | Top-5: {acc5:.4f}")

                    summary[dataset_name] = {
                        "accuracy_top1": float(acc1),
                        "accuracy_top5": float(acc5),
                        "num_classes": len(class_names),
                        "num_images": len(labels),
                        "method": "description_clip",
                        "template": "a photo of a {class}, which {descriptor}",
                        "model_name": model_name
                    }

                except Exception as e:
                    print(f"❌ Erro no dataset {dataset_name}: {e}")
                    traceback.print_exc()

            # salvar resultados
            model_safe_name = model_name.replace('/', '-')
            out_path = RESULTS_DIR / f"description_clip_results_{model_safe_name}.json"
            
            if summary:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)

                print("\n" + "=" * 70)
                print(f"📈 Resultados salvos em: {out_path}")
                print("=" * 70)

                # Mostra resumo
                print(f"\n📊 RESUMO DOS RESULTADOS ({model_name}):")
                
                print(f"{'Dataset':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Classes':<10}")
                print("-" * 70)
                for ds, results in summary.items():
                    print(f"{ds:<20} {results['accuracy_top1']:<12.4f} "
                          f"{results['accuracy_top5']:<12.4f} {results['num_classes']:<10}")
                
                avg_top1 = np.mean([r['accuracy_top1'] for r in summary.values()]) if summary else 0
                avg_top5 = np.mean([r['accuracy_top5'] for r in summary.values()]) if summary else 0
                print("-" * 70)
                print(f"{'MÉDIA':<20} {avg_top1:<12.4f} {avg_top5:<12.4f}")
                print("-" * 70)
            else:
                print(f"\n⚠️ Nenhum resultado gerado com sucesso para o modelo {model_name}.")


        except Exception as e:
            # Captura erro no carregamento do modelo (se houver)
            print(f"❌ Erro fatal durante a avaliação do modelo {model_name}: {e}")
            traceback.print_exc()

    print("\n\n*** AVALIAÇÃO DE TODOS OS MODELOS CONCLUÍDA ***")


if __name__ == "__main__":
    main()