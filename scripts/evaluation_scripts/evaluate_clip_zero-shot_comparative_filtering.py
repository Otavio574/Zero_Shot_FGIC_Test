"""
Avaliação Zero-Shot com Comparative-CLIP + Filtering Process.
Usa descritores comparativos FILTRADOS (few-shot).

Baseado no paper "Enhancing Visual Classification using Comparative Descriptors" (WACV 2025)

Este script foi modificado para rodar automaticamente todos os modelos
especificados em ALL_MODELS e ajustado para receber os embeddings adequadamente
(buscando pelo nome do modelo).
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
import sys

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings" 
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_comparative_clip_filtered"
DESCRIPTOR_DIR = BASE_DIR / "descriptors_comparative_filtering"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Lista de modelos para iterar.
ALL_MODELS = [
    'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
    'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD SUMMARY
# ============================================================

def load_datasets_from_summary(path: Path):
    """
    Carrega a lista de datasets e seus caminhos a partir do arquivo summary.json.
    """
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

    os.makedirs(DESCRIPTOR_DIR, exist_ok=True) # Garantir que o diretório existe

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
        # Normalização de cada embedding
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Média dos descritores (ou único embedding se vanilla)
    final = text_embeds.mean(dim=0)
    final = final / final.norm() # Normalização final do vetor médio
    
    return final.cpu()


# ============================================================
# CARREGA EMBEDDINGS + TEXT EMBEDDINGS FILTRADOS
# ============================================================

def load_embeddings_and_generate_filtered_text(dataset_name, model_name, descriptors, 
                                               model, clip_library):
    """
    Carrega embeddings de imagem (específicos do modelo) e gera text embeddings 
    com descritores filtrados.
    """
    # 🚨 AJUSTE PRINCIPAL: O caminho deve incluir o nome do modelo para garantir
    # o carregamento de embeddings com a dimensão correta.
    model_safe_name = model_name.replace('/', '-')
    emb_path = EMBED_DIR / f"{dataset_name}_{model_safe_name}.pt"

    if not emb_path.exists():
        print(f"⚠️  Embeddings de imagem não encontrados para {model_name}: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings de imagem: {emb_path}")
    # Carrega sempre para CPU, moveremos para o DEVICE apenas na avaliação.
    data = torch.load(emb_path, map_location="cpu")

    image_embeds = data.get("image_embeddings")
    image_paths = data.get("image_paths")

    if image_embeds is None or image_paths is None:
        print("❌ .pt inválido, faltando chaves 'image_embeddings' ou 'image_paths'.")
        return None, None, None, None

    # Normalização dos embeddings de imagem
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
        # Garante que os descritores são minúsculos (boa prática para prompts)
        filtered_descs = [d.lower() for d in descriptors.get(cls, [])]
        
        if not filtered_descs:
            vanilla_count += 1
        
        emb = get_text_embedding_filtered(
            cls, filtered_descs, model, clip_library, DEVICE
        )
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)

    print(f"✅ Text embeddings filtrados: {text_embeds.shape}")
    print(f"   Classes usando vanilla prompt: {vanilla_count}/{len(class_names)}")

    # Retorna para CPU, o movimento para DEVICE será feito na avaliação
    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT EVALUATION
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels, device):
    """
    Avaliação zero-shot via similaridade coseno.
    Os embeddings são movidos para o DEVICE aqui.
    """
    # Mover para o dispositivo para o cálculo eficiente da matriz (GPU)
    img_embeds = img_embeds.to(device).float()
    text_embeds = text_embeds.to(device).float()
    
    # Produto escalar (similaridade coseno, pois ambos estão normalizados)
    # A transpotação é feita implicitamente pelo operador @
    sims = img_embeds @ text_embeds.T
    
    # Predição Top-1
    preds = sims.argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(labels, preds)
    
    # Predição Top-5
    top5_preds = sims.topk(5, dim=-1).indices.cpu().numpy()
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc, preds


# ============================================================
# MAIN LOOP (Alterada para rodar todos os modelos)
# ============================================================

def main():
    print("🎯 Comparative-CLIP + Filtering Process Evaluation (Few-Shot)")
    print(f"💻 Device: {DEVICE}\n")

    # Garante que o diretório de resultados existe
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # === LOOP PRINCIPAL SOBRE OS MODELOS ===
    for model_name in ALL_MODELS:
        print("\n" + "#" * 70)
        print(f"🚀 INICIANDO AVALIAÇÃO PARA MODELO: {model_name}")
        print("#" * 70)
        
        try:
            # 1. Carregar o modelo específico da iteração
            print(f"🔄 Carregando CLIP ({model_name})...")
            # O nome do modelo deve ser tratado para ser seguro em caminhos de arquivo
            model_safe_name = model_name.replace('/', '-') 
            
            # Carrega o modelo CLIP
            model, _ = clip.load(model_name, device=DEVICE)
            model.eval()
            print("✅ Modelo carregado!\n")

            summary = {}
            total_avg_descriptors = 0
            dataset_count = 0

            for dataset_name, dataset_path in DATASETS.items():

                print("=" * 70)
                print(f"📊 Avaliando {dataset_name} com Comparative-CLIP (Filtered) e modelo {model_name}")
                print("=" * 70)

                try:
                    # 2. Carrega descritores filtrados
                    filtered_descriptors = load_filtered_descriptors(dataset_name)
                    
                    if not filtered_descriptors:
                        print("⏭️  Sem descritores filtrados → ignorado.")
                        continue

                    # 3. Carrega embeddings e gera text embeddings (Passando model_name)
                    image_embeds, text_embeds, labels, class_names = \
                        load_embeddings_and_generate_filtered_text(
                            dataset_name, model_name, filtered_descriptors, model, clip
                        )

                    if image_embeds is None:
                        continue

                    # 4. Avaliação zero-shot (Passando DEVICE)
                    acc, top5_acc, preds = evaluate_zero_shot(
                        image_embeds, text_embeds, labels, DEVICE
                    )
                    
                    # Calcula descritores médios para o resumo
                    current_avg_desc = sum(
                        len(filtered_descriptors.get(cls, [])) 
                        for cls in class_names
                    ) / len(class_names) if class_names else 0
                    
                    total_avg_descriptors += current_avg_desc
                    dataset_count += 1


                    print(f"\n🎯 Resultados Comparative-CLIP + Filtering para {dataset_name}:")
                    print(f"   Top-1 Accuracy: {acc:.4f}")
                    print(f"   Top-5 Accuracy: {top5_acc:.4f}")

                    summary[dataset_name] = {
                        "accuracy_top1": float(acc),
                        "accuracy_top5": float(top5_acc),
                        "num_classes": len(class_names),
                        "num_images": len(labels),
                        "model": model_name,
                        "method": "comparative_clip_filtered",
                        "template": "a photo of a {class}, {filtered_descriptor}",
                        "avg_descriptors_per_class": float(current_avg_desc)
                    }

                except Exception as e:
                    print(f"❌ Erro no dataset {dataset_name}: {e}")
                    # traceback.print_exc() # Descomente para ver o traceback completo

            # 5. Salvar resultados
            out_path = RESULTS_DIR / f"comparative_clip_filtered_results_{model_safe_name}.json"
            
            if summary:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)

                print("\n" + "=" * 70)
                print(f"📈 Resultados salvos em: {out_path}")
                print("=" * 70)
                
                # 6. Mostra resumo
                print(f"\n📊 RESUMO ({model_name} - Comparative-CLIP + Filtering):")
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
                print("\n⚠️  Nenhum resultado gerado com sucesso para este modelo.")


        except Exception as e:
            # Captura erro no carregamento do modelo (se houver)
            print(f"❌ Erro fatal durante a avaliação do modelo {model_name}: {e}")
            # traceback.print_exc() # Descomente para ver o traceback completo

    print("\n\n*** AVALIAÇÃO DE TODOS OS MODELOS CONCLUÍDA ***")

if __name__ == "__main__":
    main()
