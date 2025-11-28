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

# Define o diretório base (assumindo que o script está em BASE_DIR/scripts/analysis/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_comparative_clip"
DESCRIPTOR_DIR = BASE_DIR / "descriptors_comparative"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 📢 NOVO: Lista de modelos para iterar.
ALL_MODELS = [
    'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
    'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]

# Variável MODEL_NAME removida. O nome será definido no loop.
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
    print(f"   Classes: {len(descriptors)} | Total descritores: {total_descs} | Média/classe: {avg_per_class:.1f}")
    
    return descriptors


# ============================================================
# EMBEDDING COM DESCRITORES COMPARATIVOS
# ============================================================

def get_text_embedding_comparative(class_name: str, comparative_descriptors: list, 
                                   model, clip_library, device):
    """
    Gera embeddings de texto usando descritores comparativos.
    
    Template Comparative-CLIP: "a photo of a {class}, {comparative_descriptor}"
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
                                                 model, clip_library, model_name, device):
    """
    Carrega embeddings de imagem e gera text embeddings comparativos.
    """
    model_safe_name = model_name.replace("/", "-")
    emb_path = EMBED_DIR / f"{dataset_name}_{model_safe_name}.pt"

    if not emb_path.exists():
        print(f"⚠️  Embeddings de imagem não encontrados: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings de imagem: {emb_path}")
    # Nota: Assumimos que o arquivo .pt contém os embeddings do modelo correto,
    # embora o modelo_name não esteja no nome do arquivo aqui,
    # ele é usado no loop principal para carregar o CLIP e gerar os embeddings de texto.
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
    # Usamos Path(p).parts[-2] para pegar o nome da pasta (classe)
    class_names = sorted(list(set(Path(p).parts[-2] for p in image_paths)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[Path(p).parts[-2]] for p in image_paths])

    print(f"   Total imagens: {len(labels)} | Classes: {len(class_names)}")
    
    # 🔍 DEBUG:
    print(f"\n🔍 DEBUG - Classes do embedding:")
    print(f"   Total: {len(class_names)}")
    print(f"   Primeiras 5: {class_names[:5]}")
    
    print(f"\n🔍 DEBUG - Verificando match com descritores comparativos:")
    matches = sum(1 for cls in class_names if cls in descriptions)
    print(f"   Classes que batem: {matches}/{len(class_names)}")
    
    if matches < len(class_names):
        missing = [cls for cls in class_names if cls not in descriptions]
        print(f"   ⚠️  Classes sem descritores: {missing[:5]}...")

    # Gera text embeddings com descritores comparativos
    print("\n📝 Gerando text embeddings comparativos...")

    text_embeds_list = []
    for cls in tqdm(class_names, desc="Gerando Text Embeddings"):
        comparative_descs = descriptions.get(cls, [])
        emb = get_text_embedding_comparative(
            cls, comparative_descs, model, clip_library, device
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
    # Multiplicação de matrizes para obter similaridades
    sims = img_embeds @ text_embeds.T  # [N_imgs, N_classes]
    
    # Top-1: índice com a similaridade máxima
    preds = sims.argmax(dim=-1).numpy()
    
    # 🔍 DEBUG:
    print(f"\n🔍 Similaridades:")
    print(f"   Min: {sims.min():.4f}, Max: {sims.max():.4f}")
    print(f"   Mean: {sims.mean():.4f}, Std: {sims.std():.4f}")
    
    # Top-1 accuracy
    acc = accuracy_score(labels, preds)
    
    # Top-5 accuracy
    # Obtém os 5 maiores índices para cada imagem
    top5_preds = sims.topk(5, dim=-1).indices.numpy()
    
    # Verifica se o label verdadeiro está em qualquer um dos 5
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    print(f"   Top-5 Accuracy: {top5_acc:.4f}")
    
    return acc, top5_acc, preds


# ============================================================
# MAIN LOOP (Alterada para rodar todos os modelos)
# ============================================================

def main():
    print("🎯 Comparative-CLIP Zero-Shot Evaluation")
    print(f"💻 Device: {DEVICE}\n")

    # === LOOP PRINCIPAL SOBRE OS MODELOS ===
    for model_name in ALL_MODELS:
        print("\n" + "#" * 70)
        print(f"🚀 INICIANDO AVALIAÇÃO PARA MODELO: {model_name}")
        print("#" * 70)
        
        try:
            # Carregar o modelo específico da iteração
            print(f"🔄 Carregando CLIP ({model_name})...")
            # O processamento do preprocess não é usado aqui, apenas o modelo
            model, _ = clip.load(model_name, device=DEVICE) 
            model.eval() # Modo de avaliação
            print("✅ Modelo carregado!\n")

            summary = {}
            avg_descriptors = 0
            count_datasets = 0

            for dataset_name, dataset_path in DATASETS.items():

                print("=" * 70)
                print(f"📊 Avaliando {dataset_name} com Comparative-CLIP e modelo {model_name}")
                print("=" * 70)

                try:
                    # Carrega descritores comparativos
                    comparative_descriptions = load_comparative_descriptions(dataset_name)
                    
                    if not comparative_descriptions:
                        print("⏭️  Sem descritores comparativos → ignorado.")
                        continue

                    # Carrega embeddings de imagem (pré-calculados) e gera text embeddings
                    image_embeds, text_embeds, labels, class_names = \
                        load_embeddings_and_generate_comparative_text(
                            dataset_name, comparative_descriptions, model, clip, model_name, DEVICE
                        )

                    if image_embeds is None:
                        continue

                    # Avaliação zero-shot
                    # Converte para float() para garantir a compatibilidade de tipo antes da multiplicação
                    acc, top5_acc, preds = evaluate_zero_shot(
                        image_embeds.float(), text_embeds.float(), labels
                    )

                    # Calcula a média de descritores por classe
                    current_avg_desc = sum(
                        len(comparative_descriptions.get(cls, [])) 
                        for cls in class_names
                    ) / len(class_names) if class_names else 0

                    avg_descriptors += current_avg_desc
                    count_datasets += 1


                    print(f"\n🎯 Resultados Comparative-CLIP para {dataset_name}:")
                    print(f"   Top-1 Accuracy: {acc:.4f}")
                    print(f"   Top-5 Accuracy: {top5_acc:.4f}")

                    # Armazena os resultados no dicionário de resumo
                    summary[dataset_name] = {
                        "accuracy_top1": float(acc),
                        "accuracy_top5": float(top5_acc),
                        "num_classes": len(class_names),
                        "num_images": len(labels),
                        "method": "comparative_clip",
                        "model_name": model_name,
                        "template": "a photo of a {class}, {comparative_descriptor}",
                        "avg_descriptors_per_class": float(current_avg_desc)
                    }

                except Exception as e:
                    print(f"❌ Erro no dataset {dataset_name}: {e}")
                    # Imprime o traceback completo para debug
                    traceback.print_exc(file=sys.stdout) 

            # Salvar resultados
            # Cria um nome de arquivo seguro a partir do nome do modelo
            model_safe_name = model_name.replace('/', '-')
            out_path = RESULTS_DIR / f"comparative_clip_results_{model_safe_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4, ensure_ascii=False)

            print("\n" + "=" * 70)
            print(f"📈 Resultados salvos em: {out_path}")
            print("=" * 70)

            # Mostra resumo da iteração
            if summary:
                print(f"\n📊 RESUMO DOS RESULTADOS ({model_name}):")
                
                print(f"{'Dataset':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Classes':<10}")
                print("-" * 70)
                for ds, results in summary.items():
                    print(f"{ds:<20} {results['accuracy_top1']:<12.4f} "
                          f"{results['accuracy_top5']:<12.4f} {results['num_classes']:<10}")
                
                # Calcula a média geral se houver datasets avaliados
                avg_top1 = np.mean([r['accuracy_top1'] for r in summary.values()])
                avg_top5 = np.mean([r['accuracy_top5'] for r in summary.values()])
                print("-" * 70)
                print(f"{'MÉDIA':<20} {avg_top1:<12.4f} {avg_top5:<12.4f}")
                print("-" * 70)

        except Exception as e:
            # Captura erro no carregamento do modelo (se houver)
            print(f"❌ Erro fatal durante a avaliação do modelo {model_name}: {e}")
            traceback.print_exc(file=sys.stdout) 

    print("\n\n*** AVALIAÇÃO DE TODOS OS MODELOS CONCLUÍDA ***")


if __name__ == "__main__":
    main()