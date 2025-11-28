"""
Avaliação Zero-Shot com CLIP Baseline (Vanilla CLIP).
Usa apenas o template padrão: "a photo of a {class}"

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
from sklearn.metrics import accuracy_score
import traceback
import sys

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_clip_baseline"

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


DATASETS = load_datasets_from_summary(SUMMARY_PATH)


# ============================================================
# EMBEDDING COM TEMPLATE VANILLA CLIP
# ============================================================

def get_text_embedding_baseline(class_name: str, model, clip_library, device):
    """
    Gera embedding de texto usando apenas o template vanilla CLIP.
    
    Template: "a photo of a {class}"
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

def load_embeddings_and_generate_baseline_text(dataset_name, model_name, model, clip_library, device):
    """
    Carrega embeddings de imagem (específicos do modelo) e gera text embeddings vanilla CLIP.
    
    Args:
        dataset_name (str): Nome do dataset.
        model_name (str): Nome do modelo CLIP (ex: 'RN50', 'ViT-L/14').
        ...
    """
    # 📢 CORREÇÃO: Usa o nome do modelo no caminho do arquivo
    model_safe_name = model_name.replace('/', '-')
    emb_path = EMBED_DIR / f"{dataset_name}_{model_safe_name}.pt" # <--- ARQUIVO CORRETO

    if not emb_path.exists():
        print(f"⚠️ Embeddings de imagem não encontrados para {model_name}: {emb_path}")
        return None, None, None, None

    print(f"📂 Carregando embeddings de imagem: {emb_path}")
    # Certifique-se de carregar para CPU ou o DEVICE correto
    data = torch.load(emb_path, map_location="cpu", weights_only=False)

    # Verifica se o embedding carregado foi de fato gerado pelo modelo esperado
    loaded_clip_model = data.get("clip_model")
    if loaded_clip_model != model_name:
        print(f"❌ Erro de integridade: Arquivo '{emb_path.name}' foi gerado por '{loaded_clip_model}' mas está sendo usado com '{model_name}'.")
        print("   O processamento continuará, mas verifique se seus arquivos .pt estão corretos.")
        # Nota: O erro de dimensão não deve ocorrer mais se o arquivo existir, mas esta é uma boa verificação de sanidade.


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

    print(f" Total imagens: {len(labels)} | Classes: {len(class_names)}")

    # Gera text embeddings vanilla CLIP (sem descritores)
    print("\n📝 Gerando text embeddings vanilla CLIP...")
    print(' Template: "a photo of a {class}"')

    text_embeds_list = []
    for cls in tqdm(class_names, desc="   Classes"):
        emb = get_text_embedding_baseline(cls, model, clip_library, device)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0) 

    print(f"✅ Text embeddings baseline: {text_embeds.shape}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT EVALUATION
# ============================================================

def evaluate_zero_shot(img_embeds, text_embeds, labels):
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
    
    # 🔍 DEBUG:
    # print(f"\n🔍 Similaridades:")
    # print(f"  Min: {sims.min():.4f}, Max: {sims.max():.4f}")
    # print(f"  Mean: {sims.mean():.4f}, Std: {sims.std():.4f}")
    
    # Top-1 accuracy
    acc = accuracy_score(labels, preds)
    
    # Top-5 accuracy
    top5_preds = sims.topk(5, dim=-1).indices.cpu().numpy() # Move para CPU
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc, preds


# ============================================================
# MAIN LOOP (Alterada para rodar todos os modelos)
# ============================================================

def main():
    print("🎯 CLIP Baseline Zero-Shot Evaluation (Vanilla CLIP)")
    print(f"💻 Device: {DEVICE}")
    print('📝 Template: "a photo of a {class}"\n')
    
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

                # Carrega embeddings (usando o nome do modelo) e gera text embeddings baseline
                image_embeds, text_embeds, labels, class_names = \
                    load_embeddings_and_generate_baseline_text(
                        dataset_name, model_name, model, clip, DEVICE # 📢 NOVO: Passando model_name
                    )

                if image_embeds is None:
                    continue
                
                # 📢 NOTA: O cheque de dimensão agora é feito implicitamente ao carregar o arquivo
                # específico do modelo. Se o arquivo existe e foi gerado pelo mesmo modelo, 
                # a dimensão D (por exemplo, 512 ou 768) deve ser compatível.

                # Avaliação zero-shot
                acc, top5_acc, preds = evaluate_zero_shot(
                    image_embeds.float(), text_embeds.float(), labels # Tensors serão movidos para DEVICE dentro da função
                )

                print(f"\n🎯 Resultados CLIP Baseline para {dataset_name}:")
                print(f"   Top-1 Accuracy: {acc:.4f}")
                print(f"   Top-5 Accuracy: {top5_acc:.4f}")

                summary[dataset_name] = {
                    "accuracy_top1": float(acc),
                    "accuracy_top5": float(top5_acc),
                    "num_classes": len(class_names),
                    "num_images": len(labels),
                    "method": "vanilla_clip",
                    "template": "a photo of a {class}",
                    "model_name": model_name 
                }

            # Salvar resultados
            model_safe_name = model_name.replace('/', '-')
            out_path = RESULTS_DIR / f"clip_baseline_results_{model_safe_name}.json"
            
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
                print(f"\n⚠️  Nenhum resultado gerado com sucesso para o modelo {model_name}.")

        except Exception as e:
            print(f"❌ Erro fatal durante a avaliação do modelo {model_name}: {e}")
            traceback.print_exc()

    print("\n\n*** AVALIAÇÃO DE TODOS OS MODELOS CONCLUÍDA ***")


if __name__ == "__main__":
    main()