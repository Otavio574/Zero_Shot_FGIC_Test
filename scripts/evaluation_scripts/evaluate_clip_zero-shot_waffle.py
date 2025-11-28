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
import sys

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

# Assumindo que o script está em 'root/scripts/evaluation/zero_shot/'
# BASE_DIR será 'root/'
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback para execução em ambientes onde __file__ não está definido
    BASE_DIR = Path.cwd().parent.parent.parent 
    
SUMMARY_PATH = BASE_DIR / "outputs/analysis/summary.json"
EMBED_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "all_zero-shot_results/results_waffle_clip"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Lista de modelos CLIP a serem avaliados (RN = ResNet, ViT = Vision Transformer)
# CORREÇÃO 1: Nomes dos modelos agora usam '/' conforme exigido pelo clip.load()
ALL_MODELS = [
    'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parâmetros WaffleCLIP (do paper "Waffling around for Performance")
WAFFLE_COUNT = 15  # Número de PARES (palavra + sequência de caracteres) -> Total 30 descritores
REPS = 7  # Número de repetições para cálculo de média e desvio padrão


# ============================================================
# CARREGAMENTO E INICIALIZAÇÃO
# ============================================================

def load_datasets_from_summary(path: Path) -> dict:
    """
    Carrega a lista de datasets e seus caminhos a partir do arquivo summary.json.
    """
    if not path.exists():
        print(f"❌ summary.json não encontrado no caminho: {path}. Verifique a variável BASE_DIR.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Erro ao decodificar JSON em {path}.")
        return {}

    datasets = {}
    for item in data:
        if isinstance(item, dict) and "dataset" in item and "path" in item:
            datasets[item["dataset"]] = item["path"]

    return datasets

DATASETS = load_datasets_from_summary(SUMMARY_PATH)


# ============================================================
# GERADORES DE DESCRITORES ALEATÓRIOS (WAFFLE)
# ============================================================

def generate_random_word_descriptors(count: int, seed: int = None) -> list:
    """ Gera descritores de palavras aleatórias de um vocabulário fixo. """
    if seed is not None:
        random.seed(seed)

    # Vocabulário fixo de palavras aleatórias
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
    """ Gera descritores de sequências de caracteres aleatórias. """
    # Usa seed ligeiramente diferente para as chars para manter separação
    if seed is not None:
        random.seed(seed + 1)
    
    descriptors = []
    for _ in range(count):
        # 50% de chance de ser uma repetição de um caractere (e.g., 'aaaaaa')
        if random.random() < 0.5:
            char = random.choice(string.ascii_lowercase)
            length = random.randint(4, 8)
            desc = char * length
        else:
            # Sequência aleatória (e.g., 'qwerty')
            length = random.randint(4, 8)
            desc = ''.join(random.choices(string.ascii_lowercase, k=length))
        
        # 30% de chance de ter uma "waffle sequence" concatenada (e.g., 'qwerty fff')
        if random.random() < 0.3:
            desc += " " + random.choice(string.ascii_lowercase) * random.randint(2, 4)
        
        descriptors.append(desc)
    
    return descriptors


def generate_waffle_descriptors(count: int, seed: int = None) -> list:
    """ Combina palavras e sequências de caracteres para formar os descritores. """
    words = generate_random_word_descriptors(count, seed)
    chars = generate_random_char_descriptors(count, seed)
    
    all_descriptors = []
    # Intercala palavras e chars: [w1, c1, w2, c2, ...]
    for w, c in zip(words, chars):
        all_descriptors.append(w)
        all_descriptors.append(c)
    
    return all_descriptors


# ============================================================
# EMBEDDINGS COM DESCRITORES WAFFLE
# ============================================================

def get_text_embedding_waffle(class_name: str, waffle_descriptors: list, 
                              model, clip_library, device):
    """
    Gera embedding de texto médio (ensemble) usando o conjunto de descritores waffle.
    """
    class_readable = class_name.replace('_', ' ')
    
    # Template: "a photo of a {class}, {descriptor}"
    texts = [f"a photo of a {class_readable}, {desc}" for desc in waffle_descriptors]
    
    # Tokenização e envio para o dispositivo
    tokens = clip_library.tokenize(texts).to(device)
    
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Média dos embeddings (o coração do WaffleCLIP)
    final = text_embeds.mean(dim=0)
    final = final / final.norm() # Normalização final
    
    return final.cpu()


# ============================================================
# CARREGA EMBEDDINGS + TEXT EMBEDDINGS WAFFLE
# ============================================================

def load_embeddings_and_generate_waffle_text(dataset_name, model_name, model, clip_library, rep_seed: int):
    """
    Carrega embeddings de imagem e gera text embeddings waffle para todas as classes
    para uma dada repetição (seed).

    CORREÇÃO: Garantir que o nome do modelo no arquivo de embeddings esteja no 
    formato sanitizado (com hífens).
    """
    
    # CORREÇÃO 2: Substitui '/' por '-' para corresponder ao nome do arquivo de embeddings
    model_safe_name = model_name.replace('/', '-')
    emb_path = EMBED_DIR / f"{dataset_name}_{model_safe_name}.pt" 

    if not emb_path.exists():
        print(f"⚠️ Embeddings de imagem não encontrados para {model_name} (esperado: {emb_path})")
        return None, None, None, None

    # print(f"📂 Carregando embeddings de imagem: {emb_path}")
    data = torch.load(emb_path, map_location="cpu")

    image_embeds = data.get("image_embeddings")
    image_paths = data.get("image_paths")

    if image_embeds is None or image_paths is None:
        print("❌ .pt inválido, faltando chaves 'image_embeddings' ou 'image_paths'.")
        return None, None, None, None

    # Normalização dos embeddings de imagem (se já não estiverem normalizados)
    image_embeds = image_embeds.float()
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    # Extrai classes e labels (depende da estrutura de caminhos)
    class_names = sorted(list(set(Path(p).parts[-2] for p in image_paths)))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_to_idx[Path(p).parts[-2]] for p in image_paths])

    # print(f"   Total imagens: {len(labels)} | Classes: {len(class_names)}")

    text_embeds_list = []
    
    # Define a semente global para garantir que a sequência de seeds 
    # geradas abaixo seja a mesma para a repetição atual.
    random.seed(rep_seed) 
    
    for cls in class_names:
        # Gera uma nova seed aleatória para CADA CLASSE (para que cada classe 
        # use um conjunto ÚNICO de descritores aleatórios nesta repetição).
        cls_seed = random.randint(0, 1000000) 
        
        # Gera o conjunto de descritores waffle (2 * WAFFLE_COUNT)
        waffle_descs = generate_waffle_descriptors(WAFFLE_COUNT, seed=cls_seed)
        emb = get_text_embedding_waffle(cls, waffle_descs, model, clip_library, DEVICE)
        text_embeds_list.append(emb)

    text_embeds = torch.stack(text_embeds_list, dim=0)

    # print(f"✅ Text embeddings WaffleCLIP gerados: {text_embeds.shape}")

    return image_embeds, text_embeds, labels, class_names


# ============================================================
# ZERO-SHOT
# ============================================================

def evaluate_zero_shot(img_embeds: torch.Tensor, text_embeds: torch.Tensor, labels: np.ndarray):
    """
    Avaliação zero-shot via similaridade coseno (Top-1 e Top-5).
    """
    # Produto escalar entre imagem e texto embeddings (similaridade coseno)
    sims = img_embeds @ text_embeds.T
    
    # Top-1
    preds = sims.argmax(dim=-1).numpy()
    acc = accuracy_score(labels, preds)

    # Top-5
    # Obtém os índices dos 5 maiores scores para cada imagem
    top5_preds = sims.topk(5, dim=-1).indices.numpy()
    # Verifica se o label correto está entre os 5 preditos
    top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    
    return acc, top5_acc


# ============================================================
# MAIN LOOP (Itera sobre todos os modelos)
# ============================================================

def main():
    if not DATASETS:
        print("\n🚫 Não foi possível iniciar a avaliação. Nenhum dataset carregado de summary.json.")
        return

    print("🎲 WaffleCLIP Zero-Shot Evaluation")
    print(f"💻 Device: {DEVICE}")
    print(f"🎲 Waffle Count: {WAFFLE_COUNT} pares | Total Descriptors/Class: {2 * WAFFLE_COUNT}")
    print(f"🔁 Repetições (Média): {REPS}\n")

    # === LOOP PRINCIPAL SOBRE OS MODELOS ===
    for model_name in ALL_MODELS:
        print("\n" + "#" * 80)
        print(f"🚀 INICIANDO AVALIAÇÃO PARA MODELO: {model_name}")
        print("#" * 80)

        try:
            # 1. Carregar o modelo específico da iteração
            print(f"🔄 Carregando CLIP ({model_name})...")
            # Usa o nome padrão do modelo (com '/')
            model, _ = clip.load(model_name, device=DEVICE) 
            model.eval()
            print("✅ Modelo carregado!\n")

            summary = {}

            for dataset_name in DATASETS.keys():
                print("-" * 80)
                print(f"📊 Avaliando {dataset_name} (Modelo: {model_name})")
                print("-" * 80)

                try:
                    accuracies = []
                    top5_accuracies = []
                    
                    # 2. Loop de repetições (Waffle)
                    for rep in range(REPS):
                        print(f"🔁 Repetição {rep+1}/{REPS}")
                        
                        # Seed diferente para cada repetição (rep_seed)
                        rep_seed = rep + 42 
                        
                        # Passando 'model_name' (com '/') para carregar o modelo de imagem
                        image_embeds, text_embeds, labels, class_names = \
                            load_embeddings_and_generate_waffle_text(
                                dataset_name, model_name, model, clip, rep_seed
                            )

                        if image_embeds is None:
                            # Se não encontrou embeddings, para as repetições e pula este dataset
                            break

                        # Avaliação
                        acc, top5_acc = evaluate_zero_shot(
                            image_embeds.float(), text_embeds.float(), labels
                        )

                        print(f"   🎯 Acc Top-1: {acc:.4f} | Acc Top-5: {top5_acc:.4f}")
                        
                        accuracies.append(acc)
                        top5_accuracies.append(top5_acc)
                    
                    # 3. Calcular a média dos resultados
                    if len(accuracies) == REPS:
                        mean_acc = float(np.mean(accuracies))
                        std_acc = float(np.std(accuracies))
                        mean_top5 = float(np.mean(top5_accuracies))
                        std_top5 = float(np.std(top5_accuracies))
                        
                        print(f"\n✨ MÉDIA FINAL ({REPS} Reps): Top-1: {mean_acc:.4f} (±{std_acc:.4f}) | Top-5: {mean_top5:.4f} (±{std_top5:.4f})")

                        summary[dataset_name] = {
                            "accuracy_top1": mean_acc,
                            "std_top1": std_acc,
                            "accuracy_top5": mean_top5,
                            "std_top5": std_top5,
                            # Adicionado fallback para evitar NameError caso não haja embeddings
                            "num_classes": len(class_names) if 'class_names' in locals() and class_names is not None else 0, 
                            "num_images": len(labels) if 'labels' in locals() and labels is not None else 0,
                            "model": model_name,
                            "method": "waffle_clip_random_descriptors",
                            "repetitions": REPS,
                            "descriptors_per_class": 2 * WAFFLE_COUNT
                        }
                    elif len(accuracies) > 0:
                        print(f"⚠️ Não foi possível completar todas as repetições ({len(accuracies)}/{REPS}). Pulando a média.")
                    else:
                        print("🚫 Não foi possível carregar embeddings ou completar repetições para este dataset.")


                except Exception as e:
                    print(f"❌ Erro grave no dataset {dataset_name}: {e}")
                    # traceback.print_exc()

            # 4. Salvar resultados
            # Usa o nome sanitizado para o arquivo de resultados, como já estava sendo feito
            model_safe_name = model_name.replace('/', '-')
            out_path = RESULTS_DIR / f"waffle_clip_results_{model_safe_name}.json"
            
            if summary:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4, ensure_ascii=False)

                print("\n" + "=" * 80)
                print(f"📈 Resultados do modelo {model_name} salvos em: {out_path}")
                print("=" * 80)
                
                # 5. Mostra resumo
                print(f"\n📊 RESUMO GERAL ({model_name} - WaffleCLIP):")
                # Define o formato para alinhamento (25 para dataset, 20 para Acc)
                header = f"{'Dataset':<25} {'Top-1 Acc (±std)':<20} {'Top-5 Acc (±std)':<20} {'Classes':<10}"
                print("-" * len(header))
                print(header)
                print("-" * len(header))
                
                for ds, results in summary.items():
                    top1_str = f"{results['accuracy_top1']:.4f} (±{results['std_top1']:.4f})"
                    top5_str = f"{results['accuracy_top5']:.4f} (±{results['std_top5']:.4f})"
                    print(f"{ds:<25} {top1_str:<20} {top5_str:<20} {results['num_classes']:<10}")
                print("-" * len(header))

            else:
                print(f"\n⚠️ Nenhum resultado gerado com sucesso para o modelo {model_name}.")


        except Exception as e:
            # Captura erro no carregamento do modelo (se houver)
            print(f"❌ Erro fatal durante a avaliação do modelo {model_name}: {e}")
            # traceback.print_exc()

    print("\n" + "*" * 80)
    print("*** AVALIAÇÃO DE TODOS OS MODELOS CONCLUÍDA ***")
    print("*" * 80)

if __name__ == "__main__":
    main()