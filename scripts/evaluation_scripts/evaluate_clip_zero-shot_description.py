"""
Avaliação Zero-Shot com CLIP usando descriptions detalhadas.
IMPLEMENTAÇÃO MAX-POOLING: Classifica usando a similaridade máxima
sobre todos os templates fornecidos por classe.
"""

import os
import json
import torch
import numpy as np
# Usa o módulo 'clip' da openai, que é o padrão para zero-shot
import clip 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# ============================
# CONFIGURAÇÕES
# ============================

def load_datasets_from_summary(summary_path: Path) -> dict:
    """Carrega configuração de datasets do summary.json"""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar summary: {e}")
        return {}
    
    datasets = {}
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    elif isinstance(summary, dict) and 'datasets' in summary:
        for name, path in summary['datasets'].items():
            datasets[name] = path
            
    return datasets


path_string = "all_zero-shot_results/"

SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS = load_datasets_from_summary(SUMMARY_PATH)

MODEL_NAME = "ViT-B/32" #Modelo CLIP do OpenAI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = path_string +"results_zero_shot_max_pooling" # Novo diretório para resultados Max-Pooling

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_descriptions(dataset_name):
    """
    Carrega descriptions do dataset.
    
    Nota: Para Max-Pooling, esperamos que o arquivo contenha uma LISTA
    de templates para cada classe.
    """
    # 🚨 ATENÇÃO: Verifique se o caminho 'descriptors_dclip_optimized' está correto
    # Se você usou o nome 'descriptors' anteriormente, ajuste aqui!
    path = os.path.join("descriptors_local_llm", f"{dataset_name}_local_descriptors.json") 
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Descritores não encontrados em {path}. Verifique se o generate_descriptors.py foi executado.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Erro ao ler JSON de descritores em {path}")
        return {}


def load_embeddings_and_generate_text(dataset_name, descriptions, model):
    """
    Carrega embeddings de imagem e gera embeddings de texto de TODOS os templates por classe.
    
    Retorna:
    - image_embeds (Tensor): Embeddings de imagem (N_images, D)
    - text_embeds (Tensor): Embeddings de texto (N_templates_total, D)
    - labels (Numpy Array): Labels verdadeiros (N_images,)
    - template_counts (List[int]): Lista onde template_counts[i] é o número de templates para a classe i.
    """
    
    embedding_path = os.path.join("embeddings", f"{dataset_name}.pt")
    
    if not os.path.exists(embedding_path):
        print(f"⚠️  Embeddings não encontrados: {embedding_path}")
        return None, None, None, None
    
    print(f"📂 Carregando embeddings: {embedding_path}")
    embeddings_data = torch.load(embedding_path, weights_only=False, map_location='cpu')
    
    image_embeds = embeddings_data.get('image_embeddings')
    image_paths = embeddings_data.get('image_paths')
    
    if image_embeds is None or image_paths is None:
        print("❌ Chaves 'image_embeddings' ou 'image_paths' não encontradas no arquivo .pt.")
        return None, None, None, None

    # 1. Extrai classes e labels
    labels = []
    class_to_idx = {}
    class_names = []

    for path in image_paths:
        parts = Path(path).parts
        class_name = parts[-2] if len(parts) >= 2 else "unknown"

        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_names)
            class_names.append(class_name)

        labels.append(class_to_idx[class_name])
        
    labels = np.array(labels)
    
    # Ordenar class_names alfabeticamente (CRUCIAL para alinhar com o JSON)
    class_names.sort() 
    
    # Recria labels baseados na ordem alfabética
    class_to_idx_sorted = {name: idx for idx, name in enumerate(class_names)}
    new_labels = np.array([class_to_idx_sorted.get(class_names[original_idx], -1) 
                           for original_idx in labels])
    labels = new_labels

    print(f"   Total de imagens: {len(labels)} | Classes: {len(class_names)}")

    # 2. Gera templates de texto para Max-Pooling
    all_class_texts = []
    template_counts = [] # Conta quantos templates foram usados para cada classe
    
    # 🚨 LOOP CRUCIAL PARA MAX-POOLING 🚨
    for class_name in class_names:
        templates = descriptions.get(class_name) 
        
        if isinstance(templates, list) and templates:
            # ✅ Implementação DCLIP Max-Pooling: Adiciona todos os templates
            class_texts_to_add = templates
        elif isinstance(templates, str):
            # ⚠️ Fallback DCLIP Single-Descriptor: Se for apenas uma string (formato antigo)
            class_texts_to_add = [templates]
            print(f"⚠️ Alerta: '{class_name}' usando Single-Descriptor (string), não lista.")
        else:
            # ❌ Fallback para Template Simples (em caso de erro no JSON)
            fallback_text = f"a photo of a {sanitize_class_name(class_name)}"
            class_texts_to_add = [fallback_text]
            
        all_class_texts.extend(class_texts_to_add)
        template_counts.append(len(class_texts_to_add))
            
    print(f"\n📝 Gerando text embeddings para {len(all_class_texts)} templates totais...")
    print(f"   Classes: {len(template_counts)}. Templates por classe (min/max): {min(template_counts)}/{max(template_counts)}")

    # Exibe um exemplo
    print(f"   Exemplo de template processado: {all_class_texts[0][:100]}...")

    # Tokeniza e gera embeddings com CLIP
    text_tokens = clip.tokenize(all_class_texts, truncate=True).to(DEVICE)
    
    with torch.no_grad():
        text_embeds = model.encode_text(text_tokens)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    print(f"✅ Text embeddings prontos! Shape: {text_embeds.shape}")
    
    return image_embeds.cpu(), text_embeds.cpu(), labels, template_counts


def evaluate_zero_shot_max_pooling(image_embeds, text_embeds, labels, template_counts):
    """
    Calcula acurácia zero-shot usando a estratégia de Max-Pooling.
    """
    image_embeds = image_embeds.float()
    text_embeds = text_embeds.float()
    
    # ====== DEBUG ======
    print(f"\n🔍 DEBUG:")
    print(f"   Image embeds shape: {image_embeds.shape}")
    print(f"   Text embeds shape: {text_embeds.shape}")
    print(f"   Total de classes (baseado em template_counts): {len(template_counts)}")
    print(f"   Labels unique: {np.unique(labels)}")
    # ==================
    
    # 1. Calcula a similaridade total (N_images, N_templates_total)
    # Resultado é a matriz de similaridade entre TODAS as imagens e TODOS os templates.
    similarity_matrix_flat = image_embeds @ text_embeds.T 

    # 2. Aplica Max-Pooling para agrupar as similaridades por classe
    
    # Cria uma lista de índices para agrupar as colunas de templates por classe.
    start_index = 0
    max_similarity_by_class = []
    
    for count in template_counts:
        end_index = start_index + count
        
        # 3. Extrai as colunas de templates para a classe atual
        class_sims = similarity_matrix_flat[:, start_index:end_index]
        
        # 4. Max-Pooling: Encontra a similaridade MÁXIMA para aquela classe
        # (max_sims tem shape N_images)
        max_sims, _ = torch.max(class_sims, dim=1)
        
        max_similarity_by_class.append(max_sims.unsqueeze(1))
        
        start_index = end_index
        
    # Concatena as similaridades máximas em uma matriz final (N_images, N_classes)
    final_similarity_matrix = torch.cat(max_similarity_by_class, dim=1)
    
    # 5. Previsão final
    preds = final_similarity_matrix.argmax(dim=-1).numpy()
    
    acc = accuracy_score(labels, preds)
    return acc, preds


def plot_confusion_matrix(labels, preds, class_names, output_path):
    """Gera e salva matriz de confusão"""
    try:
        cm = confusion_matrix(labels, preds, normalize='true')
    except ValueError as e:
        print(f"❌ Erro ao calcular matriz de confusão: {e}")
        return

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='viridis', aspect='auto')
    plt.title("CLIP Zero-Shot with Max-Pooling Descriptions", fontsize=14)
    plt.colorbar()
    
    # Ajuste de fonte e ticks para evitar sobreposição em muitos datasets
    fontsize = max(3, 12 - len(class_names) // 10)
    
    plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=fontsize)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def sanitize_class_name(class_name: str) -> str:
    """Função dummy para ser usada no fallback, se necessário."""
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()


def main():
    print(f"🚀 Avaliação Zero-Shot com CLIP + Descriptions (MAX-POOLING)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}\n")
    
    # Carrega modelo CLIP
    print("🔄 Carregando modelo CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE) 
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "method": "CLIP with Max-Pooling Descriptions",
        "total_datasets": len(DATASETS),
        "successful": 0,
        "failed": 0,
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"📊 Avaliando dataset: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            descriptions = load_descriptions(dataset_name)
            if not descriptions:
                print(f"⏭️  Não há descrições para {dataset_name}. Pulando.")
                summary["failed"] += 1
                continue

            print(f"✅ Descriptions carregadas: {len(descriptions)} classes")
            
            # NOTA: O dataset_path não é mais usado aqui, mas mantido para compatibilidade
            result = load_embeddings_and_generate_text(
                dataset_name, descriptions, model
            )
            
            image_embeds, text_embeds, labels, template_counts = result
            
            if image_embeds is None:
                print(f"⏭️  Pulando {dataset_name} devido à falha no carregamento/parsing.")
                summary["failed"] += 1
                continue
                
            # 🚨 MUDANÇA PRINCIPAL: Chama a função de Max-Pooling
            acc, preds = evaluate_zero_shot_max_pooling(image_embeds, text_embeds, labels, template_counts)
            print(f"\n✅ Acurácia zero-shot (CLIP + Max-Pooling): {acc:.4f}")

            plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_cm_max_pooling.png")
            plot_confusion_matrix(labels, preds, class_names=[], output_path=plot_path) # class_names não é mais necessário aqui pois CM usa labels e preds

            summary["successful"] += 1
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes": len(template_counts),
                "num_images": len(labels),
                "confusion_matrix_plot": plot_path
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            traceback.print_exc()
            summary["failed"] += 1
            continue

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results_max_pooling.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"📈 Resultados salvos em {out_path}")
    print(f"✅ {summary['successful']} datasets processados com sucesso.")
    print(f"❌ {summary['failed']} falharam.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()