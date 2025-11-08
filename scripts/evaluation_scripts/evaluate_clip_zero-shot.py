"""
Avaliação Zero-Shot de CLIP usando embeddings pré-calculados e templates.
Este script carrega embeddings de imagens já extraídos, gera embeddings de texto
a partir dos templates CLIP, e avalia a acurácia zero-shot.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
from pathlib import Path
from glob import glob
from collections import Counter
import traceback

# ============================
# CONFIGURAÇÕES
# ============================

def load_datasets_from_summary(summary_path: Path) -> dict:
    """Carrega configuração de datasets do summary.json"""
    if not summary_path.exists():
        print(f"⚠️  Arquivo {summary_path} não encontrado!")
        return {}
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    datasets = {}
    
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    elif isinstance(summary, dict):
        # Tenta extrair a lista de datasets de um dicionário, se for o caso
        if 'datasets' in summary:
            datasets = summary['datasets']
        else:
            # Assume que o dicionário já é o mapeamento nome:caminho
            datasets = summary
    
    return datasets

SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS = load_datasets_from_summary(SUMMARY_PATH)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_baseline"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_templates(dataset_name):
    """Carrega template do dataset e sanitiza suas chaves para matching robusto."""
    path = os.path.join("descriptors", f"{dataset_name}_templates.json")
    if not os.path.exists(path):
        # Tenta o nome corrigido para o novo formato
        path = os.path.join("descriptors", f"{dataset_name}_descriptors.json") 
        if not os.path.exists(path):
            print(f"⚠️  Templates/Descriptors não encontrados: {path}")
            return {}
    
    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)
        
    # CRUCIAL: Saneia TODAS as chaves do dicionário lido
    sanitized_templates = {normalize_class_key(key): value for key, value in templates.items()}
    return sanitized_templates


def normalize_class_key(key: str) -> str:
    """Normaliza uma string de classe eliminando case, espaços e caracteres especiais comuns."""
    key = key.strip()
    key = key.lower()
    # Remove underscores, hífens e pontos (crucial para '001.Black_footed_Albatross')
    key = key.replace('_', '').replace('-', '').replace('.', '') 
    key = key.replace(' ', '')
    return key


def extract_class_from_path(path, dataset_path):
    """Extrai nome da classe de um path de imagem"""
    path_obj = Path(path)
    dataset_obj = Path(dataset_path)
    
    try:
        # Pega o path relativo ao dataset
        rel_path = path_obj.relative_to(dataset_obj)
        # A classe geralmente é a primeira pasta depois do dataset
        if len(rel_path.parts) >= 2:
            return rel_path.parts[0]
        else:
            # Fallback para o nome da pasta pai se a estrutura for plana
            return path_obj.parent.name 
    except ValueError:
        # Se não conseguir fazer relative_to, usa o nome da pasta pai
        return path_obj.parent.name


def match_descriptor_to_class(class_name: str, template: dict) -> str:
    """
    Realiza o match do nome da classe (do path) com os template
    (com chaves normalizadas) usando a normalização universal.
    """
    
    # CRUCIAL: Normaliza o nome da classe extraído do path
    class_normalized = normalize_class_key(class_name) 
    
    # O match agora é direto e garantido, pois ambos os lados foram normalizados
    if class_normalized in template:
        return template[class_normalized]

    # Fallback (Apenas se o descriptor realmente não existir no JSON)
    print(f"⚠️  Fallback para classe não mapeada: {class_name}")
    # Usa o nome legível (com espaços, sem o prefixo numérico) no template de fallback
    readable_name = class_name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    
    # ⚠️ Este é o prompt FRACA de fallback
    return f"a photo of a {readable_name}"


def load_embeddings_and_generate_text(dataset_name, dataset_path, template, model, processor):
    """Carrega embeddings de imagem e gera embeddings de texto dos template"""
    
    # Carrega image embeddings
    embedding_path = os.path.join("embeddings", f"{dataset_name}.pt")
    
    if not os.path.exists(embedding_path):
        print(f"⚠️  Embeddings não encontrados: {embedding_path}")
        return None, None, None, None
    
    print(f"📂 Carregando embeddings: {embedding_path}")
    embeddings_data = torch.load(embedding_path, map_location='cpu')
    
    # Detecta formato
    if isinstance(embeddings_data, dict):
        # Formato: {'image_embeddings': tensor, 'image_paths': list}
        image_embeds = embeddings_data.get('image_embeddings')
        image_paths = embeddings_data.get('image_paths', embeddings_data.get('paths'))
        print(f"  Formato: dicionário com paths")
    else:
        # Formato: apenas tensor
        image_embeds = embeddings_data
        image_paths = None
        print(f"  Formato: tensor direto")
    
    if image_embeds is None:
        print(f"❌ Não foi possível extrair embeddings do arquivo")
        return None, None, None, None
    
    # Normaliza embeddings se necessário
    if image_embeds.norm(dim=-1, keepdim=True).mean() > 1.1:
        print(f"  Normalizando image embeddings...")
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    # 🚨 CORREÇÃO CRÍTICA DO RESHAPE 🚨
    # Garante que o tensor está no formato [N, 512], onde N é o número de imagens.
    if image_embeds.dim() == 1:
        # Se for 1D, calculamos N = tamanho total / 512 (dimensão do ViT-B/32)
        N = image_embeds.size(0) // 512 
        # Só faz o reshape se o tamanho for divisível por 512
        if image_embeds.size(0) % 512 == 0 and N > 0:
            image_embeds = image_embeds.view(N, 512)
            print(f"  ⚠️ Embeddings 1D REDIMENSIONADOS para {image_embeds.shape}")
        else:
            # Caso contrário, o tensor está incorreto
            print(f"❌ Erro de formato: Tamanho do tensor ({image_embeds.size(0)}) não é divisível por 512.")
            return None, None, None, None
            
    # Verifica a dimensão final
    if image_embeds.dim() != 2 or image_embeds.shape[1] != 512:
        print(f"❌ Erro de formato: Shape final esperado [N, 512], encontrado {image_embeds.shape}")
        return None, None, None, None

    print(f"  Shape: {image_embeds.shape}")
    
    # Extrai classes e labels
    if image_paths:
        # Usa paths para extrair classes
        labels = []
        class_to_idx = {}
        class_names = []
        
        # CÓDIGO CORRIGIDO: Alinhamento das labels com os paths
        for path in image_paths:
            class_name_raw = extract_class_from_path(path, dataset_path)

            # O nome da classe para MATCH DEVE SER A VERSÃO NORMALIZADA (p/ casar com a chave JSON)
            class_name_for_match = normalize_class_key(class_name_raw)

            if class_name_for_match not in class_to_idx:
                class_to_idx[class_name_for_match] = len(class_names)
                # Mantém o nome da classe RAW/não normalizado para exibição
                class_names.append(class_name_raw) 

            # Labels usa o índice da classe normalizada
            labels.append(class_to_idx[class_name_for_match])

        labels = np.array(labels)
    else:
        # CÓDIGO DE FALLBACK PARA INFERÊNCIA DE CLASSES
        print("⚠️  Sem paths salvos, inferindo estrutura...")
        class_folders = {}
        dataset_path_obj = Path(dataset_path)
        
        for img_path in dataset_path_obj.rglob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                class_name = img_path.parent.name
                if class_name not in class_folders:
                    class_folders[class_name] = []
                class_folders[class_name].append(str(img_path))
        
        if not class_folders:
            print("❌ Não foi possível inferir classes")
            return None, None, None, None
        
        class_names = sorted(class_folders.keys())
        print(f"  Classes inferidas: {len(class_names)}")
        
        all_paths = []
        labels = []
        for class_idx, class_name in enumerate(class_names):
            paths = sorted(class_folders[class_name])
            all_paths.extend(paths)
            labels.extend([class_idx] * len(paths))
        
        labels = np.array(labels[:len(image_embeds)])
        print(f"  ⚠️  Labels inferidos (pode não estar perfeitamente alinhado)")
    
    print(f"  Total de imagens: {len(labels)}")
    print(f"  Classes únicas: {len(set(labels))}")
    print(f"  Distribuição de classes:")
    class_counts = Counter(labels)
    
    # Mapeia os índices de volta para os nomes de classe (usando a lista class_names)
    sorted_class_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    
    for cls_idx, count in sorted_class_counts[:5]:
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
        print(f"      {cls_name}: {count} imagens")
    
    # Gera text embeddings dos templates
    class_texts = []
    
    # --- VERIFICAÇÃO DE COBERTURA (adicionado para debug) ---
    descriptor_coverage = 0
    fallback_count = 0
    
    fallback_prefix = "a photo of a "
    
    # IMPORTANTE: Aqui, class_names é a lista de nomes de classe (RAW) na ordem do índice 0..N
    # Usamos o nome da classe (RAW) para fazer o match no template.
    for class_name_raw in class_names:
        description = match_descriptor_to_class(class_name_raw, template)
        
        # Simula o fallback fraco para contagem
        readable_name = class_name_raw.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        fallback_description = f"{fallback_prefix}{readable_name}" 
        
        if description == fallback_description:
            fallback_count += 1
        else:
            descriptor_coverage += 1
            
        class_texts.append(description)
        
    print(f"\n📝 Gerando text embeddings para {len(class_texts)} classes...")
    # ESTA É A MENSAGEM CRÍTICA
    print(f"📊 Cobertura de template: {descriptor_coverage} específicos encontrados, {fallback_count} usando fallback genérico.")
    
    # ... (restante do código)
    

    
    print(f"  Exemplos de textos:")
    for i in range(min(3, len(class_texts))):
        txt = class_texts[i]
        print(f"      {class_names[i]}: {txt[:70]}...")
    
    text_inputs = processor(
        text=class_texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    print(f"✅ Embeddings carregados e processados!")


    # === INÍCIO DO BLOCO DE DIAGNÓSTICO FINAL (INSERIR AQUI) ===
    print("\n--- DIAGNÓSTICO DO MATCH ---")
    
    # Exemplo de Chave JSON (Nome da Classe no Dicionário)
    if template:
        first_json_key_raw = list(template.keys())[0]
        print(f"1. Chave JSON Carregada (1ª): '{first_json_key_raw}'")
        print(f"   (Normalizada p/ busca): '{normalize_class_key(first_json_key_raw)}'")
    else:
        print("1. ERRO: Dicionário de template está vazio.")
        
    # Exemplo de Nome de Classe do Path
    if class_names:
        first_class_name_raw = class_names[0]
        print(f"2. Nome da Classe do Path (1ª): '{first_class_name_raw}'")
        print(f"   (Normalizada p/ busca): '{normalize_class_key(first_class_name_raw)}'")
    
        # Teste de Igualdade Agressivo
        if template:
            sane_json = normalize_class_key(first_json_key_raw)
            sane_path = normalize_class_key(first_class_name_raw)
            
            is_equal = (sane_json == sane_path)
            print(f"3. Teste de Igualdade (Saneado): {is_equal}")
            if not is_equal:
                print(f"   DIFERENÇA: len(JSON)={len(sane_json)} vs len(PATH)={len(sane_path)}")
                
    print("--- FIM DO DIAGNÓSTICO DO MATCH ---\n")
    # === FIM DO BLOCO DE DIAGNÓSTICO FINAL ===
    
    return image_embeds, text_embeds.cpu(), labels, class_names


def evaluate_zero_shot(image_embeds, text_embeds, labels):
    """Calcula acurácia zero-shot."""
    print(f"\n🔍 Calculando similaridades...")
    print(f"  Image embeds: {image_embeds.shape}")
    print(f"  Text embeds: {text_embeds.shape}")
    
    # Garante que ambos estão normalizados (embora já tenham sido no load)
    # Re-normalização é segura, mas tecnicamente desnecessária se o load for perfeito
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Similaridade de Cosseno (produto escalar de vetores normalizados)
    # Esta é a linha que falhava devido ao shape incorreto do image_embeds
    sims = image_embeds @ text_embeds.T 
    preds = sims.argmax(dim=-1).numpy()
    acc = accuracy_score(labels, preds)
    
    print(f"  Similaridade média: {sims.mean():.4f}")
    print(f"  Similaridade máxima: {sims.max():.4f}")
    print(f"  Predições únicas: {len(np.unique(preds))}")
    
    return acc, preds


def plot_confusion_matrix(labels, preds, class_names, output_path):
    """Gera e salva matriz de confusão"""
    try:
        # Limita o número de classes para visualização para evitar gráficos gigantes
        max_classes = 50
        
        # Filtra para as top N classes se houver muitas
        if len(class_names) > max_classes:
            print(f"  ⚠️  Muitas classes ({len(class_names)}), mostrando top {max_classes} mais frequentes")
            unique, counts = np.unique(labels, return_counts=True)
            # Índices das classes mais frequentes
            top_class_indices = unique[np.argsort(counts)[-max_classes:]] 
            
            mask = np.isin(labels, top_class_indices)
            labels_filtered = labels[mask]
            preds_filtered = preds[mask]
            class_names_filtered = [class_names[i] for i in top_class_indices]
            
            # Mapeia os índices filtrados para 0..N para o plot
            old_to_new_index = {old: new for new, old in enumerate(top_class_indices)}
            labels_filtered = np.array([old_to_new_index[y] for y in labels_filtered])
            # Predições que não estão no top N são mapeadas para -1 e removidas.
            preds_filtered = np.array([old_to_new_index.get(y, -1) for y in preds_filtered]) 
            
            # Remove predições que não estão no set de classes filtradas
            valid_preds_mask = (preds_filtered >= 0)
            labels_filtered = labels_filtered[valid_preds_mask]
            preds_filtered = preds_filtered[valid_preds_mask]

            cm = confusion_matrix(labels_filtered, preds_filtered, normalize='true')
            class_names = class_names_filtered
        else:
            cm = confusion_matrix(labels, preds, normalize='true')
        
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, cmap='viridis', aspect='auto')
        plt.title("Zero-Shot Confusion Matrix", fontsize=14)
        plt.colorbar()
        
        # Ajusta o tamanho da fonte
        fontsize = max(6, 12 - len(class_names) // 10)
        
        plt.xticks(np.arange(len(class_names)), class_names, rotation=90, fontsize=fontsize)
        plt.yticks(np.arange(len(class_names)), class_names, fontsize=fontsize)
        plt.xlabel('Predicted', fontsize=10)
        plt.ylabel('True', fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Matriz de confusão salva: {output_path}")
    except Exception as e:
        print(f"  ⚠️  Erro ao gerar matriz de confusão: {e}")
        traceback.print_exc()


# ============================
# AVALIAÇÃO PRINCIPAL
# ============================

def main():
    print(f"🚀 Iniciando avaliação Zero-Shot CLIP")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"📊 Datasets encontrados: {len(DATASETS)}\n")
    
    if not DATASETS:
        print("❌ Nenhum dataset encontrado! Verifique o arquivo summary.json")
        return
    
    print("🔧 Carregando modelo CLIP...")
    # Garante que o modelo e o processor são carregados apenas uma vez
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Modelo carregado!\n")

    summary = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "total_datasets": len(DATASETS),
        "successful": 0,
        "failed": 0,
        "results": {}
    }

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"📊 Avaliando dataset: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            template = load_templates(dataset_name)
            if not template:
                print(f"⚠️  Sem template, usando templates genéricos")
            
            result = load_embeddings_and_generate_text(
                dataset_name, dataset_path, template, model, processor
            )
            
            if result[0] is None:
                print(f"⏭️  Pulando {dataset_name}")
                summary["failed"] += 1
                continue
                
            image_embeds, text_embeds, labels, class_names = result

            acc, preds = evaluate_zero_shot(image_embeds.cpu(), text_embeds.cpu(), labels)
            print(f"\n✅ Acurácia zero-shot: {acc:.4f}")

            plot_path = os.path.join(RESULTS_DIR, f"{dataset_name}_cm.png")
            plot_confusion_matrix(labels, preds, class_names, plot_path)

            summary["successful"] += 1
            summary["results"][dataset_name] = {
                "accuracy": float(acc),
                "num_classes": len(class_names),
                "num_images": len(labels),
                "confusion_matrix_plot": plot_path
            }
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            traceback.print_exc()
            summary["failed"] += 1
            continue

    # Salva resultados
    out_path = os.path.join(RESULTS_DIR, "zero_shot_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    summary_path = os.path.join(RESULTS_DIR, "accuracy_summary.json")
    accuracy_only = {name: f"{data['accuracy']:.4f}" 
                      for name, data in summary["results"].items()}
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_only, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMO FINAL")
    print(f"{'='*60}")
    print(f"✅ Datasets processados com sucesso: {summary['successful']}")
    print(f"❌ Datasets com falha: {summary['failed']}")
    
    if summary["results"]:
        print(f"\n📈 Acurácias (ordenadas):")
        for name, data in sorted(summary["results"].items(), 
                                 key=lambda x: x[1]["accuracy"], 
                                 reverse=True):
            print(f"  {name:30s}: {data['accuracy']:.4f} "
                  f"({data['num_classes']} classes, {data['num_images']} imgs)")
    
    print(f"\n📁 Resultados salvos em:")
    print(f"  - {out_path}")
    print(f"  - {summary_path}")
    print(f"  - Matrizes de confusão: {RESULTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()