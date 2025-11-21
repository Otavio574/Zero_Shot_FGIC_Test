"""
Filtering Process para Comparative-CLIP (Few-Shot)
Implementa o Algoritmo 1 do paper "Enhancing Visual Classification using Comparative Descriptors"

Processo:
1. Para cada classe, calcula a mean image feature usando N imagens de treino
2. Para cada descritor, calcula similaridade com a mean image feature
3. Define threshold = similaridade(mean_image_feature, vanilla_prompt)
4. Mantém apenas top-k descritores acima do threshold
5. Se nenhum passar, usa apenas vanilla prompt
"""

import json
import torch
import numpy as np
import clip
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random

# ============================================================
# CONFIG
# ============================================================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
DESCRIPTOR_DIR = Path("descriptors_comparative_fast")
OUTPUT_DIR = Path("descriptors_comparative_filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parâmetros de filtragem
N_SHOTS = 5  # Número de imagens por classe para calcular mean feature
TOP_K = 15  # Número máximo de descritores a manter após filtro


# ============================================================
# LOAD DATASETS
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


# ============================================================
# COLETA IMAGENS POR CLASSE (FEW-SHOT)
# ============================================================

def collect_few_shot_images(dataset_path: str, n_shots: int = 5):
    """
    Coleta N imagens por classe para calcular mean image feature.
    """
    root = Path(dataset_path)
    
    # Procura por diretório train ou test
    if (root / "train").exists():
        data_dir = root / "train"
    elif (root / "test").exists():
        data_dir = root / "test"
    else:
        data_dir = root
    
    class_images = defaultdict(list)
    
    # Coleta imagens por classe
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Lista todas as imagens
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG', '.PNG'}
        images = [
            img for img in class_dir.iterdir() 
            if img.suffix in image_extensions
        ]
        
        # Seleciona N_SHOTS aleatórias
        if len(images) >= n_shots:
            selected = random.sample(images, n_shots)
        else:
            selected = images  # Usa todas se tiver menos que N_SHOTS
        
        class_images[class_name] = selected
    
    return class_images


# ============================================================
# CALCULA MEAN IMAGE FEATURE
# ============================================================

def compute_mean_image_features(class_images, model, preprocess):
    """
    Calcula mean image feature para cada classe usando CLIP.
    """
    print("📷 Calculando mean image features...")
    
    mean_features = {}
    
    for class_name, image_paths in tqdm(class_images.items(), desc="Classes"):
        features = []
        
        for img_path in image_paths:
            try:
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    feature = model.encode_image(image_input)
                    feature = feature / feature.norm(dim=-1, keepdim=True)
                
                features.append(feature.cpu())
            
            except Exception as e:
                print(f"   ⚠️  Erro ao processar {img_path}: {e}")
                continue
        
        if features:
            # Média das features
            mean_feature = torch.stack(features).mean(dim=0)
            mean_feature = mean_feature / mean_feature.norm()
            mean_features[class_name] = mean_feature.squeeze(0)
    
    print(f"✅ Mean features calculadas para {len(mean_features)} classes")
    
    return mean_features


# ============================================================
# FILTERING PROCESS (Algoritmo 1)
# ============================================================

def filter_descriptors(descriptors, mean_features, model, clip_lib):
    """
    Implementa o processo de filtragem do paper Comparative-CLIP.
    
    Algorithm 1:
    1. Para cada classe:
       a. Calcula threshold = sim(mean_image_feature, vanilla_prompt)
       b. Para cada descritor, calcula sim(mean_image_feature, descriptor_prompt)
       c. Mantém top-k descritores com similaridade > threshold
       d. Se nenhum passar, usa apenas vanilla prompt (lista vazia)
    """
    
    print("\n🔍 Aplicando processo de filtragem...")
    
    filtered_descriptors = {}
    stats = {
        'total_before': 0,
        'total_after': 0,
        'classes_filtered_out': 0,
        'avg_removed': []
    }
    
    for class_name, desc_list in tqdm(descriptors.items(), desc="Filtrando"):
        
        if class_name not in mean_features:
            print(f"   ⚠️  Mean feature não encontrada para {class_name}, pulando...")
            filtered_descriptors[class_name] = []
            continue
        
        mean_feature = mean_features[class_name].to(DEVICE)
        class_readable = class_name.replace('_', ' ')
        
        # 1. Calcula THRESHOLD: similaridade com vanilla prompt
        vanilla_prompt = f"a photo of a {class_readable}"
        vanilla_tokens = clip_lib.tokenize([vanilla_prompt]).to(DEVICE)
        
        with torch.no_grad():
            vanilla_text_feature = model.encode_text(vanilla_tokens)
            vanilla_text_feature = vanilla_text_feature / vanilla_text_feature.norm(dim=-1, keepdim=True)
        
        threshold = (mean_feature @ vanilla_text_feature.T).item()
        
        # 2. Calcula similaridade de cada descritor
        descriptor_scores = []
        
        for desc in desc_list:
            desc_prompt = f"a photo of a {class_readable}, {desc}"
            desc_tokens = clip_lib.tokenize([desc_prompt]).to(DEVICE)
            
            with torch.no_grad():
                desc_text_feature = model.encode_text(desc_tokens)
                desc_text_feature = desc_text_feature / desc_text_feature.norm(dim=-1, keepdim=True)
            
            similarity = (mean_feature @ desc_text_feature.T).item()
            descriptor_scores.append((desc, similarity))
        
        # 3. Filtra: mantém apenas descritores acima do threshold
        valid_descriptors = [
            (desc, score) for desc, score in descriptor_scores 
            if score > threshold
        ]
        
        # 4. Ordena por similaridade e pega top-k
        valid_descriptors.sort(key=lambda x: x[1], reverse=True)
        top_descriptors = [desc for desc, _ in valid_descriptors[:TOP_K]]
        
        # 5. Se nenhum passar, lista vazia (usa vanilla prompt no evaluate)
        filtered_descriptors[class_name] = top_descriptors
        
        # Stats
        stats['total_before'] += len(desc_list)
        stats['total_after'] += len(top_descriptors)
        stats['avg_removed'].append(len(desc_list) - len(top_descriptors))
        
        if len(top_descriptors) == 0:
            stats['classes_filtered_out'] += 1
    
    # Estatísticas finais
    print(f"\n📊 Estatísticas de Filtragem:")
    print(f"   Descritores antes: {stats['total_before']}")
    print(f"   Descritores depois: {stats['total_after']}")
    print(f"   Redução: {stats['total_before'] - stats['total_after']} ({((stats['total_before'] - stats['total_after']) / stats['total_before'] * 100):.1f}%)")
    print(f"   Média removidos/classe: {np.mean(stats['avg_removed']):.1f}")
    print(f"   Classes sem descritores válidos: {stats['classes_filtered_out']}")
    
    return filtered_descriptors


# ============================================================
# MAIN
# ============================================================

def main():
    print("🎯 Comparative-CLIP Descriptor Filtering (Few-Shot)")
    print(f"📦 Modelo: {MODEL_NAME}")
    print(f"💻 Device: {DEVICE}")
    print(f"🎲 N-shots: {N_SHOTS}")
    print(f"📊 Top-K: {TOP_K}\n")
    
    # Carrega CLIP
    print("🔄 Carregando CLIP...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ CLIP carregado!\n")
    
    # Carrega datasets
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    
    if not datasets:
        print("❌ Nenhum dataset no summary.json")
        return
    
    for dataset_name, dataset_path in datasets.items():
        
        print("=" * 70)
        print(f"📊 Processando {dataset_name}")
        print("=" * 70)
        
        try:
            # 1. Carrega descritores comparativos (não filtrados)
            desc_path = DESCRIPTOR_DIR / f"{dataset_name}_comparative_fast.json"
            
            if not desc_path.exists():
                print(f"⚠️  Descritores não encontrados: {desc_path}")
                continue
            
            with open(desc_path, 'r', encoding='utf-8') as f:
                descriptors = json.load(f)
            
            print(f"📂 Descritores carregados: {len(descriptors)} classes")
            
            # 2. Coleta imagens few-shot
            print(f"\n📷 Coletando {N_SHOTS} imagens por classe...")
            class_images = collect_few_shot_images(dataset_path, N_SHOTS)
            print(f"✅ Coletadas imagens de {len(class_images)} classes")
            
            # 3. Calcula mean image features
            mean_features = compute_mean_image_features(class_images, model, preprocess)
            
            # 4. Aplica filtro
            filtered_descriptors = filter_descriptors(
                descriptors, mean_features, model, clip
            )
            
            # 5. Salva descritores filtrados
            output_path = OUTPUT_DIR / f"{dataset_name}_comparative_filtered.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_descriptors, f, indent=4, ensure_ascii=False)
            
            print(f"\n✅ Descritores filtrados salvos em: {output_path}")
            
        except Exception as e:
            print(f"❌ Erro ao processar {dataset_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()