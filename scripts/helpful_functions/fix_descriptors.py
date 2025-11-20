"""
Gera descriptors corretos usando os nomes reais das classes
ao invés de nomes aleatórios/incorretos
"""

import os
import json
import torch
from pathlib import Path
from collections import Counter

EMBEDDINGS_DIR = "embeddings"
DESCRIPTORS_DIR = "descriptors"
OUTPUT_DIR = "descriptors_fixed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_class_name(class_name):
    """Limpa o nome da classe para texto natural"""
    # Remove números no início (ex: "001.Black_footed_Albatross" → "Black_footed_Albatross")
    name = class_name
    if '.' in name:
        name = name.split('.', 1)[1]
    
    # Substitui underscores e hífens por espaços
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Remove espaços extras
    name = ' '.join(name.split())
    
    return name.lower()


def generate_descriptors_from_embeddings(dataset_name):
    """Gera descriptors corretos a partir dos paths salvos nos embeddings"""
    
    print(f"\n{'='*60}")
    print(f"📝 Gerando descriptors para: {dataset_name}")
    print(f"{'='*60}\n")
    
    # Carrega embeddings
    emb_path = os.path.join(EMBEDDINGS_DIR, f"{dataset_name}.pt")
    
    if not os.path.exists(emb_path):
        print(f"❌ Embeddings não encontrados: {emb_path}")
        return
    
    data = torch.load(emb_path, map_location='cpu')
    
    if not isinstance(data, dict) or 'image_paths' not in data:
        print(f"❌ Formato inválido (sem paths salvos)")
        return
    
    paths = data['image_paths']
    print(f"📊 Total de imagens: {len(paths)}")
    
    # Extrai classes únicas
    class_names_raw = []
    for path in paths:
        parts = Path(path).parts
        class_name = parts[-2]  # Pasta pai da imagem
        class_names_raw.append(class_name)
    
    unique_classes = sorted(set(class_names_raw))
    print(f"📊 Classes únicas: {len(unique_classes)}")
    
    # Determina o tipo de dataset baseado no nome
    dataset_lower = dataset_name.lower()
    
    if 'cub' in dataset_lower or 'bird' in dataset_lower:
        template = "a photo of a {}, a type of bird"
        category = "pássaro"
    elif 'aircraft' in dataset_lower or 'plane' in dataset_lower:
        template = "a photo of a {}, a type of aircraft"
        category = "aeronave"
    elif 'car' in dataset_lower or 'vehicle' in dataset_lower:
        template = "a photo of a {}, a type of car"
        category = "carro"
    elif 'dog' in dataset_lower or 'pet' in dataset_lower:
        template = "a photo of a {}, a type of dog"
        category = "cachorro"
    elif 'flower' in dataset_lower or 'plant' in dataset_lower:
        template = "a photo of a {}, a type of flower"
        category = "flor"
    else:
        template = "a photo of a {}"
        category = "objeto"
    
    print(f"📝 Categoria detectada: {category}")
    print(f"📝 Template: {template}")
    
    # Gera descriptors
    descriptors = {}
    
    for class_name in unique_classes:
        # Limpa o nome
        clean_name = clean_class_name(class_name)
        
        # Gera descrição
        description = template.format(clean_name)
        
        # Usa o nome original como chave
        descriptors[class_name] = description
    
    # Mostra exemplos
    print(f"\n📋 Exemplos de descriptors gerados:")
    for i, (cls, desc) in enumerate(list(descriptors.items())[:10]):
        print(f"   {cls:40s} → {desc}")
    
    # Salva
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_descriptors.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptors, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Descriptors salvos em: {output_path}")
    print(f"   Total: {len(descriptors)} classes")
    
    # Também salva cópia no diretório original (backup do antigo)
    old_desc_path = os.path.join(DESCRIPTORS_DIR, f"{dataset_name}_descriptors.json")
    if os.path.exists(old_desc_path):
        backup_path = os.path.join(DESCRIPTORS_DIR, f"{dataset_name}_descriptors_OLD.json")
        os.rename(old_desc_path, backup_path)
        print(f"   Backup do antigo: {backup_path}")
    
    # Copia o novo para o diretório original
    import shutil
    shutil.copy(output_path, old_desc_path)
    print(f"   ✅ Copiado para: {old_desc_path}")


def compare_descriptors(dataset_name):
    """Compara descriptors antigos vs novos"""
    
    print(f"\n{'='*60}")
    print(f"🔍 Comparando descriptors: {dataset_name}")
    print(f"{'='*60}\n")
    
    old_path = os.path.join(DESCRIPTORS_DIR, f"{dataset_name}_descriptors_OLD.json")
    new_path = os.path.join(DESCRIPTORS_DIR, f"{dataset_name}_descriptors.json")
    
    if not os.path.exists(old_path):
        print(f"⚠️  Backup antigo não encontrado")
        return
    
    with open(old_path, 'r', encoding='utf-8') as f:
        old_desc = json.load(f)
    
    with open(new_path, 'r', encoding='utf-8') as f:
        new_desc = json.load(f)
    
    print(f"📊 Comparação:")
    print(f"   Antigo: {len(old_desc)} descriptors")
    print(f"   Novo: {len(new_desc)} descriptors")
    
    print(f"\n📋 Exemplos de mudanças (primeiras 10):")
    for i, (key, old_val) in enumerate(list(old_desc.items())[:10]):
        new_val = new_desc.get(key, "NÃO ENCONTRADO")
        print(f"\n   Classe: {key}")
        print(f"      ❌ Antigo: {old_val[:70]}...")
        print(f"      ✅ Novo:   {new_val[:70]}...")


def main():
    print(f"\n{'#'*70}")
    print(f"# CORREÇÃO DE DESCRIPTORS")
    print(f"{'#'*70}\n")
    
    # Lista datasets
    if not os.path.exists(EMBEDDINGS_DIR):
        print(f"❌ Pasta de embeddings não encontrada: {EMBEDDINGS_DIR}")
        return
    
    datasets = [f.replace('.pt', '') for f in os.listdir(EMBEDDINGS_DIR) if f.endswith('.pt')]
    print(f"📊 Datasets encontrados: {len(datasets)}")
    for ds in datasets:
        print(f"   - {ds}")
    
    # Processa cada dataset
    for dataset in datasets:
        generate_descriptors_from_embeddings(dataset)
    
    print(f"\n{'='*60}")
    print(f"✅ CONCLUÍDO!")
    print(f"{'='*60}\n")
    
    # Compara os descriptors
    for dataset in datasets:
        compare_descriptors(dataset)
    
    print(f"\n{'#'*70}")
    print(f"# PRÓXIMOS PASSOS")
    print(f"{'#'*70}\n")
    
    print("""
    1. ✅ Descriptors corrigidos foram gerados
    2. ✅ Backups dos antigos foram criados (*_OLD.json)
    3. ✅ Novos descriptors copiados para a pasta original
    
    Agora execute novamente o script de avaliação zero-shot:
    
        python evaluate_zeroshot.py
    
    A acurácia deve melhorar significativamente!
    
    Resultados esperados:
    - CUB: ~50-55% (era 3%, estava 51% no paper)
    - Aircraft: ~25-30% (era 5.7%, estava 24.96% no paper)
    """)


if __name__ == "__main__":
    main()