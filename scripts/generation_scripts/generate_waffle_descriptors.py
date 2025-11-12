"""
Gerador de descriptors para ZERO-SHOT CLIP (Waffle/Noisy CLIP - Versão Original do Paper)
Substitui o ruído semântico por ruído aleatório (caracteres ou palavras sem sentido)
para testar a robustez do CLIP.

Formato: "A photo of an {category}: a {class_name}, which has {random_noise}."
"""

import os
import json
import random
import string
from pathlib import Path
from typing import Dict, List

# ------------------------------------------------------------------------
# NOVO: GERAÇÃO DE RUÍDO ALEATÓRIO
# ------------------------------------------------------------------------

def generate_random_noise(method: str = 'words', length: int = 2) -> str:
    """Gera ruído aleatório (palavras sem sentido ou caracteres)."""
    if method == 'chars':
        # Ex: "jmhj, !J#m"
        chars = string.ascii_letters + string.punctuation + string.digits + ' '
        noise = ''.join(random.choice(chars) for _ in range(length * 5))
        return noise.strip()
    else: # method == 'words'
        # Ex: "foot loud"
        random_words = []
        for _ in range(length):
            # Gera uma "palavra" aleatória de 3 a 7 caracteres
            word_length = random.randint(3, 7)
            word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
            random_words.append(word)
        return ' '.join(random_words)

def clean_class_name(class_name: str) -> str:
    """Limpa o nome da classe para texto natural"""
    name = class_name
    
    # Remove números no início (ex: "001.Black_footed_Albatross")
    if '.' in name and name.split('.')[0].isdigit():
        name = name.split('.', 1)[1]
    
    # Substitui underscores e hífens por espaços
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Remove espaços extras
    name = ' '.join(name.split())
    
    return name.lower()


def detect_dataset_category(dataset_name: str) -> str:
    """Detecta a categoria do dataset"""
    name_lower = dataset_name.lower()
    
    if 'cub' in name_lower or 'bird' in name_lower or 'nabirds' in name_lower:
        return 'bird'
    elif 'aircraft' in name_lower or 'plane' in name_lower or 'fgvc' in name_lower:
        return 'aircraft'
    elif 'car' in name_lower or 'vehicle' in name_lower or 'stanford' in name_lower:
        return 'car'
    elif 'dog' in name_lower or 'pet' in name_lower:
        return 'dog'
    elif 'flower' in name_lower or 'oxford' in name_lower:
        return 'flower'
    elif 'food' in name_lower:
        return 'food'
    else:
        return 'object'


def get_category_word(category: str) -> str:
    """Retorna a palavra de categoria (animal, object, etc.) para o template Waffle."""
    categories = {
        'bird': "animal",
        'aircraft': "object",
        'car': "object",
        'dog': "animal",
        'flower': "plant",
        'food': "food item",
        'object': "object"
    }
    return categories.get(category, categories['object'])


def generate_descriptors_from_folders(dataset_path: str, dataset_name: str) -> Dict[str, str]:
    """
    Gera descriptors para o Waffle CLIP usando template fixo e ruído aleatório.
    """
    dataset_path_obj = Path(dataset_path)
    
    if not dataset_path_obj.exists():
        print(f"❌ Path não encontrado: {dataset_path}")
        return {}
    
    class_folders = [d for d in dataset_path_obj.iterdir() if d.is_dir()]
    
    if not class_folders:
        print(f"⚠️  Nenhuma pasta de classe encontrada em {dataset_path}")
        return {}
    
    print(f"\n📂 Dataset: {dataset_name}")
    print(f"  Classes encontradas: {len(class_folders)}")
    
    category = detect_dataset_category(dataset_name)
    category_word = get_category_word(category)
    
    # Template para Waffle CLIP: "A photo of an {category_word}: a {class_name}, which has {noise}."
    template = "A photo of an {category_word}: a {class_name}, which has {noise}."
    
    print(f"  Categoria (Waffle): {category_word}")
    print(f"  Template Base: {template}")
    print(f"  Modo: Waffle CLIP (Ruído Aleatório)")
    
    descriptors = {}
    
    for class_folder in sorted(class_folders):
        class_name_raw = class_folder.name
        class_name_clean = clean_class_name(class_name_raw)
        
        # 1. Gera ruído (usando palavras aleatórias)
        random_noise = generate_random_noise(method='words', length=random.randint(2, 3))
        
        # 2. Gera o descriptor final
        final_description = template.format(
            category_word=category_word,
            class_name=class_name_clean,
            noise=random_noise
        )
        
        descriptors[class_name_raw] = final_description
    
    # Mostra exemplos
    print(f"\n  📋 Exemplos (primeiras 10):")
    for i, (cls, desc) in enumerate(list(descriptors.items())[:10]):
        print(f"      {cls:40s} → {desc}")
    
    print(f"\n  ✅ Total: {len(descriptors)} descriptors gerados")
    
    return descriptors


def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    """Carrega configuração de datasets do summary.json (Função inalterada)"""
    if not summary_path.exists():
        print(f"❌ Arquivo não encontrado: {summary_path}")
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
        if 'datasets' in summary:
            for item in summary['datasets']:
                dataset_name = item.get('dataset')
                dataset_path = item.get('path')
                if dataset_name and dataset_path:
                    datasets[dataset_name] = dataset_path
        else:
            datasets = summary
    
    return datasets


def main():
    # Usar uma seed fixa para garantir ruído replicável entre execuções
    random.seed(42) 
    
    print(f"\n{'#'*70}")
    print(f"# GERADOR WAFFLE CLIP (Ruído Aleatório Semântico)")
    print(f"{'#'*70}\n")
    
    # Configuração
    SUMMARY_PATH = Path("outputs/analysis/summary.json")
    OUTPUT_DIR = "descriptors_waffle_clip_random" # Novo diretório para isolar
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega datasets
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    
    if not datasets:
        print("❌ Nenhum dataset encontrado no summary.json")
        return
    
    print(f"📊 Datasets encontrados: {len(datasets)}")
    
    print(f"\n{'='*70}")
    print(f"PROCESSANDO DATASETS")
    print(f"{'='*70}\n")
    
    # Processa cada dataset
    all_results = {}
    
    for dataset_name, dataset_path in datasets.items():
        try:
            descriptors = generate_descriptors_from_folders(dataset_path, dataset_name)
            
            if descriptors:
                output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_waffle_random.json")
                
                if os.path.exists(output_path):
                    backup_path = output_path.replace('.json', '_OLD.json')
                    os.rename(output_path, backup_path)
                    print(f"  📦 Backup criado: {backup_path}")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(descriptors, f, indent=2, ensure_ascii=False)
                
                print(f"  💾 Salvo em: {output_path}")
                
                all_results[dataset_name] = len(descriptors)
            else:
                all_results[dataset_name] = 0
                
        except Exception as e:
            print(f"  ❌ Erro: {e}")
            all_results[dataset_name] = 0
    
    # Resumo final
    print(f"\n{'='*70}")
    print(f"✅ CONCLUSÃO")
    print(f"{'='*70}\n")
    
    print(f"Resumo dos descriptors gerados:")
    for dataset_name, count in all_results.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {dataset_name:30s}: {count} classes")
    
    print(f"\n📁 Descriptors salvos em: {OUTPUT_DIR}/")
    
    print(f"\n{'#'*70}")
    print(f"# PRÓXIMOS PASSOS")
    print(f"{'#'*70}\n")
    
    print(f"""
    ✅ Descriptors Waffle CLIP (Ruído Aleatório) gerados!
    
    Exemplo CUB:
    "001.Black_footed_Albatross" → "A photo of an animal: a black footed albatross, which has jhjeyv asdgv."
    
    Agora, execute a avaliação zero-shot usando o arquivo gerado (ex: {OUTPUT_DIR}/{list(datasets.keys())[0]}_waffle_random.json).
    
    Resultados esperados: De acordo com o artigo, esta abordagem pode superar a performance do DCLIP.
    """)


if __name__ == "__main__":
    main()