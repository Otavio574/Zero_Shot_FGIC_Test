"""
Gerador SIMPLES de descriptors para zero-shot CLIP
NÃO usa CLIP para gerar - apenas templates com nomes de classes
"""

import os
import json
from pathlib import Path
from typing import Dict, List

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


def get_template_for_category(category: str) -> str:
    """Retorna o template apropriado para cada categoria"""
    templates = {
        'bird': "a photo of a {}, a type of bird.",
        'aircraft': "a photo of a {}, a type of aircraft.",
        'car': "a photo of a {}, a type of car.",
        'dog': "a photo of a {}, a type of dog.",
        'flower': "a photo of a {}, a type of flower.",
        'food': "a photo of {}, a type of food.",
        'object': "a photo of a {}."
    }
    return templates.get(category, templates['object'])


def generate_descriptors_from_folders(dataset_path: str, dataset_name: str) -> Dict[str, str]:
    """
    Gera descriptors simples baseados nos nomes das pastas de classes
    SEM usar CLIP - apenas templates fixos
    """
    dataset_path_obj = Path(dataset_path)
    
    if not dataset_path_obj.exists():
        print(f"❌ Path não encontrado: {dataset_path}")
        return {}
    
    # Encontra todas as pastas de classes
    class_folders = [d for d in dataset_path_obj.iterdir() if d.is_dir()]
    
    if not class_folders:
        print(f"⚠️  Nenhuma pasta de classe encontrada em {dataset_path}")
        return {}
    
    print(f"\n📂 Dataset: {dataset_name}")
    print(f"   Path: {dataset_path}")
    print(f"   Classes encontradas: {len(class_folders)}")
    
    # Detecta categoria e seleciona template
    category = detect_dataset_category(dataset_name)
    template = get_template_for_category(category)
    
    print(f"   Categoria: {category}")
    print(f"   Template: {template}")
    
    # Gera descriptors
    descriptors = {}
    
    for class_folder in sorted(class_folders):
        class_name_raw = class_folder.name
        class_name_clean = clean_class_name(class_name_raw)
        
        # Gera descrição simples
        description = template.format(class_name_clean)
        
        # Usa o nome ORIGINAL da pasta como chave (importante para matching!)
        descriptors[class_name_raw] = description
    
    # Mostra exemplos
    print(f"\n   📋 Exemplos (primeiras 10):")
    for i, (cls, desc) in enumerate(list(descriptors.items())[:10]):
        print(f"      {cls:40s} → {desc}")
    
    print(f"\n   ✅ Total: {len(descriptors)} descriptors gerados")
    
    return descriptors


def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    """Carrega configuração de datasets do summary.json"""
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
    print(f"\n{'#'*70}")
    print(f"# GERADOR SIMPLES DE DESCRIPTORS (SEM CLIP)")
    print(f"{'#'*70}\n")
    
    # Configuração
    SUMMARY_PATH = Path("outputs/analysis/summary.json")
    OUTPUT_DIR = "descriptors"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carrega datasets
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    
    if not datasets:
        print("❌ Nenhum dataset encontrado no summary.json")
        return
    
    print(f"📊 Datasets encontrados: {len(datasets)}")
    for name in datasets.keys():
        print(f"   - {name}")
    
    print(f"\n{'='*70}")
    print(f"PROCESSANDO DATASETS")
    print(f"{'='*70}\n")
    
    # Processa cada dataset
    all_results = {}
    
    for dataset_name, dataset_path in datasets.items():
        try:
            descriptors = generate_descriptors_from_folders(dataset_path, dataset_name)
            
            if descriptors:
                # Salva descriptors
                output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_templates.json")
                
                # Backup se já existe
                if os.path.exists(output_path):
                    backup_path = output_path.replace('.json', '_OLD.json')
                    os.rename(output_path, backup_path)
                    print(f"   📦 Backup criado: {backup_path}")
                
                # Salva novo
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(descriptors, f, indent=2, ensure_ascii=False)
                
                print(f"   💾 Salvo em: {output_path}")
                
                all_results[dataset_name] = len(descriptors)
            else:
                print(f"   ⚠️  Nenhum descriptor gerado")
                all_results[dataset_name] = 0
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            all_results[dataset_name] = 0
    
    # Resumo final
    print(f"\n{'='*70}")
    print(f"✅ CONCLUSÃO")
    print(f"{'='*70}\n")
    
    print(f"Resumo dos descriptors gerados:")
    for dataset_name, count in all_results.items():
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {dataset_name:30s}: {count} classes")
    
    print(f"\n📁 Descriptors salvos em: {OUTPUT_DIR}/")
    
    print(f"\n{'#'*70}")
    print(f"# VERIFICAÇÃO RÁPIDA")
    print(f"{'#'*70}\n")
    
    # Verifica alguns exemplos
    for dataset_name in list(datasets.keys())[:2]:
        desc_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_descriptors.json")
        if os.path.exists(desc_path):
            with open(desc_path, 'r', encoding='utf-8') as f:
                descs = json.load(f)
            
            print(f"\n📋 {dataset_name} - Primeiros 5 descriptors:")
            for i, (cls, desc) in enumerate(list(descs.items())[:5]):
                print(f"   {cls:40s} → {desc}")
    
    print(f"\n{'#'*70}")
    print(f"# PRÓXIMOS PASSOS")
    print(f"{'#'*70}\n")
    
    print("""
    ✅ Descriptors CORRETOS foram gerados!
    
    Agora execute a avaliação zero-shot:
    
        python evaluate_zeroshot.py
    
    Resultados esperados:
    - CUB-200-2011 (200 classes): ~50-55%
    - FGVC Aircraft (70 classes): ~25-30%
    
    Os descriptors agora usam os NOMES REAIS das classes,
    ao invés de tentar adivinhar com CLIP.
    
    Exemplo CUB:
    "001.Black_footed_Albatross" → "a photo of a black footed albatross, a type of bird."
    
    Exemplo Aircraft:
    "A320" → "a photo of a a320, a type of aircraft."
    """)


if __name__ == "__main__":
    main()