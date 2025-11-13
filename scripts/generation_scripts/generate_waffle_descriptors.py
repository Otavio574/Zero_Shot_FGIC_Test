"""
Script para gerar descritores de FALLBACK/GENÉRICOS (ou templates WaffleCLIP padrão)
processando APENAS os datasets cujas pastas existem no diretório 'datasets/'.
NENHUM LLM ESTÁ SENDO USADO.
"""


import random
import string
import os
import json
from pathlib import Path
from typing import List, Dict

# --- CONFIGURAÇÕES DE CAMINHO ---
SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS_BASE_DIR = Path("datasets") 
OUTPUT_DIR = Path("descriptors_waffle_clip_random") 
OUTPUT_DIR.mkdir(exist_ok=True)
# -------------------------------


# ===================================
# 💡 FUNÇÕES ESSENCIAIS DE UTILIDADE
# ===================================

def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    """Carrega a configuração de datasets do summary.json."""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    except Exception:
        return {}

    datasets = {}
    if isinstance(summary, list):
        for item in summary:
            dataset_name = item.get('dataset')
            dataset_path = item.get('path')
            if dataset_name and dataset_path:
                datasets[dataset_name] = dataset_path
    return datasets


def list_existing_dataset_dirs(base_dir: Path) -> List[str]:
    """Lista os nomes de todas as pastas (datasets) presentes no diretório base."""
    print(f"🔄 Varrendo pastas em: {base_dir}")
    if not base_dir.exists():
        print(f"❌ Diretório base '{base_dir}' não encontrado.")
        return []
    return [d.name for d in base_dir.iterdir() if d.is_dir()]


def get_class_names(dataset_name: str) -> List[str]:
    """
    Obtém os nomes das classes lendo os subdiretórios dentro da pasta do dataset.
    Verifica caminhos comuns: datasets/NOME/images/ ou datasets/NOME/.
    """
    
    # 1. datasets/NOME_DO_DATASET/images/ (Comum para CUB, etc.)
    class_root_1 = DATASETS_BASE_DIR / dataset_name / "images"
    
    # 2. datasets/NOME_DO_DATASET/ (Classes diretamente na raiz)
    class_root_2 = DATASETS_BASE_DIR / dataset_name

    root_to_use = None
    if class_root_1.exists():
        root_to_use = class_root_1
    elif class_root_2.exists():
        root_to_use = class_root_2
    else:
        return []

    # Busca os subdiretórios (classes)
    classes = [d.name for d in root_to_use.iterdir() if d.is_dir()]
    
    return sorted(classes)


# ===================================
# 📝 FUNÇÃO DE GERAÇÃO DE DESCRITORES (SIMPLES)
# ===================================

def random_waffle_word(min_len=3, max_len=8):
    """Gera uma palavra aleatória estilo 'xih', 'nbacghq', 'sgwcb'."""
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_lowercase, k=length))


def random_waffle_phrase(min_words=2, max_words=4):
    """Gera uma frase curta de palavras aleatórias."""
    n_words = random.randint(min_words, max_words)
    return " ".join(random_waffle_word() for _ in range(n_words))


def clean_classname(name: str) -> str:
    """
    Converte '001.Black_footed_Albatross' → 'black footed albatross'
    """
    # Remove prefixo "001."
    if "." in name:
        name = name.split(".", 1)[1]

    # Underscores → espaços
    name = name.replace("_", " ")

    return name.lower()
    

def generate_simple_descriptors(class_name: str):
    """
    Gera *uma única string* com o template estilo WaffleCLIP real.
    Exemplo:
    "A photo of an animal: a black footed albatross, which has xih exdv."
    """

    clean_name = clean_classname(class_name)
    waffle_noise = random_waffle_phrase()

    descriptor = (
        f"A photo of an animal: a {clean_name}, which has {waffle_noise}."
    )

    return descriptor   # <- agora retorna string única



# ===================================
# 🚀 FUNÇÃO PRINCIPAL
# ===================================

def main():
    print(f"--- ⚙️ Geração de Descritores WaffleCLIP (Templates Simples com Filtro) ⚙️ ---")
    
    # 1. Carrega datasets do summary
    all_datasets_from_summary = load_datasets_from_summary(SUMMARY_PATH)
    
    # 2. Lista datasets que realmente existem na pasta 'datasets/'
    existing_dirs = list_existing_dataset_dirs(DATASETS_BASE_DIR)
    
    print(f"Datasets no summary: {len(all_datasets_from_summary)}")
    print(f"Pastas encontradas: {len(existing_dirs)}")

    # 3. Filtra: processar apenas se o dataset estiver no summary E a pasta existir
    datasets_to_process = {
        name: path 
        for name, path in all_datasets_from_summary.items() 
        if name in existing_dirs
    }

    print(f"\n✅ Total de datasets a processar (filtrados e existentes): {len(datasets_to_process)}\n")

    if not datasets_to_process:
        print("⚠️ Nenhum dataset para processar. Verifique os nomes das pastas.")
        return

    # 4. Processa cada dataset
    for dataset_name, _ in datasets_to_process.items():
        print(f"\n--- Processando dataset: **{dataset_name}** ---")
        
        # Nome do arquivo de saída WaffleCLIP
        output_file = OUTPUT_DIR / f"{dataset_name}_waffle.json"
        
        if output_file.exists():
            print(f"Arquivo de descritores WaffleCLIP já existe em {output_file}. Pulando.")
            continue

        try:
            class_names = get_class_names(dataset_name)
            if not class_names:
                print("⚠️ Nenhuma classe encontrada no diretório. Pulando.")
                continue

            print(f"Encontradas {len(class_names)} classes.")
            
            all_descriptors = {}
            
            for i, class_name in enumerate(class_names):
                print(f"    > ({i+1}/{len(class_names)}) Gerando template para: {class_name}")
                # Gera a LISTA de descritores
                descriptors_list = generate_simple_descriptors(class_name) 
                
                # Salva o resultado no dicionário (formato: {classe: [desc1, desc2]})
                all_descriptors[class_name] = descriptors_list
            
            # Salva o JSON no formato esperado
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_descriptors, f, indent=4, ensure_ascii=False)
            
            print(f"✅ Descritores WaffleCLIP salvos em {output_file}")
            
        except Exception as e:
            print(f"❌ Erro ao processar o dataset {dataset_name}: {e}")
            continue
    
    print("\n--- Processo de geração de descritores WaffleCLIP finalizado. ---")


if __name__ == "__main__":
    main()