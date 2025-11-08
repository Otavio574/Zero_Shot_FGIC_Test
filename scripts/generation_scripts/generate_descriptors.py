"""
Gera descrições (descriptors) otimizadas usando um LLM (Gemini).
Substitui a lógica de template matching DCLIP-like por geração de texto.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import torch # Necessário apenas se for usar o DCLIP (não usado aqui)
# import clip # Não precisamos mais do CLIP para gerar os descriptors
import sys # Necessário para o hack do path, caso o editor ainda reclame

# --- HACK DO PATH (Manter APENAS se o editor ainda reclamar do import) ---
# Caso contrário, pode ser removido. 
# VENV_SITE_PACKAGES_PATH = r"C:\Users\ota45\OneDrive\Área de Trabalho\Testes\venv\Lib\site-packages" 
# if VENV_SITE_PACKAGES_PATH not in sys.path:
#     sys.path.append(VENV_SITE_PACKAGES_PATH)
# -----------------------------------------------------------------------

# 1. Imports necessários para o Gemini e .env
import google.generativeai as genai
from dotenv import load_dotenv

# ==================== CONFIGURAÇÃO DO GEMINI ====================

# 2. Carrega as variáveis de ambiente (o arquivo .env)
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ ERRO: GOOGLE_API_KEY não encontrada.")
    print("  1. Crie uma chave em https://ai.google.dev/")
    print("  2. Crie um arquivo .env na raiz do projeto.")
    print("  3. Adicione a linha: GOOGLE_API_KEY='SUA_CHAVE_AQUI'")
    # exit() # Descomente se quiser parar a execução se a chave não for encontrada

# 3. Configura o modelo Gemini
genai.configure(api_key=API_KEY)
# USANDO O MODELO CORRETO PARA EVITAR ERRO 404
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# ==================== FUNÇÕES DE UTILITY ====================

def sanitize_class_name(class_name: str) -> str:
    """Saneia o nome da classe (ex: '001.Black_footed_Albatross') para um conceito legível."""
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()

def get_dataset_category(dataset_name: str) -> str:
    """Mapeia o nome do dataset para uma categoria (contexto para o LLM)."""
    dataset_normalized = dataset_name.replace("-", "_").lower()
    
    category_map = {
        "cub_200_2011": "pássaro",
        "birdsnap": "pássaro",
        "fgvc_aircraft": "avião",
        "stanford_dogs": "cachorro",
        "stanford_cars": "carro",
    }
    
    for key, category in category_map.items():
        if key in dataset_normalized or dataset_normalized in key:
            return category
            
    return "objeto" 

def load_datasets_from_summary(summary_path: str) -> Dict[str, str]:
    """Carrega configuração de datasets a partir do summary.json."""
    summary_path_obj = Path(summary_path)
    if not summary_path_obj.exists():
        return {}
        
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    datasets_config = {}
    
    items = []
    if isinstance(summary, list):
        items = [item for item in summary if isinstance(item, dict)]
    elif isinstance(summary, dict) and 'datasets' in summary:
        items = [item for item in summary['datasets'] if isinstance(item, dict)]
    
    for item in items:
        dataset_name = item.get('dataset')
        dataset_path = item.get('path')
        if dataset_name and dataset_path:
            datasets_config[dataset_name] = dataset_path
    
    return datasets_config

# ==================== CLASSE GERADORA (MODIFICADA) ====================

class LLMDescriptorGenerator:
    """Gera descrições usando o Gemini para otimização de prompt."""
    
    def __init__(self, model):
        self.model = model
        
    def generate_description_with_gemini(self, concept: str, category: str) -> str:
        """
        Gera uma descrição rica para a classe usando o LLM.
        Esta é a lógica central que substitui o template matching.
        """
        
        # 4. O Prompt de Geração (O mais importante!)
        prompt = f"""
        Estou otimizando prompts para um modelo de IA de visão (CLIP).
        Meu objetivo é gerar uma descrição de uma única frase para a classe: '{concept}', que é um(a) '{category}'.
        
        A descrição deve ser visualmente rica, focada em 2-3 características distintivas, e soar como uma legenda de foto.
        
        Responda APENAS com a frase final otimizada, começando com "a photo of a..." ou similar.
        
        Exemplo (Pássaro):
        'a photo of a blue jay, a type of bird, with a bright blue crest and black markings.'
        
        Exemplo (Avião):
        'a photo of a A380, a type of aircraft, a large double-deck wide-body jet airliner.'

        Gere agora para: '{concept}'
        """
        
        try:
            response = self.model.generate_content(prompt)
            description = response.text.strip().replace("\n", "")
            return description
        except Exception as e:
            # Em caso de erro, apenas registra e retorna o fallback.
            print(f"  ❌ Erro na API Gemini (Classe: {concept}): {e}")
            # A pausa principal de 6.5s é tratada no loop chamador.
            return f"a photo of a {concept}, a type of {category}."


    def process_dataset_by_class(self, 
                                 dataset_name: str, 
                                 dataset_path: str,
                                 output_path: str) -> Dict[str, str]:
        """
        Processa dataset e gera UM descriptor otimizado por LLM para cada classe.
        """
        dataset_path_obj = Path(dataset_path)
        category = get_dataset_category(dataset_name)
        
        class_folders = sorted([d for d in dataset_path_obj.iterdir() if d.is_dir()])
        
        if not class_folders:
            print(f"⚠️  Nenhuma pasta de classe encontrada em {dataset_path_obj}")
            return {}
        
        print(f"\n🎨 Processando dataset: {dataset_name} | Categoria: **{category.upper()}**")
        print(f"  Classes encontradas: {len(class_folders)}")
        
        descriptors = {}
        failed = []
        
        for class_folder in tqdm(class_folders, desc=f"Gerando {dataset_name} (Gemini)"):
            class_name_raw = class_folder.name 
            concept = sanitize_class_name(class_name_raw) 
            
            try:
                description = self.generate_description_with_gemini(concept, category)
                
                descriptors[class_name_raw] = description
                
                # Rate limiting para a API gratuita do Gemini
                # 6.5s > 6s (60s/10RPM) para respeitar o limite de cota 429.
                time.sleep(6.5) 
                
            except Exception as e:
                # O erro 429 (Quota Exceeded) é o mais comum. Pausamos extra.
                failed.append({"class": class_folder.name, "error": str(e)})
                print("⚠️  Pausa extra de 15s devido à falha na API (possível 429).")
                time.sleep(15) 
                
        # Salva JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptors, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Geração Gemini Concluída para {dataset_name}!")
        print(f"  Classes processadas: {len(descriptors)} | Falhas: {len(failed)}")
        print(f"  Salvo em: **{output_path}**")
            
        return descriptors

# ==================== FUNÇÃO PRINCIPAL DE EXECUÇÃO ====================

def generate_descriptors_for_datasets(
    datasets_config: Dict[str, str],
    output_dir: str = "descriptors_gemini_optimized", # Novo nome
    model_instance = None
):
    """Gera descriptors Gemini para múltiplos datasets."""
    
    if model_instance is None:
        print("❌ Instância do modelo Gemini não fornecida.")
        return

    generator = LLMDescriptorGenerator(model=model_instance)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 Iniciando geração de descriptors **GEMINI OTIMIZADOS**")
    print(f"📊 Datasets: {len(datasets_config)}")
    print(f"📁 Output: {output_dir}")
    print(f"{'='*70}\n")
    
    for dataset_name, dataset_path in datasets_config.items():
        output_path = os.path.join(output_dir, f"{dataset_name}_gemini_descriptors.json")
        generator.process_dataset_by_class(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            output_path=output_path,
        )
    
    print(f"\n{'='*70}")
    print(f"✅ Geração Gemini Finalizada!")
    print(f"📂 Verifique a pasta: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    
    # 6. Verifica se a API Key foi carregada antes de continuar
    if not API_KEY:
        print("Encerrando. Chave API do Google não configurada.")
    else:
        SUMMARY_PATH = "outputs/analysis/summary.json"
        OUTPUT_DIR = "descriptors_gemini_optimized"
        
        datasets = load_datasets_from_summary(SUMMARY_PATH)
        
        if not datasets:
            print("❌ Não foi possível carregar datasets do summary.json.")
        else:
            generate_descriptors_for_datasets(
                datasets_config=datasets,
                output_dir=OUTPUT_DIR,
                model_instance=gemini_model
            )
        
        print("\n💡 Próxima Ação:")
        print(f"1. Verifique os JSONs em **{OUTPUT_DIR}** (eles devem ter descrições ricas).")
        print(f"2. Execute o script de avaliação (evaluate_clip_zero-shot_description.py) apontando para este novo diretório.")