import json
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm

def sanitize_cub_concept(class_name: str) -> str:
    """
    Saneia o nome da classe CUB (ex: '001.Black_footed_Albatross') para um conceito legível
    (ex: 'black footed albatross').
    """
    # 1. Remove prefixo numérico
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    
    # 2. Converte para formato legível (espaços, minúsculas)
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()

def generate_zero_shot_descriptors(
    dataset_path: str,
    output_path: str
) -> Dict[str, str]:
    """
    Gera descriptors de texto zero-shot (prompt engineering) diretamente dos
    nomes das pastas, sem Template Matching.
    
    O template fixo é o melhor para Zero-Shot baseline (Ex: "a photo of the X").
    """
    dataset_path_obj = Path(dataset_path)
    
    # Encontra todas as subpastas (classes)
    class_folders = [d for d in dataset_path_obj.iterdir() if d.is_dir()]
    
    if not class_folders:
        print(f"⚠️ Nenhuma pasta de classe encontrada em {dataset_path}")
        return {}
    
    print(f"\n🎨 Gerando {len(class_folders)} descriptors Zero-Shot por nome da classe...")
    
    descriptors = {}
    
    # 🚨 TEMPLATE FORTE E FIXO para Zero-Shot (recomendado por CLIP)
    template_format = "a photo of the {} bird, spotted in its natural habitat."

    for class_folder in tqdm(class_folders, desc="Gerando Descriptors"):
        # Chave JSON: Nome EXATO da pasta (ex: '001.Black_footed_Albatross')
        class_key = class_folder.name 
        
        # Nome legível: 'black footed albatross'
        readable_name = sanitize_cub_concept(class_key)
        
        # Descriptor: "a photo of the black footed albatross bird, spotted in its natural habitat."
        descriptor_text = template_format.format(readable_name)
        
        descriptors[class_key] = descriptor_text
        
    # Salva JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptors, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Concluído! Descriptors gerados diretamente a partir dos nomes das classes.")
    print(f"💾 Salvo em: {output_path}")
    
    return descriptors

# ============== NOVO FLUXO DE USO (dentro do seu main) ==============

if __name__ == "__main__":
    # Supondo que você tem seu 'datasets' carregado do summary.json
    
    # EXECUTAR PARA CADA DATASET:
    dataset_name = "CUB_200_2011"
    dataset_path = "./datasets/CUB_200_2011" # Substitua pelo caminho real
    output_dir = "descriptors"
    output_path = os.path.join(output_dir, f"{dataset_name}_descriptors.json")
    
    # Executa a Geração Direta
    generate_zero_shot_descriptors(
        dataset_path=dataset_path,
        output_path=output_path
    )
    
    print("\n🎉 Novo JSON de descriptors está pronto para a avaliação zero-shot!")
    # Lembre-se de implementar as correções de saneamento na sua função de avaliação também.