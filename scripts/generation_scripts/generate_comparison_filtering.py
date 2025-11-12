"""
Etapa Final DCLIP: Filtering (Filtro)
Implementa a Seção 3.3 do paper. Filtra descritores comparativos
que são visualmente inconsistentes, usando o prompt base como limiar.

MODIFICAÇÕES:
1. Carregamento robusto de 'SUMMARY_PATH' para obter nomes de dataset corretos.
2. Função 'get_class_label_map' para mapear nomes de classe para índices.
3. Função 'calculate_class_prototypes' corrigida para:
    a) Carregar 'image_embeddings' e 'image_paths'.
    b) Reconstruir o tensor de labels numéricos a partir dos caminhos.
    c) Calcular corretamente os protótipos de classe (média).
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# CAMINHO PARA SEUS DESCRIÇÕES GERADAS (JSON)
# ⚠️ Ajuste este caminho se necessário
INPUT_DESCRIPTORS_DIR = Path("descriptors_comparative_rag")

# CAMINHO ONDE ESTÃO SEUS EMBEDDINGS DE IMAGEM
EMBEDDINGS_DIR = Path("embeddings")

# CAMINHO DE SAÍDA PARA OS DESCRITORES FILTRADOS
OUTPUT_FILTERED_DIR = Path("descriptors_comparative_filtered")

# CAMINHO PARA O ARQUIVO QUE LISTA SEUS DATASETS E PASTAS
SUMMARY_PATH = Path("outputs/analysis/summary.json")

# Modelo CLIP para gerar Text Embeddings (deve ser o mesmo da avaliação!)
CLIP_MODEL = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Carregando CLIP para Text Embeddings...")
clip_model = AutoModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()
print("✅ CLIP carregado!")

# ============================================================
# FUNÇÕES AUXILIARES DE CARREGAMENTO E MAPEAMENTO
# ============================================================

def sanitize_class_name(class_name: str) -> str:
    """Função para limpar os nomes das classes (igual à do gerador)."""
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()

def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    """Carrega o mapeamento DatasetName -> Path do arquivo summary.json."""
    if not summary_path.exists():
        print(f"⚠️ Aviso: Arquivo summary.json não encontrado em {summary_path}.")
        return {}
    
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar summary.json: {e}")
        return {}
    
    datasets = {}
    if isinstance(data, list):
        for d in data:
            if "dataset" in d:
                # Usa o nome do dataset como chave
                datasets[d["dataset"]] = d.get("path", "") 
    
    return datasets

def get_class_label_map(class_labels: List[str]) -> Dict[str, int]:
    """Cria um mapeamento de nome da classe (string) para índice numérico (int)."""
    return {name: i for i, name in enumerate(class_labels)}


def get_text_embedding(texts: List[str]) -> torch.Tensor:
    """Gera embeddings de texto normalizados (para uso com protótipos)."""
    with torch.no_grad():
        inputs = clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)
        
        embeddings = clip_model.get_text_features(**inputs)
        # Normaliza
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

# ============================================================
# FUNÇÃO CENTRAL DE CÁLCULO
# ============================================================

def calculate_class_prototypes(
    dataset_name: str,
    class_labels: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Calcula o Protótipo da Classe (Feature de Imagem Média)
    Reconstrói os labels numéricos a partir dos caminhos das imagens.
    """
    embeddings_path = EMBEDDINGS_DIR / f"{dataset_name}.pt"
    if not embeddings_path.exists():
        print(f"❌ Erro: Arquivo de embeddings de imagem não encontrado em {embeddings_path}.")
        return {}

    data = torch.load(embeddings_path)
    
    # Validação e Carregamento das chaves
    if 'image_embeddings' not in data or 'image_paths' not in data:
        print(f"❌ Erro: Arquivo {embeddings_path.name} não contém as chaves 'image_embeddings' e/ou 'image_paths'.")
        return {}
        
    image_features = data['image_embeddings'].to(DEVICE)
    image_paths: List[str] = data['image_paths']
    
    # 1. Reconstruir o Tensor de Labels Numéricos a partir dos caminhos
    name_to_idx = get_class_label_map(class_labels)
    image_labels_list = []
    
    for path in image_paths:
        try:
            # Assume que o nome da pasta da classe é o nome completo (Ex: '001.Black_footed_Albatross')
            class_folder_name = Path(path).parent.name
            image_labels_list.append(name_to_idx[class_folder_name])
        except KeyError:
            # Se o nome da pasta não estiver na lista de classes (algo estranho)
            image_labels_list.append(-1)
            
    # Converte a lista de labels para um tensor
    image_labels = torch.tensor(image_labels_list, dtype=torch.long).to(DEVICE)
    
    # Normaliza as features da imagem (crucial para Cosine Similarity)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    prototypes = {}
    
    for i, class_name in enumerate(class_labels):
        # Seleciona as features da classe atual
        mask = (image_labels == i)
        
        # Filtra apenas as features da classe i
        class_features = image_features[mask]
        
        if class_features.numel() == 0:
            # Este aviso é normal se a reconstrução falhou para esta classe específica
            continue
            
        # Calcula a média (o protótipo)
        prototype = class_features.mean(dim=0, keepdim=True)
        prototypes[class_name] = prototype
        
    return prototypes


def filter_descriptors(
    dataset_name: str,
    descriptors: Dict[str, List[str]],
    prototypes: Dict[str, torch.Tensor]
) -> Dict[str, List[str]]:
    """
    Aplica a filtragem Top-k e de Limiar (Lower Bound) para cada classe.
    """
    filtered_descriptors = {}
    
    print(f"   Aplicando filtro de Limiar ({len(prototypes)} classes)...")
    
    # Use list(descriptors.items()) para evitar modificação durante iteração
    for class_raw, comp_descriptors in tqdm(descriptors.items(), desc="Filtrando"):
        
        if class_raw not in prototypes:
            # Não é possível filtrar se não há protótipo (ex: classe sem imagens válidas)
            # Mantém os descritores originais, mas isso pode ser arriscado
            filtered_descriptors[class_raw] = comp_descriptors
            continue
            
        prototype = prototypes[class_raw]
        class_clean = sanitize_class_name(class_raw)
        
        # 1. Calcular Similaridade Base (Lower Bound)
        base_prompt = [f"A photo of a {class_clean}."]
        base_embed = get_text_embedding(base_prompt)
        sim_base = F.cosine_similarity(prototype, base_embed).item()

        # 2. Calcular Similaridade dos Descritores Gerados
        comp_embeds = get_text_embedding(comp_descriptors)
        sim_scores = F.cosine_similarity(prototype, comp_embeds)
        
        retained_descriptors = []
        
        # 3. Aplicar a Filtragem (Retém apenas descritores melhores que o Limiar Base)
        for score, descriptor in zip(sim_scores, comp_descriptors):
            if score.item() >= sim_base:
                retained_descriptors.append(descriptor)
                
        # 4. Regra de Queda (Fallback)
        if not retained_descriptors:
            # Se nenhum descritor for mantido, usa o prompt base
            retained_descriptors.append(base_prompt[0]) 
            
        filtered_descriptors[class_raw] = retained_descriptors
        
    return filtered_descriptors

# ============================================================
# MAIN
# ============================================================

def main():
    
    os.makedirs(OUTPUT_FILTERED_DIR, exist_ok=True)
    
    # Novo: Carrega o mapeamento de datasets do summary.json
    dataset_map = load_datasets_from_summary(SUMMARY_PATH)
    if not dataset_map:
        print(f"❌ Não foi possível carregar o mapeamento de datasets de {SUMMARY_PATH}. Abortando.")
        return
        
    print(f"\n{'='*70}")
    print("🧠 INICIANDO FILTRAGEM DCLIP (Visual Validation)")
    print(f"    Limiar: Similaridade com o prompt base do CLIP.")
    print(f"{'='*70}\n")
    
    # 1. Encontrar e processar todos os arquivos de descritores gerados
    descriptor_files = list(INPUT_DESCRIPTORS_DIR.glob("*.json"))

    if not descriptor_files:
        print(f"❌ Nenhum arquivo JSON de descritores encontrado em {INPUT_DESCRIPTORS_DIR}. Abortando.")
        return

    for file_path in descriptor_files:
        
        # 2. Lógica: Encontrar o nome do dataset completo (Ex: CUB_200_2011)
        dataset_name = None
        for full_name in dataset_map.keys():
            # Verifica se o nome completo do dataset está contido no nome do arquivo JSON
            if full_name in file_path.stem:
                dataset_name = full_name
                break
        
        if not dataset_name:
            print(f"❌ Aviso: Não foi possível identificar o dataset para o arquivo {file_path.name} no summary. Pulando.")
            continue
            
        print(f"\n--- Processando Dataset: {dataset_name} ---")
        
        # A. Carregar Descritores Gerados
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                descriptors = json.load(f)
            class_labels = sorted(list(descriptors.keys())) # As classes precisam estar ordenadas para o mapeamento
        except Exception as e:
            print(f"❌ Erro ao carregar {file_path}: {e}")
            continue

        # B. Calcular Protótipos (Protótipo agora usa o dataset_name correto, e.g., 'CUB_200_2011')
        prototypes = calculate_class_prototypes(dataset_name, class_labels)
        
        if not prototypes:
            print(f"❌ Não foi possível calcular protótipos para {dataset_name}. Pulando a filtragem.")
            continue
            
        # C. Aplicar a Filtragem
        filtered_results = filter_descriptors(dataset_name, descriptors, prototypes)
        
        # D. Salvar Resultados Filtrados
        output_path = OUTPUT_FILTERED_DIR / f"{dataset_name}_comparative_filtered.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=2, ensure_ascii=False)
            
        print(f"   ✅ Filtragem concluída e salva em: {output_path}")

    print(f"\n{'='*70}")
    print(f"✅ FILTRAGEM DCLIP CONCLUÍDA PARA TODOS OS DATASETS.")
    print(f"📁 Novos arquivos de descritores prontos para avaliação em {OUTPUT_FILTERED_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()