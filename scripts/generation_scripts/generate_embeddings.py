import os
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import json
import traceback
# Não precisamos do 'from torchvision import transforms' se usarmos o CLIPProcessor corretamente
from transformers import CLIPProcessor, CLIPModel

# ============================
# CONFIGURAÇÕES GERAIS
# ============================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
DATASETS_DIR = Path("datasets") # Diretório raiz onde os datasets estão
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================
# CARREGAR MODELO CLIP
# ============================

print("🚀 Carregando modelo CLIP (baseline)...")
# Usa GPU se disponível, caso contrário, usa CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device) # type: ignore

# 🚨 CORREÇÃO 1: Forçar o uso do processador ideal (fast) para evitar o warning
# e garantir o pré-processamento estrito.
processor = CLIPProcessor.from_pretrained(model_name, use_fast=True) 

model.eval() # Modo de avaliação (sem treinamento)
print(f"✅ Modelo carregado! (Device: {device})")

# ============================
# CARREGAR LISTA DE DATASETS
# ============================

try:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        datasets_summary = json.load(f)
except FileNotFoundError:
    print(f"❌ Erro: Arquivo de summary não encontrado em {SUMMARY_PATH}")
    datasets_summary = []
except json.JSONDecodeError:
    print(f"❌ Erro: Falha ao ler o JSON em {SUMMARY_PATH}")
    datasets_summary = []

# Normaliza a estrutura para uma lista de dicionários com 'dataset' e 'path'
if isinstance(datasets_summary, dict) and 'datasets' in datasets_summary:
    datasets = [{'dataset': name, 'path': path} for name, path in datasets_summary['datasets'].items()]
elif isinstance(datasets_summary, list):
    datasets = datasets_summary
else:
    datasets = []
    
# ============================
# FUNÇÕES AUXILIARES
# ============================

def get_image_paths(folder, exts=(".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
    """Busca recursiva de imagens em subpastas"""
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths


def generate_embeddings_for_dataset(dataset_name: str, dataset_path: Path, limit: int = -1):
    """Gera e salva embeddings de imagem para um dataset"""
    print(f"\n📦 Processando dataset: {dataset_name}")
    image_paths = get_image_paths(dataset_path)

    # 🚨 CORREÇÃO CRÍTICA: Ordenação Consistente de Caminhos 🚨
    image_paths.sort() 
    # ----------------------------------------------------

    if len(image_paths) == 0:
        print(f"⚠️ Nenhuma imagem encontrada em {dataset_path}")
        return False

    # Define as imagens a processar (usa todas se limit for -1)
    paths_to_process = image_paths if limit == -1 else image_paths[:limit]
    
    all_embeds = []
    valid_paths = []
    skipped_paths_count = 0

    print(f"🖼️ Total de imagens a processar: {len(paths_to_process)}")

    for img_path in tqdm(paths_to_process, desc=f"🔹 {dataset_name}"):
        try:
            image = Image.open(img_path).convert("RGB")
            # Usa o processor do CLIP para pré-processar. Retorna [1, C, H, W]
            inputs = processor(images=image, return_tensors="pt").to(device) # type: ignore
            
            with torch.no_grad():
                # Lógica de Geração do Embedding. Retorna [1, 512]
                embeds = model.get_image_features(**inputs)
                
                # ❌ REMOVIDO: embeds.squeeze(0). MANTENHA O SHAPE [1, 512]
                
                # Normaliza o embedding (essencial para o CLIP).
                # A normalização é aplicada na dimensão do embedding (-1), mantendo o shape [1, 512].
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                
            # Adicionamos o tensor [1, 512] à lista.
            all_embeds.append(embeds.cpu())
            valid_paths.append(img_path)
            
        except Exception as e:
            skipped_paths_count += 1
            # Opcional: printar uma amostra de erros
            if skipped_paths_count <= 5: 
                print(f"\n[Erro] Ignorando imagem {Path(img_path).name}: {e}")
            continue

    if skipped_paths_count > 0:
        print(f"\n❌ Atenção: {skipped_paths_count} imagens foram ignoradas devido a erros na leitura/processamento.")
        
    if all_embeds:
        # 🚨 CORREÇÃO DA CONCATENAÇÃO: Concatenar tensores [1, 512] na dimensão 0 resulta em [N, 512].
        all_embeds = torch.cat(all_embeds, dim=0) 
        out_path = OUTPUT_DIR / f"{dataset_name}.pt"
        
        print(f"\n--- Verificação Final ---")
        print(f"Total de Embeds gerados: {len(all_embeds)}")
        
        # O print do shape agora deve ser [N, 512]
        print(f"Shape final de Embeds: {all_embeds.shape}") 
        print(f"Caminho de Saída: {out_path.resolve()}")
        
        try:
            # Tenta salvar
            torch.save({
                "image_embeddings": all_embeds,
                "image_paths": valid_paths
            }, out_path)
            
# ... (o restante da função é o mesmo) ...
            # Confirma que salvou
            if out_path.exists():
                print(f"💾 ✅ Embeddings SALVOS COM SUCESSO em {out_path.name} ({len(all_embeds)} imagens).")
                return True
            else:
                # Caso o torch.save não levante exceção, mas o arquivo não apareça
                print(f"❌ Falha: A função torch.save retornou, mas o arquivo {out_path.name} não foi criado/encontrado.")
                return False
                
        except Exception as e:
            print(f"❌ ERRO GRAVE ao tentar salvar o arquivo {out_path.name}: {e}")
            print("Verifique permissões de escrita, espaço em disco ou se o caminho é válido.")
            traceback.print_exc()
            return False
            
    else:
        print(f"⚠️ Nenhuma embedding gerada para {dataset_name}")
        return False

# ============================
# LOOP PRINCIPAL INCREMENTAL
# ============================

def main():
    print("\n🔍 Verificando datasets...")
    
    # 🚨 NOVO BLOCO: Forçar a regeneração dos datasets problemáticos 🚨
    # Isso garante que o CUB e o FGVC serão processados com o script corrigido.
    datasets_to_regenerate = {"CUB_200_2011", "FGVC_Aircraft"}
    
    total = len(datasets)
    
    # Filtra datasets que precisam ser processados (regenerar ou ainda não existem)
    new_datasets = []
    for ds in datasets:
        name = ds["dataset"]
        # Se estiver na lista de regeneração OU o arquivo .pt não existir
        if name in datasets_to_regenerate or not (OUTPUT_DIR / f"{name}.pt").exists():
            new_datasets.append(ds)

    print(f"📊 Total de datasets no summary: {total}")
    print(f"✅ Já processados (e não forçados a regenerar): {total - len(new_datasets)}")
    print(f"🚀 Para processar/regenerar: {len(new_datasets)} → {[d['dataset'] for d in new_datasets]}")

    for ds in new_datasets:
        name = ds["dataset"]
        path = Path(ds["path"])

        if not path.exists():
            print(f"⚠️ Caminho inválido para {name}: {path}")
            continue

        # Nota: O limite aqui é -1, processando todas as imagens por padrão.
        success = generate_embeddings_for_dataset(name, path, limit=-1) 
        if not success:
            print(f"❌ Falha ao gerar embeddings para {name}")

    print("\n🏁 Finalizado! Embeddings atualizados em /embeddings.")


if __name__ == "__main__":
    main()