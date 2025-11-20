"""
Script para re-extrair embeddings garantindo imagens COLORIDAS
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "embeddings"

def check_image_mode(image_path):
    """Verifica o modo de cor da imagem"""
    img = Image.open(image_path)
    return img.mode

def convert_to_rgb(image_path):
    """Converte imagem para RGB se necessário"""
    img = Image.open(image_path)
    
    # Se já é RGB, retorna
    if img.mode == 'RGB':
        return img
    
    # Converte para RGB
    return img.convert('RGB')

def extract_embeddings_rgb(dataset_path, dataset_name):
    """Extrai embeddings garantindo que imagens sejam RGB"""
    
    print(f"\n{'='*70}")
    print(f"🎨 EXTRAÇÃO COM IMAGENS COLORIDAS: {dataset_name}")
    print(f"{'='*70}\n")
    
    dataset_path = Path(dataset_path)
    
    # Verifica se existe
    if not dataset_path.exists():
        print(f"❌ Path não encontrado: {dataset_path}")
        return
    
    # Coleta todas as imagens
    print("📂 Coletando imagens...")
    image_paths = []
    
    for class_folder in sorted(dataset_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            images = list(class_folder.glob(f'*{ext}'))
            image_paths.extend(images)
    
    print(f"   Total: {len(image_paths)} imagens")
    
    # Verifica modos de cor
    print("\n🔍 Verificando modos de cor...")
    modes = {}
    
    for img_path in tqdm(image_paths[:100], desc="Amostra"):
        mode = check_image_mode(img_path)
        modes[mode] = modes.get(mode, 0) + 1
    
    print(f"\n   Modos encontrados (amostra de 100):")
    for mode, count in modes.items():
        status = "❌" if mode != 'RGB' else "✅"
        print(f"      {status} {mode}: {count}/100")
    
    if 'L' in modes or 'P' in modes:
        print(f"\n   ⚠️  ATENÇÃO: Imagens em escala de cinza/palette detectadas!")
        print(f"   Todas serão convertidas para RGB.")
    
    # Carrega modelo
    print(f"\n⏳ Carregando CLIP ({MODEL_NAME})...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Modelo carregado!")
    
    # Extrai embeddings
    print(f"\n🔄 Extraindo embeddings...")
    
    embeddings = []
    valid_paths = []
    failed = []
    
    for img_path in tqdm(image_paths, desc="Processando"):
        try:
            # Converte para RGB (força cor)
            img = convert_to_rgb(img_path)
            
            # Verifica se realmente é RGB agora
            if img.mode != 'RGB':
                raise ValueError(f"Falha ao converter para RGB: {img.mode}")
            
            # Processa
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                embedding = model.get_image_features(**inputs)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            embeddings.append(embedding.cpu())
            valid_paths.append(str(img_path))
            
        except Exception as e:
            failed.append({"path": str(img_path), "error": str(e)})
    
    if failed:
        print(f"\n⚠️  Falhas: {len(failed)}/{len(image_paths)}")
        if len(failed) <= 10:
            for f in failed:
                print(f"   - {f['path']}: {f['error']}")
    
    # Concatena embeddings
    embeddings_tensor = torch.cat(embeddings, dim=0)
    
    print(f"\n📊 Embeddings extraídos:")
    print(f"   Total: {len(embeddings_tensor)}")
    print(f"   Shape: {embeddings_tensor.shape}")
    print(f"   Norm média: {embeddings_tensor.norm(dim=-1).mean():.6f}")
    
    # Verifica diversidade
    sample_size = min(100, len(embeddings_tensor))
    sample = embeddings_tensor[:sample_size]
    sim_matrix = sample @ sample.T
    mask = ~torch.eye(sample_size, dtype=bool)
    avg_sim = sim_matrix[mask].mean().item()
    
    print(f"   Similaridade média: {avg_sim:.4f}")
    
    if avg_sim > 0.6:
        print(f"      ⚠️  Ainda muito similar! Pode haver outro problema.")
    elif avg_sim > 0.4:
        print(f"      ⚠️  Similaridade um pouco alta, mas aceitável.")
    else:
        print(f"      ✅ Boa diversidade de embeddings!")
    
    # Salva
    output_path = Path(OUTPUT_DIR) / f"{dataset_name}.pt"
    
    # Backup se já existe
    if output_path.exists():
        backup_path = output_path.with_suffix('.pt.backup')
        os.rename(output_path, backup_path)
        print(f"\n📦 Backup criado: {backup_path}")
    
    # Salva novos embeddings
    data = {
        'image_embeddings': embeddings_tensor,
        'image_paths': valid_paths
    }
    
    torch.save(data, output_path)
    print(f"✅ Embeddings salvos em: {output_path}")
    
    # Teste rápido
    print(f"\n🧪 TESTE RÁPIDO:")
    
    # Carrega descriptors
    desc_path = Path("descriptors") / f"{dataset_name}_descriptors.json"
    
    if desc_path.exists():
        with open(desc_path, 'r', encoding='utf-8') as f:
            descriptors = json.load(f)
        
        # Extrai classes
        classes_from_paths = [Path(p).parts[-2] for p in valid_paths]
        unique_classes = sorted(set(classes_from_paths))
        
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        labels = torch.tensor([class_to_idx[cls] for cls in classes_from_paths])
        
        # Gera text embeddings
        texts = [descriptors.get(cls, f"a photo of a {cls}") for cls in unique_classes]
        text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            text_embeds = model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Teste nas primeiras 500 imagens
        test_size = min(500, len(embeddings_tensor))
        test_embeds = embeddings_tensor[:test_size]
        test_labels = labels[:test_size]
        
        sims = test_embeds @ text_embeds.cpu().T
        preds = sims.argmax(dim=-1)
        
        acc = (preds == test_labels).float().mean().item()
        
        print(f"   Testando nas primeiras {test_size} imagens:")
        print(f"   Acurácia: {acc:.4f} ({acc*100:.2f}%)")
        
        if acc > 0.30:
            print(f"   ✅ SUCESSO! Acurácia muito melhor que antes!")
        elif acc > 0.15:
            print(f"   ⚠️  Melhorou, mas ainda abaixo do esperado")
        else:
            print(f"   ❌ Problema persiste, investigar mais")
    
    return len(embeddings_tensor)


def main():
    print(f"\n{'#'*70}")
    print(f"# RE-EXTRAÇÃO DE EMBEDDINGS COM IMAGENS COLORIDAS")
    print(f"{'#'*70}\n")
    
    # Dataset problemático
    dataset_name = "CUB_200_2011"
    dataset_path = "datasets/CUB_200_2011"
    
    print(f"🎯 Alvo: {dataset_name}")
    print(f"📁 Path: {dataset_path}")
    
    # Verifica se existe
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset não encontrado!")
        print(f"\nVerifique se o path está correto.")
        print(f"O CUB-200-2011 deve ter esta estrutura:")
        print(f"   datasets/CUB_200_2011/")
        print(f"      001.Black_footed_Albatross/")
        print(f"      002.Laysan_Albatross/")
        print(f"      ...")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extrai
    num_extracted = extract_embeddings_rgb(dataset_path, dataset_name)
    
    print(f"\n{'#'*70}")
    print(f"# CONCLUSÃO")
    print(f"{'#'*70}\n")
    
    if num_extracted:
        print(f"""
✅ Embeddings re-extraídos com sucesso!

Total: {num_extracted} imagens

PRÓXIMOS PASSOS:

1. Execute a avaliação zero-shot novamente:
   python evaluate_zeroshot.py

2. A acurácia deve melhorar SIGNIFICATIVAMENTE:
   Esperado: ~50-55% (era 1.27%)

3. Se ainda estiver baixo, pode ser necessário:
   - Verificar se as imagens originais são coloridas
   - Usar um modelo CLIP maior (ViT-L/14)
   - Usar ensemble de templates
        """)
    else:
        print("❌ Falha na extração!")


if __name__ == "__main__":
    main()