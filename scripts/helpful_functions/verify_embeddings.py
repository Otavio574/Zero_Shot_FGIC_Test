"""
Verifica a sanidade dos embeddings extraídos
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import pathlib

EMBEDDINGS_DIR = "embeddings"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_embedding_sanity(dataset_name):
    """Verifica se os embeddings fazem sentido"""
    
    print(f"\n{'='*70}")
    print(f"🔍 VERIFICAÇÃO DE SANIDADE: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Carrega embeddings salvos
    emb_path = f"./embeddings/{dataset_name}.pt"
    data = torch.load(emb_path, map_location='cpu')
    
    saved_embeds = data['image_embeddings']
    paths = data['image_paths']
    
    print(f"📊 EMBEDDINGS SALVOS:")
    print(f"   Shape: {saved_embeds.shape}")
    print(f"   Dtype: {saved_embeds.dtype}")
    print(f"   Device: {saved_embeds.device}")
    print(f"   Norm média: {saved_embeds.norm(dim=-1).mean():.6f}")
    print(f"   Std: {saved_embeds.std():.6f}")
    print(f"   Min: {saved_embeds.min():.6f}")
    print(f"   Max: {saved_embeds.max():.6f}")
    
    # Verifica se está normalizado
    norms = saved_embeds.norm(dim=-1)
    is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=0.01)
    
    if is_normalized:
        print(f"   ✅ Embeddings NORMALIZADOS (L2 norm ≈ 1.0)")
    else:
        print(f"   ⚠️  Embeddings NÃO normalizados!")
        print(f"      Norm min: {norms.min():.6f}, max: {norms.max():.6f}")
    
    # Estatísticas adicionais
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"   Média dos valores: {saved_embeds.mean():.6f}")
    print(f"   Percentil 25: {torch.quantile(saved_embeds, 0.25):.6f}")
    print(f"   Percentil 50 (mediana): {torch.quantile(saved_embeds, 0.50):.6f}")
    print(f"   Percentil 75: {torch.quantile(saved_embeds, 0.75):.6f}")
    
    # Similaridade interna (embeddings devem ser diversos)
    print(f"\n🔗 DIVERSIDADE DOS EMBEDDINGS:")
    
    # Pega amostra aleatória
    sample_size = min(100, len(saved_embeds))
    indices = torch.randperm(len(saved_embeds))[:sample_size]
    sample = saved_embeds[indices]
    
    # Normaliza se necessário
    sample_norm = sample / sample.norm(dim=-1, keepdim=True)
    
    # Calcula similaridade média
    sim_matrix = sample_norm @ sample_norm.T
    
    # Remove diagonal (auto-similaridade = 1.0)
    mask = ~torch.eye(sample_size, dtype=bool)
    off_diagonal = sim_matrix[mask]
    
    print(f"   Similaridade média entre imagens: {off_diagonal.mean():.4f}")
    print(f"   Similaridade std: {off_diagonal.std():.4f}")
    print(f"   Similaridade min: {off_diagonal.min():.4f}")
    print(f"   Similaridade max: {off_diagonal.max():.4f}")
    
    if off_diagonal.mean() > 0.5:
        print(f"   ⚠️  ATENÇÃO: Embeddings muito similares entre si!")
        print(f"      Isso indica que as imagens não estão sendo bem diferenciadas.")
    else:
        print(f"   ✅ Embeddings têm boa diversidade")
    
    # Teste prático: re-extrai embedding de UMA imagem e compara
    print(f"\n{'='*70}")
    print(f"🧪 TESTE PRÁTICO: Re-extração de uma imagem")
    print(f"{'='*70}\n")
    
    # Pega primeira imagem
    test_image_path = paths[0]
    saved_embed = saved_embeds[0]
    
    print(f"   Imagem: {test_image_path}")
    print(f"   Classe: {pathlib.Path(test_image_path).parts[-2]}")
    
    # Carrega modelo CLIP
    print(f"\n⏳ Carregando CLIP ({MODEL_NAME})...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Re-extrai embedding
    try:
        image = Image.open(test_image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            fresh_embed = model.get_image_features(**inputs)
            fresh_embed = fresh_embed / fresh_embed.norm(dim=-1, keepdim=True)
        
        fresh_embed = fresh_embed.cpu()
        
        print(f"\n📊 COMPARAÇÃO:")
        print(f"   Embedding salvo:")
        print(f"      Shape: {saved_embed.shape}")
        print(f"      Norm: {saved_embed.norm():.6f}")
        print(f"      Primeiros 5 valores: {saved_embed[:5]}")
        
        print(f"\n   Embedding recém-extraído:")
        print(f"      Shape: {fresh_embed.shape}")
        print(f"      Norm: {fresh_embed.norm():.6f}")
        print(f"      Primeiros 5 valores: {fresh_embed[0, :5]}")
        
        # Calcula similaridade
        saved_norm = saved_embed / saved_embed.norm()
        fresh_norm = fresh_embed[0] / fresh_embed[0].norm()
        similarity = (saved_norm @ fresh_norm).item()
        
        print(f"\n   🎯 SIMILARIDADE: {similarity:.6f}")
        
        if similarity > 0.99:
            print(f"   ✅ PERFEITO! Embeddings são idênticos (diff < 1%)")
        elif similarity > 0.95:
            print(f"   ✅ MUITO BOM! Embeddings são quase idênticos (diff < 5%)")
        elif similarity > 0.90:
            print(f"   ⚠️  ATENÇÃO! Embeddings têm diferença significativa (5-10%)")
        elif similarity > 0.5:
            print(f"   ⚠️  PROBLEMA! Embeddings têm diferença grande (10-50%)")
        else:
            print(f"   🔴 CRÍTICO! Embeddings são COMPLETAMENTE DIFERENTES!")
            print(f"      Possíveis causas:")
            print(f"      - Modelo diferente usado na extração")
            print(f"      - Preprocessamento diferente")
            print(f"      - Embeddings corrompidos")
        
        # Testa zero-shot com essa imagem
        print(f"\n🧪 TESTE ZERO-SHOT COM ESSA IMAGEM:")
        
        # Carrega descriptors
        from pathlib import Path
        import json
        
        desc_path = f"./descriptors/{dataset_name}_templates.json"
        with open(desc_path, 'r', encoding='utf-8') as f:
            descriptors = json.load(f)
        
        # Pega classes
        classes = sorted(descriptors.keys())
        true_class = pathlib.Path(test_image_path).parts[-2]
        true_idx = classes.index(true_class)
        
        # Gera text embeddings
        texts = [descriptors[cls] for cls in classes]
        text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            text_embeds = model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Testa com embedding SALVO
        sims_saved = (saved_norm.unsqueeze(0) @ text_embeds.cpu().T).squeeze()
        pred_saved = sims_saved.argmax().item()
        
        # Testa com embedding FRESCO
        sims_fresh = (fresh_norm.unsqueeze(0) @ text_embeds.cpu().T).squeeze()
        pred_fresh = sims_fresh.argmax().item()
        
        print(f"\n   True class: {true_class} (índice {true_idx})")
        print(f"\n   Com embedding SALVO:")
        print(f"      Predito: {classes[pred_saved]} (índice {pred_saved})")
        print(f"      Similaridade: {sims_saved[pred_saved]:.4f}")
        print(f"      Correto: {'✅' if pred_saved == true_idx else '❌'}")
        
        print(f"\n   Com embedding FRESCO:")
        print(f"      Predito: {classes[pred_fresh]} (índice {pred_fresh})")
        print(f"      Similaridade: {sims_fresh[pred_fresh]:.4f}")
        print(f"      Correto: {'✅' if pred_fresh == true_idx else '❌'}")
        
        # Top 5 predições
        print(f"\n   Top 5 classes mais similares (embedding salvo):")
        top5_saved = sims_saved.argsort(descending=True)[:5]
        for rank, idx in enumerate(top5_saved, 1):
            marker = "✅" if idx == true_idx else "  "
            print(f"      {marker} {rank}. {classes[idx]:40s} (sim: {sims_saved[idx]:.4f})")
        
    except Exception as e:
        print(f"   ❌ Erro ao processar imagem: {e}")
        import traceback
        traceback.print_exc()


def main():
    print(f"\n{'#'*70}")
    print(f"# VERIFICAÇÃO DE SANIDADE DOS EMBEDDINGS")
    print(f"{'#'*70}\n")
    
    dataset = "CUB_200_2011"
    check_embedding_sanity(dataset)
    
    print(f"\n{'#'*70}")
    print(f"# PRÓXIMOS PASSOS")
    print(f"{'#'*70}\n")
    
    print("""
    Se o teste mostrou baixa similaridade (<90%) entre embeddings:
    
    1. 🔴 EMBEDDINGS CORROMPIDOS ou MODELO ERRADO
       → Re-extraia os embeddings com o script correto
       
    2. ⚠️  Se embeddings muito similares entre si (média >0.5):
       → As imagens não estão sendo bem diferenciadas
       → Verifique o preprocessamento
       
    3. ✅ Se similaridade >99% e zero-shot funciona:
       → O problema está em outro lugar (labels, descriptors)
    """)


if __name__ == "__main__":
    main()