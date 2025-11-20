"""
Diagnóstico profundo para encontrar o problema de alinhamento
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoProcessor, AutoModel

EMBEDDINGS_DIR = "embeddings"
DESCRIPTORS_DIR = "descriptors"
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def analyze_alignment(dataset_name):
    """Analisa o alinhamento entre embeddings, paths e labels"""
    
    print(f"\n{'='*70}")
    print(f"🔬 ANÁLISE PROFUNDA: {dataset_name}")
    print(f"{'='*70}\n")
    
    # 1. Carrega embeddings
    emb_path = os.path.join(EMBEDDINGS_DIR, f"{dataset_name}.pt")
    data = torch.load(emb_path, map_location='cpu')
    
    image_embeds = data['image_embeddings']
    paths = data['image_paths']
    
    print(f"📊 DADOS CARREGADOS:")
    print(f"   Embeddings: {image_embeds.shape}")
    print(f"   Paths: {len(paths)}")
    
    # 2. Extrai classes dos paths
    classes_from_paths = []
    for p in paths:
        parts = Path(p).parts
        class_name = parts[-2]  # Pasta pai
        classes_from_paths.append(class_name)
    
    unique_classes = sorted(set(classes_from_paths))
    print(f"   Classes únicas: {len(unique_classes)}")
    
    # 3. Verifica ordenação
    print(f"\n🔍 ANÁLISE DE ORDENAÇÃO:")
    print(f"   Primeiras 20 imagens e suas classes:")
    
    for i in range(min(20, len(paths))):
        print(f"      [{i:3d}] {classes_from_paths[i]:40s} | {Path(paths[i]).name}")
    
    # Verifica se está agrupado
    transitions = 0
    prev_class = classes_from_paths[0]
    transition_points = []
    
    for i, cls in enumerate(classes_from_paths[1:], 1):
        if cls != prev_class:
            transitions += 1
            transition_points.append((i, prev_class, cls))
            prev_class = cls
    
    print(f"\n   Total de transições: {transitions}")
    print(f"   Esperado (se agrupado): {len(unique_classes) - 1}")
    
    if transitions == len(unique_classes) - 1:
        print(f"   ✅ Imagens AGRUPADAS por classe (ordem correta)")
    else:
        print(f"   ⚠️  Imagens NÃO estão perfeitamente agrupadas!")
        print(f"\n   Primeiras 5 transições:")
        for i, (idx, prev, curr) in enumerate(transition_points[:5]):
            print(f"      [{idx:4d}] {prev} → {curr}")
    
    # 4. Verifica distribuição
    print(f"\n📊 DISTRIBUIÇÃO DE CLASSES:")
    class_counts = Counter(classes_from_paths)
    
    print(f"   Primeiras 10 classes:")
    for cls, count in class_counts.most_common(10):
        print(f"      {cls:40s}: {count:3d} imagens")
    
    # Verifica se é balanceado
    counts_values = list(class_counts.values())
    is_balanced = (max(counts_values) - min(counts_values)) <= 5
    
    print(f"\n   Min: {min(counts_values)}, Max: {max(counts_values)}, Média: {np.mean(counts_values):.1f}")
    
    if is_balanced:
        print(f"   ✅ Dataset BALANCEADO")
    else:
        print(f"   ⚠️  Dataset DESBALANCEADO")
    
    # 5. Carrega descriptors
    desc_path = os.path.join(DESCRIPTORS_DIR, f"{dataset_name}_descriptors.json")
    with open(desc_path, 'r', encoding='utf-8') as f:
        descriptors = json.load(f)
    
    print(f"\n📝 DESCRIPTORS:")
    print(f"   Total: {len(descriptors)}")
    
    # Verifica matching
    print(f"\n🔗 MATCHING (classes nos paths vs descriptors):")
    
    matched = 0
    missing = []
    
    for cls in unique_classes[:10]:  # Testa primeiras 10
        if cls in descriptors:
            matched += 1
            desc = descriptors[cls]
            print(f"   ✅ {cls:40s} → {desc[:50]}...")
        else:
            missing.append(cls)
            print(f"   ❌ {cls:40s} → NÃO ENCONTRADO!")
    
    print(f"\n   Match rate (primeiras 10): {matched}/10")
    
    # 6. TESTE ZERO-SHOT SIMPLIFICADO
    print(f"\n{'='*70}")
    print(f"🧪 TESTE ZERO-SHOT COM AS PRIMEIRAS 100 IMAGENS")
    print(f"{'='*70}\n")
    
    # Carrega modelo
    print("⏳ Carregando CLIP...")
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Pega primeiras 100 imagens
    num_samples = min(100, len(paths))
    sample_embeds = image_embeds[:num_samples]
    sample_paths = paths[:num_samples]
    sample_classes = [classes_from_paths[i] for i in range(num_samples)]
    
    # Cria mapeamento classe -> índice
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    sample_labels = np.array([class_to_idx[cls] for cls in sample_classes])
    
    print(f"   Amostras: {num_samples}")
    print(f"   Classes únicas nesta amostra: {len(set(sample_classes))}")
    print(f"   Distribuição: {Counter(sample_classes).most_common(5)}")
    
    # Gera text embeddings
    class_texts = []
    for cls in unique_classes:
        if cls in descriptors:
            class_texts.append(descriptors[cls])
        else:
            # Fallback
            clean = cls.replace('_', ' ').replace('-', ' ').lower()
            if '.' in clean:
                clean = clean.split('.', 1)[1]
            class_texts.append(f"a photo of a {clean}, a type of bird.")
    
    text_inputs = processor(
        text=class_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Normaliza image embeds
    sample_embeds = sample_embeds / sample_embeds.norm(dim=-1, keepdim=True)
    
    # Calcula similaridades
    sims = sample_embeds @ text_embeds.cpu().T
    preds = sims.argmax(dim=-1).numpy()
    
    # Acurácia
    acc = (preds == sample_labels).mean()
    
    print(f"\n   📊 RESULTADOS:")
    print(f"      Acurácia: {acc:.4f} ({acc*100:.2f}%)")
    print(f"      Similaridade média: {sims.mean():.4f}")
    print(f"      Similaridade máxima: {sims.max():.4f}")
    
    # Mostra exemplos detalhados
    print(f"\n   🔍 PRIMEIROS 10 EXEMPLOS:")
    print(f"      {'Idx':<5} {'True Class':<40} {'Pred Class':<40} {'Correct':<8} {'Sim':<6}")
    print(f"      {'-'*110}")
    
    for i in range(min(10, num_samples)):
        true_cls = unique_classes[sample_labels[i]]
        pred_cls = unique_classes[preds[i]]
        correct = "✅" if preds[i] == sample_labels[i] else "❌"
        sim = sims[i, preds[i]].item()
        
        print(f"      [{i:3d}] {true_cls:<40} {pred_cls:<40} {correct:<8} {sim:.3f}")
    
    # Analisa erros mais comuns
    print(f"\n   📉 ANÁLISE DE ERROS:")
    
    errors = []
    for i in range(num_samples):
        if preds[i] != sample_labels[i]:
            true_cls = unique_classes[sample_labels[i]]
            pred_cls = unique_classes[preds[i]]
            errors.append((true_cls, pred_cls))
    
    error_counts = Counter(errors)
    print(f"      Total de erros: {len(errors)}/{num_samples}")
    print(f"\n      Top 5 confusões mais comuns:")
    
    for (true_cls, pred_cls), count in error_counts.most_common(5):
        print(f"         {true_cls:<35} → {pred_cls:<35} ({count}x)")
    
    # 7. TESTE COM TODAS AS IMAGENS
    print(f"\n{'='*70}")
    print(f"🧪 TESTE ZERO-SHOT COM TODAS AS IMAGENS")
    print(f"{'='*70}\n")
    
    all_labels = np.array([class_to_idx[cls] for cls in classes_from_paths])
    all_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    
    all_sims = all_embeds @ text_embeds.cpu().T
    all_preds = all_sims.argmax(dim=-1).numpy()
    
    all_acc = (all_preds == all_labels).mean()
    
    print(f"   Total de imagens: {len(all_labels)}")
    print(f"   Acurácia: {all_acc:.4f} ({all_acc*100:.2f}%)")
    print(f"   Similaridade média: {all_sims.mean():.4f}")
    
    # Distribuição de predições
    pred_counts = Counter(all_preds)
    print(f"\n   📊 Distribuição de predições:")
    print(f"      Classes preditas: {len(pred_counts)}/{len(unique_classes)}")
    print(f"      Top 10 classes mais preditas:")
    
    for pred_idx, count in pred_counts.most_common(10):
        pred_cls = unique_classes[pred_idx]
        percentage = count / len(all_labels) * 100
        print(f"         {pred_cls:<40}: {count:4d} ({percentage:.1f}%)")
    
    return all_acc


def main():
    print(f"\n{'#'*70}")
    print(f"# DIAGNÓSTICO PROFUNDO - ALINHAMENTO DE DADOS")
    print(f"{'#'*70}\n")
    
    # Foca no CUB que está com problema
    dataset = "CUB_200_2011"
    
    acc = analyze_alignment(dataset)
    
    print(f"\n{'#'*70}")
    print(f"# CONCLUSÃO")
    print(f"{'#'*70}\n")
    
    if acc < 0.10:
        print(f"""
❌ PROBLEMA GRAVE DETECTADO!

Acurácia de {acc*100:.2f}% é MUITO baixa para zero-shot CLIP no CUB.
Esperado: ~50-55%

POSSÍVEIS CAUSAS:

1. 🔴 EMBEDDINGS CORROMPIDOS
   - Os embeddings de imagem podem estar corrompidos
   - Solução: Re-extrair embeddings do zero

2. 🔴 MODELO ERRADO
   - Você pode ter usado um modelo diferente para extrair
   - Solução: Verificar qual modelo foi usado na extração

3. 🔴 NORMALIZAÇÃO ERRADA
   - Embeddings podem não estar normalizados corretamente
   - Solução: Garantir normalização L2

4. 🔴 ORDEM COMPLETAMENTE ERRADA
   - Apesar das verificações, pode haver shuffle oculto
   - Solução: Re-extrair garantindo shuffle=False

PRÓXIMA AÇÃO:
Execute o script de extração de embeddings com este diagnóstico
e compare os resultados.
        """)
    elif acc < 0.30:
        print(f"""
⚠️  ACURÁCIA ABAIXO DO ESPERADO

Acurácia de {acc*100:.2f}% está abaixo dos ~50% esperados.

Possíveis melhorias:
- Usar templates ensemble (múltiplos templates por classe)
- Verificar qualidade das imagens
- Usar modelo CLIP maior (ViT-B/16 ou ViT-L/14)
        """)
    else:
        print(f"""
✅ ACURÁCIA DENTRO DO ESPERADO!

Acurácia de {acc*100:.2f}% está próxima dos resultados do paper.
        """)


if __name__ == "__main__":
    main()