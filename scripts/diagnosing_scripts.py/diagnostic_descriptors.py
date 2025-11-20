"""
Script de Diagnóstico para Descritores DCLIP e Comparative-CLIP
Ajuda a identificar por que os resultados estão muito baixos.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DCLIP_DIR = Path("descriptors_dclip")
COMPARATIVE_DIR = Path("descriptors_comparative")

DATASET = "CUB200"  # Ajuste conforme necessário


# ============================================================
# ANÁLISE DE DESCRITORES
# ============================================================

def load_and_analyze_descriptors(descriptor_path: Path, method_name: str):
    """
    Carrega e analisa descritores para identificar problemas.
    """
    
    if not descriptor_path.exists():
        print(f"❌ Arquivo não encontrado: {descriptor_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"📊 Analisando {method_name}: {descriptor_path.name}")
    print(f"{'='*70}\n")
    
    with open(descriptor_path, 'r', encoding='utf-8') as f:
        descriptors = json.load(f)
    
    # Estatísticas básicas
    num_classes = len(descriptors)
    desc_counts = [len(descs) for descs in descriptors.values()]
    
    print(f"📌 Estatísticas Gerais:")
    print(f"   Total de classes: {num_classes}")
    print(f"   Total de descritores: {sum(desc_counts)}")
    print(f"   Média por classe: {np.mean(desc_counts):.1f}")
    print(f"   Mediana por classe: {np.median(desc_counts):.1f}")
    print(f"   Min por classe: {min(desc_counts)}")
    print(f"   Max por classe: {max(desc_counts)}")
    
    # Distribuição de tamanhos
    print(f"\n📊 Distribuição de descritores por classe:")
    hist = Counter(desc_counts)
    for count in sorted(hist.keys())[:10]:  # Top 10
        print(f"   {count} descritores: {hist[count]} classes")
    
    # Análise de qualidade
    print(f"\n🔍 Análise de Qualidade:")
    
    # Descritores vazios
    empty_classes = [cls for cls, descs in descriptors.items() if len(descs) == 0]
    if empty_classes:
        print(f"   ⚠️  Classes SEM descritores: {len(empty_classes)}")
        print(f"      Exemplos: {empty_classes[:5]}")
    else:
        print(f"   ✅ Todas as classes têm descritores")
    
    # Descritores muito curtos (< 10 caracteres)
    short_descs = []
    for cls, descs in descriptors.items():
        short = [d for d in descs if len(d) < 10]
        if short:
            short_descs.append((cls, short))
    
    if short_descs:
        print(f"   ⚠️  Classes com descritores muito curtos: {len(short_descs)}")
        print(f"      Exemplo: {short_descs[0][0]} → {short_descs[0][1][:3]}")
    else:
        print(f"   ✅ Não há descritores muito curtos")
    
    # Descritores repetidos/genéricos
    all_descs_flat = []
    for descs in descriptors.values():
        all_descs_flat.extend(descs)
    
    desc_freq = Counter(all_descs_flat)
    most_common = desc_freq.most_common(10)
    
    print(f"\n📝 Top 10 Descritores Mais Comuns:")
    for desc, count in most_common:
        percentage = (count / num_classes) * 100
        print(f"   '{desc[:60]}' → {count} classes ({percentage:.1f}%)")
    
    # 🚨 ALERTA: Descritores genéricos/inúteis
    generic_patterns = [
        "unique visual feature",
        "distinct features",
        "has distinct",
        "characteristic appearance",
        "typical appearance"
    ]
    
    problematic_classes = []
    for cls, descs in descriptors.items():
        for desc in descs:
            if any(pattern in desc.lower() for pattern in generic_patterns):
                problematic_classes.append(cls)
                break
    
    if problematic_classes:
        print(f"\n   🚨 PROBLEMA DETECTADO!")
        print(f"   Classes com descritores GENÉRICOS/INÚTEIS: {len(problematic_classes)}")
        print(f"   Isso explica a performance baixa!")
        print(f"   Exemplos: {problematic_classes[:5]}")
    
    # Mostra exemplos de descritores
    print(f"\n📋 Exemplos de Descritores (3 primeiras classes):")
    for idx, (cls, descs) in enumerate(list(descriptors.items())[:3]):
        print(f"\n   {idx+1}. {cls}:")
        for desc in descs[:5]:  # Primeiros 5 descritores
            print(f"      - {desc}")
        if len(descs) > 5:
            print(f"      ... (+{len(descs)-5} mais)")
    
    return {
        'num_classes': num_classes,
        'avg_descriptors': np.mean(desc_counts),
        'empty_classes': len(empty_classes),
        'short_descriptors': len(short_descs),
        'problematic_classes': len(problematic_classes),
        'descriptors': descriptors
    }


# ============================================================
# COMPARAÇÃO ENTRE MÉTODOS
# ============================================================

def compare_methods(dclip_stats, comparative_stats):
    """
    Compara estatísticas entre DCLIP e Comparative-CLIP.
    """
    
    if not dclip_stats or not comparative_stats:
        return
    
    print(f"\n{'='*70}")
    print(f"⚖️  COMPARAÇÃO: DCLIP vs Comparative-CLIP")
    print(f"{'='*70}\n")
    
    print(f"{'Métrica':<30} {'DCLIP':<15} {'Comparative':<15}")
    print(f"{'-'*60}")
    print(f"{'Média desc/classe':<30} {dclip_stats['avg_descriptors']:<15.1f} {comparative_stats['avg_descriptors']:<15.1f}")
    print(f"{'Classes vazias':<30} {dclip_stats['empty_classes']:<15} {comparative_stats['empty_classes']:<15}")
    print(f"{'Descritores curtos':<30} {dclip_stats['short_descriptors']:<15} {comparative_stats['short_descriptors']:<15}")
    print(f"{'Classes problemáticas':<30} {dclip_stats['problematic_classes']:<15} {comparative_stats['problematic_classes']:<15}")
    
    # Diagnóstico
    print(f"\n💡 Diagnóstico:")
    
    if dclip_stats['avg_descriptors'] < 5:
        print(f"   ⚠️  DCLIP: Poucos descritores! Esperado ~15-20, encontrado {dclip_stats['avg_descriptors']:.1f}")
    
    if comparative_stats['avg_descriptors'] < 30:
        print(f"   ⚠️  Comparative: Poucos descritores! Esperado ~80, encontrado {comparative_stats['avg_descriptors']:.1f}")
    
    if comparative_stats['problematic_classes'] > comparative_stats['num_classes'] * 0.5:
        print(f"   🚨 Comparative: MAIS DE 50% DAS CLASSES TÊM DESCRITORES GENÉRICOS!")
        print(f"      Isso explica por que está igual ao baseline.")
        print(f"      Solução: Reprocessar com o gerador corrigido do Qwen2.")


# ============================================================
# RECOMENDAÇÕES
# ============================================================

def print_recommendations(dclip_stats, comparative_stats):
    """
    Fornece recomendações baseadas na análise.
    """
    
    print(f"\n{'='*70}")
    print(f"💡 RECOMENDAÇÕES")
    print(f"{'='*70}\n")
    
    if dclip_stats and dclip_stats['avg_descriptors'] < 10:
        print(f"📌 DCLIP:")
        print(f"   1. Seus descritores são muito poucos ou genéricos")
        print(f"   2. Considere usar descritores do repositório original do DCLIP")
        print(f"   3. Ou regere com GPT-3/GPT-4 usando o prompt correto")
    
    if comparative_stats and comparative_stats['problematic_classes'] > 50:
        print(f"\n📌 Comparative-CLIP:")
        print(f"   1. 🚨 PROBLEMA CRÍTICO: Descritores genéricos detectados!")
        print(f"   2. O gerador Qwen2 não está funcionando corretamente")
        print(f"   3. Use o código corrigido que enviei (com extract_json_from_text)")
        print(f"   4. Verifique se o Qwen está realmente gerando JSONs válidos")
        print(f"   5. Considere aumentar K_SIMILAR_CLASSES e NUM_COMPARISONS_PER_PAIR")
    
    print(f"\n📌 Geral:")
    print(f"   1. Teste primeiro com um dataset menor (ex: 10 classes)")
    print(f"   2. Verifique manualmente os descritores gerados")
    print(f"   3. Compare com descritores do paper original")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🔍 Diagnóstico de Descritores CLIP")
    print("="*70)
    
    # Analisa DCLIP
    dclip_path = DCLIP_DIR / f"{DATASET}_dclip.json"
    dclip_stats = load_and_analyze_descriptors(dclip_path, "DCLIP")
    
    # Analisa Comparative-CLIP
    comparative_path = COMPARATIVE_DIR / f"{DATASET}_comparative.json"
    comparative_stats = load_and_analyze_descriptors(comparative_path, "Comparative-CLIP")
    
    # Comparação
    if dclip_stats and comparative_stats:
        compare_methods(dclip_stats, comparative_stats)
    
    # Recomendações
    print_recommendations(dclip_stats, comparative_stats)


if __name__ == "__main__":
    main