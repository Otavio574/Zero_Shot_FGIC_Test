"""
Diagnóstico detalhado do cache Qwen2-VL.
"""

from pathlib import Path
import os


def diagnose_cache():
    print("="*70)
    print("🔍 DIAGNÓSTICO DO CACHE")
    print("="*70 + "\n")
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / "models--Qwen--Qwen2-VL-7B-Instruct"
    
    if not model_cache.exists():
        print("❌ Cache não encontrado!")
        return
    
    snapshots_dir = model_cache / "snapshots"
    snapshot = list(snapshots_dir.iterdir())[0]
    
    print(f"📁 Snapshot: {snapshot.name}\n")
    print("📋 Arquivos esperados:\n")
    
    expected_files = [
        "config.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    
    missing = []
    found = []
    wrong_structure = []
    
    for expected in expected_files:
        file_path = snapshot / expected
        
        if file_path.exists() and file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            found.append(f"✓ {expected} ({size_mb:.1f} MB)")
        elif file_path.exists() and file_path.is_dir():
            wrong_structure.append(f"⚠️ {expected} (É uma PASTA, deveria ser arquivo!)")
        else:
            missing.append(f"✗ {expected}")
    
    print("\n🟢 ENCONTRADOS:")
    for f in found:
        print(f"   {f}")
    
    if missing:
        print("\n🔴 FALTANDO:")
        for m in missing:
            print(f"   {m}")
    
    if wrong_structure:
        print("\n⚠️ ESTRUTURA ERRADA:")
        for w in wrong_structure:
            print(f"   {w}")
    
    # Lista TUDO que está na pasta
    print("\n📦 TUDO que está no snapshot:\n")
    
    all_items = sorted(snapshot.iterdir(), key=lambda x: x.name)
    
    for item in all_items:
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"   📄 {item.name} ({size_mb:.1f} MB)")
        elif item.is_dir():
            print(f"   📁 {item.name}/ (PASTA)")
            # Lista conteúdo da pasta
            for sub in item.iterdir():
                if sub.is_file():
                    size_mb = sub.stat().st_size / (1024 * 1024)
                    print(f"      └─ 📄 {sub.name} ({size_mb:.1f} MB)")
                else:
                    print(f"      └─ 📁 {sub.name}/")
    
    # Verifica o erro específico
    print("\n🔍 VERIFICAÇÃO DO ERRO ESPECÍFICO:\n")
    
    problematic_path = snapshot / "model-00001-of-00005.safetensors" / "model-00005-of-00005.safetensors"
    
    if problematic_path.exists():
        print(f"❌ PROBLEMA CONFIRMADO!")
        print(f"   Arquivo está em: {problematic_path}")
        print(f"   Deveria estar em: {snapshot / 'model-00005-of-00005.safetensors'}")
        
        parent = problematic_path.parent
        print(f"\n   📁 Conteúdo de {parent.name}:")
        for item in parent.iterdir():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"      - {item.name} ({size_mb:.1f} MB)")
    else:
        print("✓ Caminho problemático não existe")
    
    print("\n" + "="*70)
    print("💡 DIAGNÓSTICO COMPLETO")
    print("="*70 + "\n")


if __name__ == "__main__":
    diagnose_cache()