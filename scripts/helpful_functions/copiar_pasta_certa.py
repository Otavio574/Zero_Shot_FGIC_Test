"""
Copia o modelo do cache corrompido para uma pasta local limpa.
Solução definitiva para o bug de symlinks do Windows.
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def copy_model_to_local():
    print("="*70)
    print("📦 COPIANDO MODELO PARA PASTA LOCAL (MODO CORREÇÃO)")
    print("="*70 + "\n")
    
    # Origem: cache do HuggingFace
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    source = cache_dir / "models--Qwen--Qwen2-VL-7B-Instruct" / "snapshots"
    
    if not source.exists():
        print("❌ Cache não encontrado!")
        return False
    
    # Pega o snapshot
    snapshots = list(source.iterdir())
    if not snapshots:
        print("❌ Nenhum snapshot encontrado!")
        return False
    
    snapshot_dir = snapshots[0]
    
    # Destino: pasta local
    dest = Path("models_local") / "Qwen2-VL-7B-Instruct"
    
    print(f"📁 Origem: {snapshot_dir}")
    print(f"📁 Destino: {dest}\n")
    
    # LIMPA destino para garantir cópia limpa
    if dest.exists():
        print("🧹 Limpando pasta de destino antiga...")
        shutil.rmtree(dest)
    
    # Cria pasta de destino
    dest.mkdir(parents=True, exist_ok=True)
    
    # Lista TODOS os itens (arquivos e pastas)
    print("🔍 Escaneando estrutura do cache...\n")
    
    files_to_copy = []
    
    for item in snapshot_dir.iterdir():
        if item.is_file():
            # Arquivo direto - copia normalmente
            files_to_copy.append((item, dest / item.name))
        elif item.is_dir():
            # Pasta - verifica se é bug de symlink
            # Bug: cria pasta "model-00001-of-00005.safetensors/" com arquivo dentro
            inner_items = list(item.iterdir())
            
            if inner_items:
                for inner in inner_items:
                    if inner.is_file():
                        # Este é o arquivo REAL que está dentro da pasta errada
                        # Copia usando o nome do arquivo (não da pasta)
                        files_to_copy.append((inner, dest / inner.name))
                        print(f"⚠️ Corrigindo: {item.name}/ → {inner.name}")
    
    print(f"\n📋 {len(files_to_copy)} arquivos para copiar\n")
    
    # Copia cada arquivo
    copied = 0
    
    for src, dst in tqdm(files_to_copy, desc="Copiando"):
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"\n⚠️ Erro ao copiar {src.name}: {e}")
    
    print(f"\n✅ Copiados: {copied} arquivos")
    
    # Verifica arquivos essenciais
    print("\n🔍 Verificando arquivos essenciais...\n")
    
    essential = [
        "config.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json",
    ]
    
    all_ok = True
    for filename in essential:
        file_path = dest / filename
        if file_path.exists() and file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {filename} - FALTANDO!")
            all_ok = False
    
    # Verifica se há PASTAS (não deveria ter)
    print("\n🔍 Verificando se há pastas (não deveria ter)...\n")
    
    dirs_found = [d for d in dest.iterdir() if d.is_dir()]
    
    if dirs_found:
        print("⚠️ PASTAS ENCONTRADAS (ESTRUTURA ERRADA):")
        for d in dirs_found:
            print(f"   - {d.name}/")
        all_ok = False
    else:
        print("✓ Nenhuma pasta encontrada (estrutura correta!)")
    
    if all_ok:
        print("\n" + "="*70)
        print("✅ MODELO COPIADO COM SUCESSO!")
        print("="*70)
        print(f"\n📁 Modelo local em: {dest.absolute()}")
        print("\n💡 Agora baixe os arquivos do processor:")
        print("   python download_processor.py")
        print("="*70 + "\n")
        return True
    else:
        print("\n❌ Alguns arquivos estão faltando ou estrutura incorreta!")
        return False


if __name__ == "__main__":
    copy_model_to_local()