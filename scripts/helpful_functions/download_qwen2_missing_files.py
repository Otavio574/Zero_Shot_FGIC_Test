"""
Baixa arquivos do processor/tokenizer diretamente na pasta local.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil


def download_processor_to_local():
    print("="*70)
    print("📥 BAIXANDO PROCESSOR PARA PASTA LOCAL")
    print("="*70 + "\n")
    
    repo_id = "Qwen/Qwen2-VL-7B-Instruct"
    local_dir = Path("models_local") / "Qwen2-VL-7B-Instruct"
    
    if not local_dir.exists():
        print("❌ Pasta local não existe!")
        print("💡 Execute primeiro: python copy_to_local.py")
        return False
    
    # Arquivos do processor/tokenizer
    processor_files = [
        "preprocessor_config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    
    print(f"📦 Baixando para: {local_dir}\n")
    
    downloaded = 0
    
    for filename in processor_files:
        dest_file = local_dir / filename
        
        # Se já existe, pula
        if dest_file.exists():
            print(f"⏭️  {filename} (já existe)")
            continue
        
        print(f"⬇️  {filename}...", end=" ", flush=True)
        
        try:
            # Baixa para cache temporário
            temp_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir_use_symlinks=False
            )
            
            # Copia para pasta local
            shutil.copy2(temp_path, dest_file)
            print("✓")
            downloaded += 1
            
        except Exception as e:
            print(f"⚠️ ({e})")
    
    print(f"\n✅ Baixados: {downloaded} arquivos")
    
    # Verifica se está completo
    print("\n🔍 Verificando modelo completo...\n")
    
    all_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    missing = []
    for filename in all_files:
        if not (local_dir / filename).exists():
            missing.append(filename)
    
    if missing:
        print("⚠️ Arquivos faltando:")
        for m in missing:
            print(f"   - {m}")
        return False
    else:
        print("✅ Todos os arquivos essenciais presentes!")
        print("\n" + "="*70)
        print("🎉 MODELO LOCAL COMPLETO!")
        print("="*70)
        print(f"\n📁 Localização: {local_dir.absolute()}")
        print("\n💡 Agora edite o generate_descriptors_local.py:")
        print('   MODEL_NAME = "models_local/Qwen2-VL-7B-Instruct"')
        print("="*70 + "\n")
        return True


if __name__ == "__main__":
    download_processor_to_local()