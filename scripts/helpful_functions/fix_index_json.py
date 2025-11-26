"""
Teste de carregamento com debug detalhado.
"""

import os
import sys
from pathlib import Path

# Adiciona logging detalhado
import logging
logging.basicConfig(level=logging.DEBUG)

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def test_load():
    print("="*70)
    print("🔍 TESTE DE CARREGAMENTO COM DEBUG")
    print("="*70 + "\n")
    
    model_dir = Path("models_local") / "Qwen2-VL-7B-Instruct"
    
    print(f"📁 Diretório: {model_dir.absolute()}\n")
    
    # Lista arquivos
    print("📋 Arquivos presentes:")
    for f in sorted(model_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   ✓ {f.name} ({size_mb:.1f} MB)")
    print()
    
    # Testa carregar config
    print("1️⃣ Carregando config...")
    try:
        config = AutoConfig.from_pretrained(
            str(model_dir.absolute()),
            trust_remote_code=True,
            local_files_only=True
        )
        print("   ✅ Config carregado\n")
    except Exception as e:
        print(f"   ❌ Erro: {e}\n")
        return
    
    # Tenta carregar modelo com safe_serialization desabilitado
    print("2️⃣ Tentando carregar modelo...")
    print("   (pode demorar alguns segundos)\n")
    
    # Muda para o diretório
    original_cwd = os.getcwd()
    os.chdir(model_dir)
    
    try:
        # Importa a classe específica
        from transformers import Qwen2VLForConditionalGeneration
        
        # Tenta carregar
        print("   Executando from_pretrained...\n")
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            ".",
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",  # CPU primeiro para testar
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("\n✅ MODELO CARREGADO COM SUCESSO!")
        print(f"   Tipo: {type(model)}")
        print(f"   Device: {model.device}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERRO DE ARQUIVO NÃO ENCONTRADO:")
        print(f"   {e}\n")
        
        # Extrai o path problemático
        error_msg = str(e)
        if "model-00001-of-00005.safetensors" in error_msg:
            print("🔍 DIAGNÓSTICO:")
            print("   O transformers está construindo paths incorretos internamente.")
            print("   Isso é um bug conhecido do transformers + Windows.")
            print()
            print("💡 POSSÍVEIS SOLUÇÕES:")
            print("   1. Atualizar transformers:")
            print("      pip install --upgrade transformers>=4.45.0")
            print()
            print("   2. Usar WSL (Windows Subsystem for Linux)")
            print()
            print("   3. Usar outro modelo (ex: LLaVA, BLIP-2)")
            
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    test_load()