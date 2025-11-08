import json
import os
from pathlib import Path

# ============================
# CONFIGURAÇÕES
# ============================
DESCRIPTOR_PATH = Path("descriptors/CUB_200_2011_descriptors.json")
OUTPUT_PATH = Path("descriptors/CUB_200_2011_sanitized.json") # Novo arquivo!

def sanitize_and_regenerate_descriptors():
    """
    Carrega o JSON, garante que as chaves estão limpas (sem espaços invisíveis)
    e regenera os descriptors usando um template básico.
    """
    if not DESCRIPTOR_PATH.exists():
        print(f"❌ Erro: Arquivo de descriptors não encontrado em {DESCRIPTOR_PATH}")
        return

    try:
        # Tenta ler com codificação padrão (utf-8)
        with open(DESCRIPTOR_PATH, "r", encoding="utf-8") as f:
            descriptors = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar o JSON (tentando ler com 'latin-1'): {e}")
        try:
            # Tenta ler com outra codificação (comum em falhas)
            with open(DESCRIPTOR_PATH, "r", encoding="latin-1") as f:
                descriptors = json.load(f)
        except Exception as e:
            print(f"❌ Falha crítica ao carregar o JSON: {e}")
            return

    new_descriptors = {}
    print(f"📂 JSON original carregado com {len(descriptors)} chaves.")

    for class_key, description in descriptors.items():
        # 1. Sanear a Chave (nome da classe)
        # Remove espaços em branco antes ou depois da chave
        sane_key = class_key.strip()
        
        # 2. Normalizar o Descriptor (o texto)
        # Se você quer usar um descriptor mais forte do que o seu JSON atual:
        
        # Limpa o prefixo numérico para criar um descriptor mais legível (ex: 'Black-footed Albatross')
        parts = sane_key.split('.', 1)
        if len(parts) == 2 and parts[0].isdigit():
            bird_name = parts[1]
        else:
            bird_name = sane_key
            
        # Converte nome CUB para formato legível por humanos (ex: 'Black-footed Albatross' -> 'black footed albatross')
        readable_name = bird_name.lower().replace('_', ' ').replace('-', ' ')
        
        # Cria um descriptor Zero-Shot forte (CLIP Prompting)
        new_description = f"a photo of the bird {readable_name}, spotted in its natural habitat."
        
        # Mapeia a chave sanada para o novo descriptor
        new_descriptors[sane_key] = new_description

    # Salva o novo dicionário limpo
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_descriptors, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saneamento concluído! {len(new_descriptors)} descriptors gerados.")
    print(f"💾 Novo arquivo salvo em: {OUTPUT_PATH}")
    print("\n⚠️ Próxima Ação: Troque o nome do arquivo no seu script de avaliação!")

if __name__ == "__main__":
    sanitize_and_regenerate_descriptors()