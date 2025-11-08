"""
Gera 'waffle descriptors' (descrições com ruído) para todos os datasets
com arquivos *_descriptors.json na pasta `descriptors/`.

Saída: descriptors/{dataset_name}_waffle.json
"""

import json
import random
import os
from pathlib import Path

# ==========================
# CONFIGURAÇÕES
# ==========================
DESCRIPTORS_DIR = Path("descriptors")

# Palavras aleatórias que serão injetadas nas descrições
RANDOM_WORDS = [
    "Humvee", "banana", "cucumber", "airplane", "candle", "motorcycle",
    "keyboard", "pizza", "toaster", "telescope", "lighthouse",
    "cactus", "guitar", "shark", "pencil", "umbrella", "volcano",
    "spaceship", "mountain", "train", "cloud", "carrot", "dragonfly",
    "window", "river", "comet", "camera", "skyscraper", "shoe",
]

# Configurações do ruído
N_RANDOM_WORDS = 3      # número de palavras randômicas por variação
N_VARIATIONS = 3         # número de variações por classe
SEED = 42
random.seed(SEED)

# ==========================
# FUNÇÕES
# ==========================

def generate_waffle_for_dataset(source_path: Path):
    """Gera e salva descritores waffle a partir de um arquivo base"""
    dataset_name = source_path.stem.replace("_descriptors", "")
    output_path = source_path.parent / f"{dataset_name}_waffle.json"

    print(f"\n🧇 Gerando waffle descriptors para {dataset_name}...")

    with open(source_path, "r", encoding="utf-8") as f:
        base_desc = json.load(f)

    waffle_desc = {}

    for cls, texts in base_desc.items():
        waffle_desc[cls] = []
        if isinstance(texts, list):
            base_text = random.choice(texts)
        else:
            base_text = str(texts)

        for _ in range(N_VARIATIONS):
            # Gera ruído textual misturando palavras aleatórias
            noise = ", ".join(random.sample(RANDOM_WORDS, N_RANDOM_WORDS))
            # Pode injetar o ruído de maneira mais "natural" na frase
            phrase = f"{base_text}. Random context: {noise}."
            waffle_desc[cls].append(phrase)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(waffle_desc, f, indent=4, ensure_ascii=False)

    print(f"✅ Waffle descriptors salvos em: {output_path}")
    print(f"💬 Exemplo para '{list(waffle_desc.keys())[0]}':")
    for ex in waffle_desc[list(waffle_desc.keys())[0]][:3]:
        print(f"   - {ex}")


# ==========================
# EXECUÇÃO PRINCIPAL
# ==========================

def main():
    print("🚀 Iniciando geração de waffle descriptors...\n")

    # Agora o script pega TODOS os arquivos *_descriptors.json
    descriptor_files = sorted(DESCRIPTORS_DIR.glob("*_descriptors.json"))

    if not descriptor_files:
        print("❌ Nenhum arquivo *_descriptors.json encontrado em 'descriptors/'.")
        return

    print(f"📂 {len(descriptor_files)} arquivos detectados:")
    for f in descriptor_files:
        print(f"   - {f.name}")

    for source_path in descriptor_files:
        try:
            generate_waffle_for_dataset(source_path)
        except Exception as e:
            print(f"❌ Erro ao processar {source_path.name}: {e}")

    print("\n🎯 Geração de waffle descriptors finalizada com sucesso!")
    print(f"📁 Todos os arquivos foram salvos em: {DESCRIPTORS_DIR}")


if __name__ == "__main__":
    main()
