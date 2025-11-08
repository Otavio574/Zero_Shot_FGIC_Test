"""
Gera descrições (descriptors) otimizadas localmente para DCLIP,
usando modelos open-source via Hugging Face Transformers.
Compatível com qualquer modelo instruct (Llama, Mistral, Gemma, Qwen...).
"""

import os
import json
import time
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from transformers import pipeline

# ============================================================
# CONFIGURAÇÃO DO MODELO LOCAL
# ============================================================

# ⚠️ Troque aqui o modelo se quiser outro (ex: "meta-llama/Meta-Llama-3.1-8B-Instruct")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
SUMMARY_PATH = Path("outputs/analysis/summary.json")

print(f"🚀 Carregando modelo local: {MODEL_NAME}")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
)
print("✅ Modelo carregado com sucesso!")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def sanitize_class_name(class_name: str) -> str:
    """Remove prefixos numéricos e substitui underscores."""
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()


def get_dataset_category(dataset_name: str) -> str:
    """Mapeia dataset → categoria contextual para o modelo."""
    name = dataset_name.lower().replace("-", "_")
    if "bird" in name or "cub" in name:
        return "bird"
    if "dog" in name:
        return "dog"
    if "car" in name:
        return "car"
    if "aircraft" in name or "plane" in name:
        return "aircraft"
    if "leaf" in name or "plant" in name or "flower" in name:
        return "plant"
    if "food" in name:
        return "food"
    return "object"


def load_datasets_from_summary(summary_path: str) -> Dict[str, str]:
    """Carrega datasets do summary.json (compatível com o formato real)."""
    path = Path(summary_path)
    if not path.exists():
        print(f"❌ summary.json não encontrado em: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = {}
    # adapta para o formato usado por você
    for d in data:
        if "dataset" in d and "path" in d:
            datasets[d["dataset"]] = d["path"]

    return datasets

# ============================================================
# GERADOR PRINCIPAL
# ============================================================

class LocalDescriptorGenerator:
    """Gera descrições com LLM local (sem limite de cota)."""

    def __init__(self, pipe):
        self.pipe = pipe

    def generate_description(self, concept: str, category: str) -> str:
        """Gera uma descrição rica e visual."""
        prompt = (
            f"Analyze and describe a photo of a {concept}, a type of {category}. "
            "Your description must be highly focused on **distinctive and discriminative visual attributes** "
            "(e.g., specific colors, unique markings, structural shape, or notable environment). "
            "The description must be rich in visual detail. "
            "Respond with a single, concise sentence that is useful for distinguishing this concept from others, "
            "always starting with 'a photo of a...'."
        )
        try:
            result = self.pipe(prompt, max_new_tokens=60, temperature=0.8, do_sample=True)
            text = result[0]["generated_text"]
            # filtra apenas a resposta principal
            if "a photo of" in text:
                text = text[text.find("a photo of"):]
            return text.strip().replace("\n", " ")
        except Exception as e:
            print(f"⚠️ Erro gerando descrição para '{concept}': {e}")
            return f"a photo of a {concept}, a type of {category}."

    def process_dataset(self, dataset_name: str, dataset_path: str, output_dir: str):
        """Processa todas as classes de um dataset e salva JSON."""
        print(f"\n📘 Dataset: {dataset_name}")
        category = get_dataset_category(dataset_name)
        dataset_path = Path(dataset_path)
        classes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])

        if not classes:
            print(f"⚠️ Nenhuma pasta de classe encontrada em {dataset_path}")
            return

        output_path = Path(output_dir) / f"{dataset_name}_local_descriptors.json"
        os.makedirs(output_path.parent, exist_ok=True)

        descriptors = {}
        for c in tqdm(classes, desc=f"Gerando descrições ({dataset_name})"):
            concept = sanitize_class_name(c.name)
            desc = self.generate_description(concept, category)
            descriptors[c.name] = desc
            # pequeno delay opcional se quiser limitar uso de GPU
            time.sleep(0.05)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(descriptors, f, indent=2, ensure_ascii=False)

        print(f"✅ Descrições salvas em {output_path}")
        print(f"🧩 Total de classes: {len(descriptors)}")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    OUTPUT_DIR = "descriptors_local_llm"

    datasets = load_datasets_from_summary(SUMMARY_PATH)
    if not datasets:
        print("❌ Nenhum dataset carregado.")
        return

    generator = LocalDescriptorGenerator(pipe)

    print(f"\n{'='*70}")
    print(f"🚀 Iniciando geração de descrições locais (sem API)")
    print(f"📊 Total de datasets: {len(datasets)}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    for dataset_name, dataset_path in datasets.items():
        generator.process_dataset(dataset_name, dataset_path, OUTPUT_DIR)

    print("\n🎯 Geração finalizada com sucesso!")
    print(f"Todos os descritores foram salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
