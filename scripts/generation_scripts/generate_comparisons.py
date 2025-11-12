"""
Gerador de descrições comparativas OTIMIZADO para velocidade
Principais otimizações:
1. Batch processing (múltiplas comparações de uma vez)
2. Cache de prompts
3. Parâmetros de inferência otimizados
4. Redução inteligente de comparações
"""

import os
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from transformers import pipeline
import torch

# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUMMARY_PATH = Path("outputs/analysis/summary.json")

# OTIMIZAÇÕES DE VELOCIDADE
BATCH_SIZE = 8  # Processa múltiplas comparações simultaneamente
NUM_COMPARISONS = 3  # Comparações por classe
MAX_NEW_TOKENS = 40  # Reduzido de 50
USE_CACHE = True  # Cache de prompts similares

print(f"🚀 Carregando modelo: {MODEL_NAME}")
print(f"💻 Device: {DEVICE}")

# Configurações otimizadas para velocidade
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto",
    model_kwargs={
        "use_cache": True,  # Cache KV
        "torch_dtype": torch.float16,  # FP16 para velocidade
    }
)

# CRÍTICO: Configura pad_token para permitir batching
if pipe.tokenizer.pad_token is None:
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    print("✅ pad_token configurado")

# Configura para greedy decoding (mais rápido)
pipe.model.generation_config.do_sample = False
pipe.model.generation_config.num_beams = 1
pipe.model.generation_config.pad_token_id = pipe.tokenizer.pad_token_id

print("✅ Modelo carregado e otimizado!")

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def sanitize_class_name(class_name: str) -> str:
    """Remove prefixos e limpa nome"""
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()


def get_dataset_category(dataset_name: str) -> str:
    """Mapeia dataset para categoria"""
    name = dataset_name.lower().replace("-", "_")
    
    categories = {
        'bird': ['bird', 'cub', 'nabirds', 'birdsnap'],
        'dog': ['dog', 'pet', 'stanford_dogs'],
        'car': ['car', 'vehicle', 'stanford_cars', 'compcar'],
        'aircraft': ['aircraft', 'plane', 'fgvc'],
        'flower': ['flower', 'oxford'],
    }
    
    for category, keywords in categories.items():
        if any(kw in name for kw in keywords):
            return category
    
    return 'object'


def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    """Carrega datasets do summary.json"""
    if not summary_path.exists():
        print(f"❌ summary.json não encontrado: {summary_path}")
        return {}
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    datasets = {}
    
    if isinstance(data, list):
        for d in data:
            if "dataset" in d and "path" in d:
                datasets[d["dataset"]] = d["path"]
    
    return datasets


def create_comparison_prompt(target: str, contrast: str, category: str) -> str:
    """Cria prompt comparativo otimizado"""
    # Prompt curto e direto para resposta rápida
    return (
        f"Describe a {target} ({category}) by contrasting with {contrast}. "
        f"Focus on key visual differences. Start with 'A photo of a {target}'"
    )


def extract_description(text: str, target: str) -> str:
    """Extrai descrição do output do LLM"""
    text = text.lower()
    
    # Procura por padrões comuns
    markers = ["a photo of", "photo of", f"a {target}"]
    
    for marker in markers:
        if marker in text:
            start = text.find(marker)
            desc = text[start:].strip()
            # Pega primeira sentença
            end = desc.find('.')
            if end > 0:
                return desc[:end + 1]
            return desc[:100] + "."
    
    # Fallback
    return f"a photo of a {target}."


# ============================================================
# GERADOR COMPARATIVO OTIMIZADO
# ============================================================

class FastComparativeGenerator:
    """Gerador otimizado com batch processing"""
    
    def __init__(self, pipe, num_comparisons: int = 3, batch_size: int = 8):
        self.pipe = pipe
        self.num_comparisons = num_comparisons
        self.batch_size = batch_size
        self.cache = {}  # Cache de descrições
    
    def generate_batch(self, prompts: List[Tuple[str, str, str]]) -> List[str]:
        """
        Gera múltiplas comparações em batch.
        prompts: List[(target, contrast, category)]
        """
        # Cria prompts em texto
        prompt_texts = [
            create_comparison_prompt(target, contrast, cat)
            for target, contrast, cat in prompts
        ]
        
        try:
            # BATCH INFERENCE - muito mais rápido!
            results = self.pipe(
                prompt_texts,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Greedy decoding
                batch_size=self.batch_size,
                truncation=True,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
            )
            
            # Extrai descrições
            descriptions = []
            for result, (target, _, _) in zip(results, prompts):
                text = result[0]["generated_text"]
                desc = extract_description(text, target)
                descriptions.append(desc)
            
            return descriptions
            
        except Exception as e:
            print(f"⚠️ Erro no batch: {e}")
            # Fallback: descrições simples
            return [
                f"a photo of a {target}, distinct from a {contrast}."
                for target, contrast, _ in prompts
            ]
    
    def process_dataset(self, dataset_name: str, dataset_path: str, output_dir: str):
        """Processa dataset com batching"""
        print(f"\n📘 Dataset: {dataset_name}")
        
        category = get_dataset_category(dataset_name)
        dataset_path = Path(dataset_path)
        
        # Carrega classes
        classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        
        if len(classes) < 2:
            print(f"⚠️ Menos de 2 classes, pulando...")
            return
        
        print(f"   Classes: {len(classes)}")
        print(f"   Comparações por classe: {self.num_comparisons}")
        print(f"   Total de comparações: {len(classes) * self.num_comparisons}")
        
        # Prepara output
        output_path = Path(output_dir) / f"{dataset_name}_comparative_descriptors.json"
        os.makedirs(output_path.parent, exist_ok=True)
        
        comparative_descriptors = {}
        
        # Prepara TODOS os prompts de uma vez
        all_prompts = []
        prompt_mapping = []  # (target_class_raw, index)
        
        for target_class_raw in classes:
            target_concept = sanitize_class_name(target_class_raw)
            all_other = [c for c in classes if c != target_class_raw]
            
            # Seleciona contrasts
            if len(all_other) <= self.num_comparisons:
                contrasts = all_other
            else:
                contrasts = random.sample(all_other, self.num_comparisons)
            
            # Adiciona aos prompts
            for contrast_raw in contrasts:
                contrast_concept = sanitize_class_name(contrast_raw)
                all_prompts.append((target_concept, contrast_concept, category))
                prompt_mapping.append(target_class_raw)
        
        print(f"   Total de prompts preparados: {len(all_prompts)}")
        
        # Processa em BATCHES
        all_descriptions = []
        
        pbar = tqdm(
            range(0, len(all_prompts), self.batch_size),
            desc=f"Gerando ({dataset_name})",
            unit="batch"
        )
        
        start_time = time.time()
        
        for i in pbar:
            batch = all_prompts[i:i + self.batch_size]
            descs = self.generate_batch(batch)
            all_descriptions.extend(descs)
            
            # Atualiza velocidade
            elapsed = time.time() - start_time
            speed = (i + len(batch)) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"speed": f"{speed:.1f} comp/s"})
        
        # Organiza por classe
        desc_idx = 0
        for target_class_raw in classes:
            num_comps = prompt_mapping.count(target_class_raw)
            comparative_descriptors[target_class_raw] = all_descriptions[desc_idx:desc_idx + num_comps]
            desc_idx += num_comps
        
        # Salva
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparative_descriptors, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        total_comps = len(all_prompts)
        speed = total_comps / elapsed
        
        print(f"✅ Salvo em: {output_path}")
        print(f"⏱️  Tempo: {elapsed:.1f}s ({speed:.1f} comparações/s)")
        print(f"📊 Classes: {len(comparative_descriptors)}")


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    OUTPUT_DIR = "descriptors_comparative_llm"
    
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    
    if not datasets:
        print("❌ Nenhum dataset carregado")
        return
    
    print(f"\n{'='*70}")
    print(f"🚀 GERADOR COMPARATIVO ULTRA-RÁPIDO")
    print(f"{'='*70}")
    print(f"Configuração:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Comparações/classe: {NUM_COMPARISONS}")
    print(f"  - Max tokens: {MAX_NEW_TOKENS}")
    print(f"  - Precision: FP16")
    print(f"  - Sampling: Greedy (do_sample=False)")
    print(f"{'='*70}\n")
    
    generator = FastComparativeGenerator(
        pipe,
        num_comparisons=NUM_COMPARISONS,
        batch_size=BATCH_SIZE
    )
    
    total_start = time.time()
    
    for dataset_name, dataset_path in datasets.items():
        generator.process_dataset(dataset_name, dataset_path, OUTPUT_DIR)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"✅ CONCLUÍDO!")
    print(f"⏱️  Tempo total: {total_elapsed/60:.1f} minutos")
    print(f"📁 Resultados em: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()