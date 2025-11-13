"""
DCLIP Comparativo com RAG (Retrieval Augmented Generation)
✅ Usa descrições LLM existentes como contexto
✅ CLIP para encontrar classes similares
✅ Prompt otimizado para comparação
✅ Batching ultra-rápido
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from transformers import pipeline, AutoProcessor, AutoModel

logging.basicConfig(level=logging.ERROR)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Diretório onde estão as descrições LLM por dataset
DESCRIPTIONS_DIR = Path("descriptors_local_llm")  # Ajuste conforme necessário

CLIP_MODEL = "openai/clip-vit-base-patch32"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUMMARY_PATH = Path("outputs/analysis/summary.json")

NUM_SIMILAR = 2      # Comparações por classe
BATCH_SIZE = 16      # Batch size
MAX_NEW_TOKENS = 50  # Reduzido - comparações são mais curtas

print("🚀 Carregando CLIP...")
clip_model = AutoModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()
print("✅ CLIP carregado!")

print("🚀 Carregando LLM...")
llm_pipe = pipeline(
    "text-generation",
    model=LLM_MODEL,
    device_map="auto",
    torch_dtype="auto",
)

if llm_pipe.tokenizer.pad_token is None:
    llm_pipe.tokenizer.pad_token = llm_pipe.tokenizer.eos_token
    llm_pipe.model.config.pad_token_id = llm_pipe.tokenizer.eos_token_id

print("✅ LLM carregado!")

# ============================================================
# FUNÇÕES
# ============================================================

def sanitize_class_name(class_name: str) -> str:
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()


def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    if not summary_path.exists():
        return {}

    with open(summary_path, "r") as f:
        data = json.load(f)

    datasets = {}
    if isinstance(data, list):
        for d in data:
            if "dataset" in d and "path" in d:
                datasets[d["dataset"]] = d["path"]

    return datasets


def load_descriptions_for_dataset(dataset_name: str, descriptions_dir: Path) -> Optional[Dict[str, str]]:
    """Carrega descrições LLM existentes para o dataset"""
    possible_paths = [
        descriptions_dir / f"{dataset_name}_local_descriptors.json",
        descriptions_dir / f"{dataset_name}_llm_descriptors.json",
        descriptions_dir / f"{dataset_name}.json",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    descs = json.load(f)
                print(f"   ✅ Descrições carregadas: {path.name} ({len(descs)} classes)")
                return descs
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {path}: {e}")

    print(f"   ⚠️ Nenhuma descrição encontrada para {dataset_name}")
    return None


def find_similar_classes_clip(target_class: str, all_classes: List[str], num_similar: int) -> List[str]:
    """Encontra classes similares usando CLIP"""
    texts = [f"a photo of a {sanitize_class_name(cls)}" for cls in all_classes]
    target_text = f"a photo of a {sanitize_class_name(target_class)}"

    with torch.no_grad():
        inputs = clip_processor(
            text=texts + [target_text],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        embeds = clip_model.get_text_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)

    class_embeds = embeds[:-1]
    target_embed = embeds[-1:]
    similarities = (target_embed @ class_embeds.T).squeeze(0)

    target_idx = all_classes.index(target_class)
    similarities[target_idx] = -999

    top_k = similarities.topk(min(num_similar, len(all_classes) - 1))
    similar_indices = top_k.indices.cpu().tolist()

    return [all_classes[i] for i in similar_indices]


def create_comparison_prompt(target_raw: str, similar_raw: str, target_desc: str, similar_desc: str) -> str:
    """Cria prompt RAG otimizado para comparação"""
    target_clean = sanitize_class_name(target_raw)
    similar_clean = sanitize_class_name(similar_raw)

    max_desc_len = 200
    if len(target_desc) > max_desc_len:
        target_desc = target_desc[:max_desc_len] + "..."
    if len(similar_desc) > max_desc_len:
        similar_desc = similar_desc[:max_desc_len] + "..."

    prompt = (
        f"Compare two classes:\n\n"
        f"Class A ({target_clean}):\n{target_desc}\n\n"
        f"Class B ({similar_clean}):\n{similar_desc}\n\n"
        f"Task: Identify the single most important visual difference between them. "
        f"Write one sentence starting with 'The {target_clean}' that highlights "
        f"how it differs from the {similar_clean}."
    )

    return f"[INST] {prompt} [/INST]"


def extract_comparison(text: str, target_clean: str) -> str:
    """Extrai a comparação do output do LLM"""
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[1].strip()

    start_pattern = f"The {target_clean}"

    if start_pattern.lower() in text.lower():
        start_idx = text.lower().find(start_pattern.lower())
        desc = text[start_idx:].strip()
        sentences = desc.split('.')
        result = sentences[0].strip() + '.'

        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        if len(result) > 250:
            result = result[:247] + "..."
        return result

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines:
        if 15 < len(line) < 250:
            if any(w in line.lower() for w in ['while', 'whereas', 'unlike', 'compared', 'has', 'is']):
                return line if line.endswith('.') else line + '.'

    return f"The {target_clean} has distinctive features distinguishing it."


# ============================================================
# GERADOR
# ============================================================

class DCLIPRAGGenerator:
    """Gerador com RAG usando descrições existentes"""

    def __init__(self, llm_pipe, descriptions_dir: Path, num_similar: int = 2, batch_size: int = 16):
        self.llm_pipe = llm_pipe
        self.descriptions_dir = descriptions_dir
        self.num_similar = num_similar
        self.batch_size = batch_size

    def process_dataset(self, dataset_name: str, dataset_path: str, output_dir: str) -> tuple[bool, str]:
        """Processa um dataset, retornando (sucesso: bool, razão: str)"""
        print(f"\n📘 Dataset: {dataset_name}")

        output_path = Path(output_dir) / f"{dataset_name}_comparative_descriptors.json"
        os.makedirs(output_path.parent, exist_ok=True)

        if output_path.exists():
            return False, "Already Exists"

        descriptions = load_descriptions_for_dataset(dataset_name, self.descriptions_dir)
        if not descriptions:
            return False, "No Base Descriptions"

        dataset_path = Path(dataset_path)
        classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

        if len(classes) < 2:
            return False, "Too Few Classes"

        print(f"   Classes: {len(classes)}")
        print(f"   Comparações/classe: {self.num_similar}")

        print("   🔍 Similaridades (CLIP)...")
        class_similars = {}

        for target in tqdm(classes, desc="CLIP", leave=False):
            similars = find_similar_classes_clip(target, classes, self.num_similar)
            class_similars[target] = similars

        print("   📝 Prompts RAG...")
        all_prompts = []
        prompt_map = []

        for target_raw in classes:
            target_clean = sanitize_class_name(target_raw)
            target_desc = descriptions.get(target_raw, f"A photo of a {target_clean}.")

            for similar_raw in class_similars[target_raw]:
                similar_desc = descriptions.get(similar_raw, f"A photo of a {sanitize_class_name(similar_raw)}.")
                prompt = create_comparison_prompt(target_raw, similar_raw, target_desc, similar_desc)
                all_prompts.append(prompt)
                prompt_map.append((target_raw, target_clean))

        print(f"   Total prompts: {len(all_prompts)}")

        print(f"   🤖 Gerando (batch={self.batch_size})...")
        try:
            results = self.llm_pipe(
                all_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                batch_size=self.batch_size,
            )
        except Exception as e:
            print(f"   ❌ Erro na geração LLM: {e}")
            return False, f"LLM Error: {e}"

        comparative_descriptors = {cls: [] for cls in classes}

        for i, result in enumerate(tqdm(results, desc="Extração", leave=False)):
            target_raw, target_clean = prompt_map[i]
            text = result[0]["generated_text"]
            comparison = extract_comparison(text, target_clean)
            comparative_descriptors[target_raw].append(comparison)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparative_descriptors, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Salvo: {output_path.name}")
        print(f"   📊 {len(comparative_descriptors)} classes")
        return True, "Success"


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR = "descriptors_comparative_rag"
    datasets = load_datasets_from_summary(SUMMARY_PATH)

    if not datasets:
        print("❌ Nenhum dataset")
        return

    if not DESCRIPTIONS_DIR.exists():
        print(f"❌ Diretório de descrições não encontrado: {DESCRIPTIONS_DIR}")
        print("   Crie este diretório e coloque os JSONs de descrições lá.")
        return

    print(f"\n{'='*70}")
    print("🚀 DCLIP COMPARATIVO COM RAG")
    print(f"{'='*70}")
    print("✅ Usa descrições LLM existentes como contexto")
    print("✅ CLIP para classes similares")
    print("✅ Prompt otimizado para comparação")
    print("✅ Batching para velocidade")
    print("")
    print(f"   Descrições base: {DESCRIPTIONS_DIR}")
    print(f"   Comparações/classe: {NUM_SIMILAR}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max tokens: {MAX_NEW_TOKENS}")
    print(f"{'='*70}\n")

    generator = DCLIPRAGGenerator(
        llm_pipe,
        DESCRIPTIONS_DIR,
        num_similar=NUM_SIMILAR,
        batch_size=BATCH_SIZE
    )

    success = 0
    skipped_log = {}

    for dataset_name, dataset_path in datasets.items():
        try:
            processed, reason = generator.process_dataset(dataset_name, dataset_path, OUTPUT_DIR)
            if processed:
                success += 1
            else:
                skipped_log[dataset_name] = reason
                if reason not in ["LLM Error", "Already Exists"]:
                    print(f"   ➡️  Razão do pulo: {reason}")
        except Exception as e:
            skipped_log[dataset_name] = f"Fatal Error: {e}"
            print(f"   ❌ Erro fatal ao processar {dataset_name}: {e}")

    print(f"\n{'='*70}")
    print("✅ CONCLUÍDO!")
    print(f"{'='*70}")
    print(f"   Processados com sucesso: {success}")
    print(f"   Pulados/Falhas: {len(skipped_log)}")

    if skipped_log:
        print("\n### ⏭️ Detalhes dos Datasets Pulados/Falhos:")
        for name, reason in skipped_log.items():
            status = "✅ JÁ PRONTO" if reason == "Already Exists" else f"❌ {reason}"
            print(f"   - {name:25s}: {status}")

    print(f"\n   Resultados salvos em: {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
