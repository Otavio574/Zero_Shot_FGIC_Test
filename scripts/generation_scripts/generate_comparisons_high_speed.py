"""
Comparative-CLIP Descriptor Generator (Qwen2-7B, Fast Version)
--------------------------------------------------------------

Modo focado em VELOCIDADE (Opção B):

- K_SIMILAR_CLASSES = 6
- NUM_COMPARISONS_PER_PAIR = 4  → ~24 descritores por classe
- BATCH_SIZE = 32
- max_new_tokens reduzido

Otimizações principais:
1. Embeddings de texto das classes CLIP são calculados APENAS 1x por dataset.
2. Similaridade entre classes feita via produto interno (cosine, pois já normalizado).
3. Geração com Qwen2 em batch com corte correto da parte gerada.
4. Parsing de JSON robusto para respostas “sujas”.
"""

import os
import json
import re
from pathlib import Path

import torch
import clip
import numpy as np
from tqdm import tqdm

# ============================
#  CONFIG GERAL
# ============================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
OUTPUT_DIR = Path("descriptors_comparative_fast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Comparative config (modo rápido) ----
K_SIMILAR_CLASSES = 6           # Menos vizinhos → mais rápido
NUM_COMPARISONS_PER_PAIR = 4    # Menos descritores por par
BATCH_SIZE = 32                 # Tamanho do batch para Qwen

# ============================
#  QWEN2
# ============================

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL = "Qwen/Qwen2-7B-Instruct"

print("🔄 Carregando Qwen2-7B-Instruct…")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
# Garantir pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

qwen = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("✅ Qwen carregado!\n")


# ============================
#  FUNÇÕES AUXILIARES
# ============================

def load_datasets_from_summary(path: Path):
    """
    summary.json: lista de objetos { "dataset": "...", "path": "..." }
    """
    if not path.exists():
        print("❌ summary.json não encontrado!")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = {}
    for item in data:
        if "dataset" in item and "path" in item:
            datasets[item["dataset"]] = item["path"]

    return datasets


def extract_classes_from_dataset(dataset_path: str):
    """
    Tenta detectar classes em:
    - dataset/train/<classe>/
    - dataset/test/<classe>/
    - dataset/<classe>/
    """
    root = Path(dataset_path)

    if not root.exists():
        print(f"⚠️  Dataset não encontrado, pulando: {dataset_path}")
        return []

    if (root / "train").exists():
        classes = [d.name for d in (root / "train").iterdir() if d.is_dir()]
    elif (root / "test").exists():
        classes = [d.name for d in (root / "test").iterdir() if d.is_dir()]
    else:
        classes = [d.name for d in root.iterdir() if d.is_dir()]

    return sorted(classes)


# ============================
#  CLIP – EMBEDDINGS DE CLASSES
# ============================

def compute_class_text_embeddings(class_names, model, clip_lib):
    """
    Calcula embeddings de texto CLIP para TODAS as classes 1x por dataset.
    Retorna tensor [num_classes, dim] normalizado (cosine ready).
    """
    prompts = [f"a photo of a {c.replace('_', ' ')}" for c in class_names]
    tokens = clip_lib.tokenize(prompts).to(DEVICE)

    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    return text_emb  # [C, D]


def find_similar_classes(target_index, class_names, class_embeds, k):
    """
    Encontra as k classes mais similares usando produto interno (cosine similarity),
    sem overflow em fp16 e sem conflitos de device (CPU vs CUDA).
    """

    device = class_embeds.device

    with torch.no_grad():
        target = class_embeds[target_index:target_index+1]           # [1, D]
        sims = (target @ class_embeds.T).squeeze(0).clone()          # [C]

        # Máscara booleana no MESMO device
        mask = torch.ones_like(sims, dtype=torch.bool, device=device)
        mask[target_index] = False

        # Similaridades válidas (remove a própria classe)
        valid_sims = sims[mask]                                      # [C-1]

        # top-k dentro dos válidos
        topk_vals, topk_idx_valid = torch.topk(valid_sims, k=k)

        # Mapeia de volta para índices originais
        all_indices = torch.arange(len(class_names), device=device)  # NO MESMO DEVICE
        original_indices = all_indices[mask][topk_idx_valid]

    return [class_names[int(i)] for i in original_indices]




# ============================
#  PARSING DE JSON DE RESPOSTA
# ============================

def extract_json_from_text(text: str):
    """
    Extrai JSON de array da resposta do modelo.

    - Remove blocos de markdown
    - Tenta achar algo tipo ["...", "..."]
    - Se falhar, tenta json.loads no texto inteiro
    """
    # Remove ``` e ```json
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)

    # Normalizar aspas esquisitas (caso Qwen invente aspas “ ”)
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

    # Tentar encontrar um array JSON
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        candidate = match.group(0).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: tentar usar o texto inteiro
    try:
        return json.loads(text.strip())
    except Exception:
        return None


# ============================
#  QWEN – GERAÇÃO BATCHEADA
# ============================

def generate_comparative_descriptors_for_pairs_batched(
    target_class: str,
    similar_classes: list,
    num_desc: int,
    llm,
    tok
):
    """
    Gera descritores comparativos entre target_class e cada classe em similar_classes,
    em batches. Retorna uma LISTA de descritores (strings).
    """

    target_readable = target_class.replace("_", " ")
    all_messages = []

    for sim in similar_classes:
        sim_readable = sim.replace("_", " ")
        user_content = f"""You are a visual recognition expert.

Target class: "{target_readable}"
Similar class: "{sim_readable}"

Generate {num_desc} concise visual descriptors that clearly distinguish "{target_readable}" from "{sim_readable}" in photos.

Focus ONLY on:
- physical differences (size, shape, color, markings)
- distinctive local features (head, wings, tail, pattern, beak, legs)
- visible appearance differences

Respond ONLY with a JSON array of SHORT strings.
Example: ["darker wing edges", "longer, curved beak", "more rounded head"]

JSON array:"""

        messages = [
            {
                "role": "system",
                "content": "You generate ONLY valid JSON arrays of strings for visual descriptors. No explanations."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        all_messages.append(messages)

    all_desc = []

    # Processa em batches
    for i in range(0, len(all_messages), BATCH_SIZE):
        batch_messages = all_messages[i : i + BATCH_SIZE]

        # Aplica template de chat específico do Qwen2
        batch_texts = [
            tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in batch_messages
        ]

        encoded = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        encoded = {k: v.to(llm.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = llm.generate(
                **encoded,
                max_new_tokens=160,   # menor para velocidade
                temperature=0.7,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]

        # Para cada item do batch, recorta só a parte nova
        for gen_ids, mask in zip(generated, attn_mask):
            input_len = int(mask.sum().item())
            new_tokens = gen_ids[input_len:]
            resp_text = tok.decode(new_tokens, skip_special_tokens=True)

            parsed = extract_json_from_text(resp_text)

            if isinstance(parsed, list):
                # filtra strings minimamente decentes
                valid = [
                    s.strip() for s in parsed
                    if isinstance(s, str) and len(s.strip()) > 3
                ]
                all_desc.extend(valid)
            else:
                # fallback: coloca algo genérico só pra não perder
                all_desc.append(f"{target_readable} has distinctive visual features")

    # Remoção de duplicatas muito repetidas (sem ser muito agressivo)
    unique_desc = []
    seen = set()
    for d in all_desc:
        key = d.lower()
        if key not in seen:
            seen.add(key)
            unique_desc.append(d)

    return unique_desc


# ============================
#  GERAÇÃO POR DATASET
# ============================

def generate_comparative_descriptors_dataset(dataset_name, class_names, model, clip_lib):
    print(f"\n==============================")
    print(f"📊 Dataset: {dataset_name}")
    print(f"Classes: {len(class_names)}")
    print(f"K similares: {K_SIMILAR_CLASSES}")
    print(f"Descritores por par: {NUM_COMPARISONS_PER_PAIR}")
    print("==============================\n")

    # 1) Pré-calcular embeddings de todas as classes uma única vez
    print("🔄 Calculando embeddings CLIP de classes (1x por dataset)...")
    class_embeds = compute_class_text_embeddings(class_names, model, clip_lib)
    print("✅ Embeddings prontos!\n")

    results = {}

    for idx, cls in enumerate(tqdm(class_names, desc="Classes")):
        similar = find_similar_classes(idx, class_names, class_embeds, K_SIMILAR_CLASSES)

        descriptors = generate_comparative_descriptors_for_pairs_batched(
            cls,
            similar,
            NUM_COMPARISONS_PER_PAIR,
            qwen,
            tokenizer
        )

        results[cls] = descriptors

    return results


# ============================
#  SALVAR
# ============================

def save_descriptors(d, name):
    out = OUTPUT_DIR / f"{name}_comparative_fast.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Salvo: {out}")
    total = sum(len(v) for v in d.values())
    avg = total / len(d) if d else 0
    print(f"Total descritores: {total}")
    print(f"Média por classe: {avg:.1f}")

    # Mostra exemplos
    if d:
        any_class = next(iter(d.keys()))
        print(f"\nExemplo de classe: {any_class}")
        for desc in d[any_class][:5]:
            print(f"  - {desc}")


# ============================
#  MAIN
# ============================

def main():
    print("🎯 Comparative-CLIP Descriptor Generator (FAST)")
    print(f"💻 Device: {DEVICE}")
    print(f"🤖 Qwen model: {QWEN_MODEL}\n")

    print("🔄 Carregando CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ CLIP carregado!\n")

    datasets = load_datasets_from_summary(SUMMARY_PATH)
    if not datasets:
        print("❌ Nenhum dataset no summary.json")
        return

    for name, path in datasets.items():
        print(f"\n==============================")
        print(f"📁 Dataset: {name}")
        print(f"📍 Path: {path}")
        print("==============================")

        class_names = extract_classes_from_dataset(path)

        if not class_names:
            print(f"⚠️  Nenhuma classe encontrada em {name}, pulando...")
            continue

        desc = generate_comparative_descriptors_dataset(
            name, class_names, model, clip
        )

        save_descriptors(desc, name)


if __name__ == "__main__":
    main()
