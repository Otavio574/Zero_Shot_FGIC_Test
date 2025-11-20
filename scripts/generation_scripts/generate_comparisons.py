"""
Comparative-CLIP Descriptor Generator (Qwen2-7B, Batched + Fixed)
---------------------------------------------------------

1. Extrai classes de cada dataset via summary.json
2. Para cada classe:
      - encontra K classes mais similares via CLIP
      - gera descritores comparativos via QWEN2 (batched)
3. Salva cada dataset em descriptors_comparative/<dataset>_comparative.json

✅ FIX: Usa chat template correto do Qwen2 para evitar retornar prompt completo
"""

import os
import json
import torch
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import re

# ============================
#  CONFIG
# ============================

SUMMARY_PATH = Path("outputs/analysis/summary.json")
OUTPUT_DIR = Path("descriptors_comparative_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Comparative config ----
# 🔥 AJUSTE: Para gerar descritores como no paper original (80+ por classe)
K_SIMILAR_CLASSES = 10  # Aumentado de 5 para 10
NUM_COMPARISONS_PER_PAIR = 8  # Aumentado de 3 para 8 (10 × 8 = 80 descritores!)
BATCH_SIZE = 32  # Reduzido porque vamos gerar mais tokens

# ---- QWEN2 ----
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL = "Qwen/Qwen2-7B-Instruct"

print("🔄 Carregando Qwen2 7B…")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
qwen = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("✅ Qwen carregado!\n")


# ============================
#  LOAD DATASETS
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


# ============================
#  EXTRACT CLASSES
# ============================

def extract_classes_from_dataset(dataset_path: str):
    root = Path(dataset_path)

    if not root.exists():
        print(f"⚠️  Dataset não encontrado, pulando: {dataset_path}")
        return []
    # caso clássico CUB/FGVC
    if (root / "train").exists():
        classes = [d.name for d in (root / "train").iterdir() if d.is_dir()]
    elif (root / "test").exists():
        classes = [d.name for d in (root / "test").iterdir() if d.is_dir()]
    else:
        # dataset tipo CUB_200_2011 inteiro
        classes = [d.name for d in root.iterdir() if d.is_dir()]

    return sorted(classes)


# ============================
#  SIMILARITY VIA CLIP
# ============================

def find_similar_classes(target_class, all_classes, model, clip_lib, k):
    prompts = [f"a photo of a {c.replace('_',' ')}" for c in all_classes]
    tokens = clip_lib.tokenize(prompts).to(DEVICE)

    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    target_idx = all_classes.index(target_class)

    target = text_emb[target_idx:target_idx+1].cpu().numpy()
    all_emb = text_emb.cpu().numpy()

    sims = cosine_similarity(target, all_emb)[0]

    top_idx = np.argsort(sims)[::-1][1:k+1]
    return [all_classes[i] for i in top_idx]


# ============================
#  EXTRACT JSON FROM RESPONSE
# ============================

def extract_json_from_text(text: str):
    """
    Extrai JSON de uma resposta que pode conter texto extra.
    Procura por array JSON válido na resposta.
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Procura por array JSON
    # Tenta encontrar algo como ["...", "...", ...]
    match = re.search(r'\[.*\]', text, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Fallback: tenta parsear o texto inteiro
    try:
        return json.loads(text.strip())
    except:
        return None


# ============================
#  BATCHED QWEN GENERATION (FIXED)
# ============================

def generate_comparative_descriptors_for_pairs_batched(
    target_class,
    similar_classes,
    num_desc,
    llm,
    tok
):
    """
    Gera descritores para várias classes similares em batch.
    
    ✅ FIX: Usa apply_chat_template do Qwen2 corretamente
    """
    target_readable = target_class.replace("_", " ")
    
    # 🔍 DEBUG
    print(f"   Gerando para {len(similar_classes)} classes similares, {num_desc} desc/par")

    # Prepara mensagens de chat para cada classe similar
    all_messages = []
    for sim in similar_classes:
        sim_readable = sim.replace("_", " ")

        messages = [
            {
                "role": "system",
                "content": "You are a visual recognition expert. Generate only valid JSON arrays. Do not include any explanatory text."
            },
            {
                "role": "user",
                "content": f"""Generate {num_desc} concise visual descriptors that distinguish "{target_readable}" from "{sim_readable}".

Focus on unique physical differences (size, color, shape, patterns, distinctive features).

Respond ONLY with a JSON array of strings. Example: ["darker wing edges", "longer beak", "more rounded head"]

JSON array:"""
            }
        ]
        
        all_messages.append(messages)

    all_desc = []

    # Processa em batches
    for i in range(0, len(all_messages), BATCH_SIZE):
        batch_messages = all_messages[i:i+BATCH_SIZE]
        
        # Aplica chat template para cada mensagem no batch
        batch_texts = [
            tok.apply_chat_template(
                msgs, 
                tokenize=False, 
                add_generation_prompt=True
            ) 
            for msgs in batch_messages
        ]
        
        # Tokeniza o batch
        encoded = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(llm.device)

        # Gera respostas (aumentado max_new_tokens para mais descritores)
        with torch.no_grad():
            generated = llm.generate(
                **encoded,
                max_new_tokens=400,  # Aumentado de 200 para 400
                temperature=0.7,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # Decodifica apenas a parte nova (sem o prompt)
        responses = []
        for gen_ids, input_len in zip(generated, encoded['input_ids'].shape[1:]):
            # Pega apenas os tokens gerados (não o prompt)
            new_tokens = gen_ids[len(encoded['input_ids'][0]):]
            response_text = tok.decode(new_tokens, skip_special_tokens=True)
            responses.append(response_text)

        # Parse das respostas
        for idx, resp in enumerate(responses):
            try:
                # Tenta extrair JSON da resposta
                parsed = extract_json_from_text(resp)
                
                if parsed and isinstance(parsed, list):
                    # Filtra strings válidas
                    valid_descs = [
                        d.strip() for d in parsed 
                        if isinstance(d, str) and len(d.strip()) > 5
                    ]
                    all_desc.extend(valid_descs)
                    
                    # 🔍 DEBUG: mostra quantos descritores foram extraídos
                    if len(all_desc) <= 20:  # Só mostra no começo
                        print(f"      ✓ Extraídos {len(valid_descs)} descritores")
                else:
                    # Fallback individual
                    all_desc.append(f"{target_readable} has distinct features")
                    print(f"      ⚠️  Parsed não é lista válida")
                    
            except Exception as e:
                print(f"   ⚠️  Parse error: {str(e)[:50]}")
                all_desc.append(f"{target_readable} has distinct features")

    # 🔍 DEBUG FINAL
    print(f"   ✅ Total gerado: {len(all_desc)} descritores")
    
    return all_desc


# ============================
#  GENERATE FULL DATASET
# ============================

def generate_comparative_descriptors_dataset(dataset_name, class_names, model, clip_lib):
    print(f"\n==============================")
    print(f"📊 Dataset: {dataset_name}")
    print(f"Classes: {len(class_names)}")
    print(f"K={K_SIMILAR_CLASSES} similares")
    print(f"Descritores por par={NUM_COMPARISONS_PER_PAIR}")
    print("==============================\n")

    results = {}

    for cls in tqdm(class_names, desc="Classes"):
        similar = find_similar_classes(cls, class_names, model, clip_lib, K_SIMILAR_CLASSES)

        descriptors = generate_comparative_descriptors_for_pairs_batched(
            cls,
            similar,
            NUM_COMPARISONS_PER_PAIR,
            qwen,
            tokenizer
        )

        results[cls] = descriptors
        
        # Debug: mostra alguns exemplos
        if len(results) <= 3:
            print(f"   {cls}: {descriptors[:3]}")

    return results


# ============================
#  SAVE
# ============================

def save_descriptors(d, name):
    out = OUTPUT_DIR / f"{name}_comparative.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Salvo: {out}")
    total = sum(len(v) for v in d.values())
    avg = total / len(d) if d else 0
    print(f"Total descritores: {total}")
    print(f"Média por classe: {avg:.1f}")
    
    # Mostra exemplos de descritores gerados
    if d:
        first_class = list(d.keys())[0]
        print(f"\nExemplo ({first_class}):")
        for desc in d[first_class][:5]:
            print(f"  - {desc}")


# ============================
#  MAIN
# ============================

def main():

    print("🔄 Carregando CLIP...")
    model, _ = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    print("✅ CLIP carregado!\n")

    datasets = load_datasets_from_summary(SUMMARY_PATH)

    if not datasets:
        print("❌ Nenhum dataset no summary.json")
        return

    for name, path in datasets.items():
        class_names = extract_classes_from_dataset(path)

        if not class_names:
            print(f"⚠️ Nada encontrado em {name}")
            continue

        desc = generate_comparative_descriptors_dataset(
            name, class_names, model, clip
        )

        save_descriptors(desc, name)


if __name__ == "__main__":
    main()