"""
Gerador DCLIP Comparativo - MÉTODO CORRETO DO PAPER
1. ✅ Usa CLIP para encontrar classes similares (não aleatório)
2. ✅ Prompt exato do paper
3. ✅ Few-shot learning (2 exemplos de 10)
4. ✅ Batching para velocidade
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
from transformers import pipeline, AutoProcessor, AutoModel

logging.basicConfig(level=logging.ERROR)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# CLIP para similaridade
CLIP_MODEL = "openai/clip-vit-base-patch32"

# LLM para comparações
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUMMARY_PATH = Path("outputs/analysis/summary.json")

# Parâmetros
NUM_SIMILAR = 2      # Comparações por classe
BATCH_SIZE = 16     # Batch para LLM
MAX_NEW_TOKENS = 60

print(f"🚀 Carregando CLIP...")
clip_model = AutoModel.from_pretrained(CLIP_MODEL).to(DEVICE)
clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()
print("✅ CLIP carregado!")

print(f"🚀 Carregando LLM...")
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
# FEW-SHOT EXAMPLES (10 exemplos - pool do paper)
# ============================================================

FEW_SHOT_POOL = [
    {
        "Q": "What are useful features for distinguishing a golden retriever from a labrador retriever in the photo?",
        "A": "There are several useful visual features to tell the photo is a golden retriever, not a labrador retriever. The golden retriever has longer, wavier golden fur, while the labrador has shorter coat. The golden retriever's muzzle is more tapered."
    },
    {
        "Q": "What are useful features for distinguishing a boeing 737 from a boeing 747 in the photo?",
        "A": "There are several useful visual features to tell the photo is a boeing 737, not a boeing 747. The 737 has a single-deck fuselage, while the 747 has a distinctive double-deck hump. The 737 has two engines, the 747 has four."
    },
    {
        "Q": "What are useful features for distinguishing a cardinal from a blue jay in the photo?",
        "A": "There are several useful visual features to tell the photo is a cardinal, not a blue jay. The cardinal has bright red plumage with a crest, while the blue jay is blue and white. The cardinal has a shorter, conical beak."
    },
    {
        "Q": "What are useful features for distinguishing a rose from a tulip in the photo?",
        "A": "There are several useful visual features to tell the photo is a rose, not a tulip. The rose has layered petals in spiral pattern, while tulip has smooth cup-shaped petals. Roses have thorny stems, tulips are smooth."
    },
    {
        "Q": "What are useful features for distinguishing a siamese cat from a persian cat in the photo?",
        "A": "There are several useful visual features to tell the photo is a siamese cat, not a persian cat. The siamese has short coat with color points on face, ears, and paws, while persian has long fluffy fur. Siamese has triangular face shape."
    },
    {
        "Q": "What are useful features for distinguishing a hawk from an eagle in the photo?",
        "A": "There are several useful visual features to tell the photo is a hawk, not an eagle. Hawks are smaller with rounded wing tips, while eagles have larger wingspan with finger-like feathers. Hawks have smaller beaks."
    },
    {
        "Q": "What are useful features for distinguishing a sedan from a coupe in the photo?",
        "A": "There are several useful visual features to tell the photo is a sedan, not a coupe. The sedan has four doors and longer wheelbase, while coupe has two doors and sportier roofline. Sedans have vertical rear windows."
    },
    {
        "Q": "What are useful features for distinguishing a maple leaf from an oak leaf in the photo?",
        "A": "There are several useful visual features to tell the photo is a maple leaf, not an oak leaf. The maple has pointed lobes with deep indentations, while oak has rounded lobes. Maple leaves are more symmetrical."
    },
    {
        "Q": "What are useful features for distinguishing a sparrow from a finch in the photo?",
        "A": "There are several useful visual features to tell the photo is a sparrow, not a finch. Sparrows have streaked brown plumage, while finches are more colorful. Sparrows have distinctive eye stripe pattern."
    },
    {
        "Q": "What are useful features for distinguishing a sunflower from a daisy in the photo?",
        "A": "There are several useful visual features to tell the photo is a sunflower, not a daisy. The sunflower is larger with dark center disc, while daisy is smaller with yellow center. Sunflower petals are broader."
    }
]

# ============================================================
# FUNÇÕES
# ============================================================

def sanitize_class_name(class_name: str) -> str:
    parts = class_name.split('.', 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.lower().replace('_', ' ').replace('-', ' ').strip()


def get_dataset_category(dataset_name: str) -> str:
    name = dataset_name.lower()
    if any(k in name for k in ['bird', 'cub']): return 'bird'
    if 'dog' in name: return 'dog breed'
    if any(k in name for k in ['car', 'vehicle']): return 'car'
    if any(k in name for k in ['aircraft', 'plane']): return 'aircraft'
    if 'flower' in name: return 'flower'
    return 'object'


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


def find_similar_classes_clip(
    target_class: str,
    all_classes: List[str],
    num_similar: int
) -> List[str]:
    """
    ✅ MÉTODO DO PAPER: Encontra classes similares via CLIP text similarity
    Seção 3.2.1 do paper
    """
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
    
    # Cosine similarity
    similarities = (target_embed @ class_embeds.T).squeeze(0)
    
    # Exclui próprio target
    target_idx = all_classes.index(target_class)
    similarities[target_idx] = -999
    
    # Top-k
    top_k = similarities.topk(min(num_similar, len(all_classes) - 1))
    similar_indices = top_k.indices.cpu().tolist()
    
    return [all_classes[i] for i in similar_indices]


def create_fewshot_prompt(target: str, similar: str) -> str:
    """
    ✅ MÉTODO DO PAPER: Prompt com few-shot learning
    Seção 3.2.2 do paper
    """
    # Seleciona 2 exemplos aleatórios (como no paper)
    examples = random.sample(FEW_SHOT_POOL, 2)
    
    # Formato Mistral-Instruct
    prompt = "[INST] You are a visual expert. Answer the following questions:\n\n"
    
    for ex in examples:
        prompt += f"Q: {ex['Q']}\n"
        prompt += f"A: {ex['A']}\n\n"
    
    # Pergunta real
    prompt += f"Q: What are useful features for distinguishing a {target} from a {similar} in the photo?\n"
    prompt += f"A: [/INST]"
    
    return prompt


def extract_features(text: str, target: str) -> str:
    """Extrai features comparativas da resposta"""
    # Remove o prompt se presente
    if "[/INST]" in text:
        text = text.split("[/INST]", 1)[1].strip()
    
    # Procura pelo padrão esperado
    lower_text = text.lower()
    
    markers = [
        "there are several useful visual features",
        f"a {target.lower()}, not a",
        "visual features"
    ]
    
    best_start = -1
    for marker in markers:
        pos = lower_text.find(marker)
        if pos != -1:
            best_start = pos
            break
    
    if best_start != -1:
        desc = text[best_start:].strip()
        
        # Limita a ~150 chars ou 2 sentenças
        sentences = desc.split('.')
        if len(sentences) > 2:
            desc = '.'.join(sentences[:2]) + '.'
        elif not desc.endswith('.'):
            desc += '.'
        
        # Capitaliza
        if desc and desc[0].islower():
            desc = desc[0].upper() + desc[1:]
        
        return desc
    
    # Fallback: pega texto útil
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines:
        if len(line) > 20 and len(line) < 200:
            if any(w in line.lower() for w in ['while', 'whereas', 'has', 'is']):
                return line if line.endswith('.') else line + '.'
    
    # Último fallback
    return f"Visual features distinguish the {target} from similar classes."


# ============================================================
# GERADOR
# ============================================================

class DCLIPGenerator:
    """Gerador seguindo método DCLIP do paper"""
    
    def __init__(self, llm_pipe, num_similar: int = 2, batch_size: int = 16):
        self.llm_pipe = llm_pipe
        self.num_similar = num_similar
        self.batch_size = batch_size
    
    def process_dataset(self, dataset_name: str, dataset_path: str, output_dir: str):
        print(f"\n📘 Dataset: {dataset_name}")
        
        category = get_dataset_category(dataset_name)
        dataset_path = Path(dataset_path)
        
        classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        
        if len(classes) < 2:
            print(f"⚠️ Menos de 2 classes")
            return
        
        print(f"   Classes: {len(classes)}")
        print(f"   Similaridades/classe: {self.num_similar}")
        
        output_path = Path(output_dir) / f"{dataset_name}_comparative_descriptors.json"
        os.makedirs(output_path.parent, exist_ok=True)
        
        # 1. Encontra similares com CLIP
        print("   🔍 Encontrando classes similares (CLIP)...")
        class_similars = {}
        
        for target in tqdm(classes, desc="Similaridades"):
            similars = find_similar_classes_clip(target, classes, self.num_similar)
            class_similars[target] = similars
        
        # 2. Prepara prompts
        print("   📝 Preparando prompts (few-shot)...")
        all_prompts = []
        prompt_map = []
        
        for target_raw in classes:
            target_clean = sanitize_class_name(target_raw)
            
            for similar_raw in class_similars[target_raw]:
                similar_clean = sanitize_class_name(similar_raw)
                
                prompt = create_fewshot_prompt(target_clean, similar_clean)
                all_prompts.append(prompt)
                prompt_map.append((target_raw, target_clean))
        
        print(f"   Total de prompts: {len(all_prompts)}")
        
        # 3. Gera em batch
        print(f"   🤖 Gerando comparações (batch={self.batch_size})...")
        
        try:
            results = self.llm_pipe(
                all_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                batch_size=self.batch_size,
            )
        except Exception as e:
            print(f"❌ Erro no batching: {e}")
            return
        
        # 4. Organiza resultados
        comparative_descriptors = {cls: [] for cls in classes}
        
        for i, result in enumerate(tqdm(results, desc="Processando")):
            target_raw, target_clean = prompt_map[i]
            text = result[0]["generated_text"]
            
            # Extrai features
            features = extract_features(text, target_clean)
            comparative_descriptors[target_raw].append(features)
        
        # 5. Salva
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparative_descriptors, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Salvo: {output_path}")
        print(f"   📊 {len(comparative_descriptors)} classes processadas")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR = "descriptors_comparative_dclip"
    
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    
    if not datasets:
        print("❌ Nenhum dataset")
        return
    
    print(f"\n{'='*70}")
    print(f"🚀 DCLIP COMPARATIVO (MÉTODO CORRETO DO PAPER)")
    print(f"{'='*70}")
    print(f"✅ CLIP similarity para encontrar classes similares")
    print(f"✅ Prompt exato do paper + few-shot (2/10)")
    print(f"✅ Batching para velocidade")
    print(f"   Similaridades/classe: {NUM_SIMILAR}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"{'='*70}\n")
    
    generator = DCLIPGenerator(
        llm_pipe,
        num_similar=NUM_SIMILAR,
        batch_size=BATCH_SIZE
    )
    
    for dataset_name, dataset_path in datasets.items():
        generator.process_dataset(dataset_name, dataset_path, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print(f"✅ CONCLUÍDO!")
    print(f"📁 {OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()