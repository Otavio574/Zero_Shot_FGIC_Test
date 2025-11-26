"""
SOTA-style DCLIP descriptor generator using Qwen2-7B-Instruct (Mode 1 verification).

- Generates short, visual-only, discriminative attributes per class.
- Post-processing: filtering, dedup, diversity, fallback, and light semantic checks.
- Output ready to be used as DCLIP text descriptors.
- Skips datasets that already have descriptors generated.

Output format (JSON):
{
  "001.Black_footed_Albatross": [
    "large seabird",
    "dark brown upper wings",
    "white underbelly",
    "long narrow wings",
    "hooked pale beak",
    "black webbed feet"
  ],
  ...
}
"""

import json
import re
import pathlib
from typing import Dict, List

import torch
torch.cuda.empty_cache()
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
SUMMARY_PATH = pathlib.Path("..") / ".." / "outputs" / "analysis" / "summary.json"
OUTPUT_DIR = pathlib.Path("..") / ".." / "descriptors_dclip"
PROJECT_ROOT = pathlib.Path("..") / ".."

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch for RTX 4060 Ti 8GB
BATCH_SIZE = 16

# Desired number of attributes per class
MIN_FEATURES = 6
MAX_FEATURES = 10


# ============================================================
# BASIC HELPERS
# ============================================================

def sanitize_class_name(class_name: str) -> str:
    """Ex: '001.Black_footed_Albatross' -> 'black footed albatross'."""
    parts = class_name.split(".", 1)
    if len(parts) == 2 and parts[0].isdigit():
        class_name = parts[1]
    return class_name.replace("_", " ").replace("-", " ").strip().lower()


def get_dataset_category(dataset_name: str) -> str:
    """
    Map datasets to high-level semantic categories (for prompts + light validation).
    Adapted to your list of datasets.
    """
    name = dataset_name.lower()

    # Birds
    if "cub" in name or "birdsnap" in name or "bird" in name:
        return "bird"
    if "butterfly" in name:
        return "butterfly"
    if "fish" in name:
        return "fish"

    # Aircraft
    if "aircraft" in name or "fgvc" in name or "plane" in name:
        return "aircraft"

    # Cars
    if "compcars" in name or "stanford_cars" in name or "car" in name:
        return "car"

    # Dogs / Pets
    if "stanford_dogs" in name or "dog" in name:
        return "dog"
    if "oxford-iiit_pet" in name or "pet" in name:
        return "pet"

    # Plants / Leaves / Flowers
    if "flavia" in name or "leaf" in name or "plantclef" in name or "plant" in name:
        return "plant"
    if "flower" in name or "oxford_flowers" in name:
        return "flower"

    # Food
    if "food-101" in name or "foodx" in name or "food" in name:
        return "food"

    # Scenes
    if "mit_indoor" in name or "indoor_scenes" in name:
        return "scene"

    # Sketch / shapes etc.
    if "slowsketch" in name or "lzupsd" in name or "mnist" in name:
        return "object"

    # Caltech_256, generic
    if "caltech_256" in name:
        return "object"

    # Fallback
    return "object"


def load_datasets_from_summary(summary_path: pathlib.Path) -> Dict[str, str]:
    if not summary_path.exists():
        print(f"❌ summary.json not found at: {summary_path}")
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    datasets = {}
    for d in data:
        if "dataset" in d and "path" in d:
            datasets[d["dataset"]] = d["path"]
    return datasets


# ============================================================
# PROMPTS
# ============================================================

def build_prompt(concept: str, category: str) -> str:
    """
    Prompt otimizado para Qwen2-7B.
    Força atributos visuais curtos e específicos.
    """
    return f"""
You are assisting in building a visual recognition system.

Task:
List short visual attributes that help distinguish "{concept}", which is a type of {category}, in a photo.

Rules:
1) Describe only visible physical traits: colors, textures, patterns, shapes, proportions, body parts, structural elements.
2) Each attribute should be a short phrase of 2 to 8 words.
3) Do not mention behavior, habitat, background, location, season, or actions.
4) Do not mention non-visual facts (diet, distribution, rarity, taxonomy, intelligence, personality).
5) Do not reference, paraphrase, or restate these rules.
6) Only return the list.

Output:
Return a list where each line begins with "- ".
No introduction. No explanation. No extra text.

Begin now.
- 
"""


def build_expand_prompt(concept: str, category: str, existing: List[str]) -> str:
    """
    Second-pass prompt: ask Qwen2 to add more attributes, different from existing ones.
    """
    joined = "\n".join(f"- {attr}" for attr in existing[:8])
    return f"""
You are assisting in building a visual recognition system.

The concept is "{concept}", a type of {category}.

You already have the following visual attributes:
{joined}

Task:
Propose additional short visual attributes that are clearly different from the ones above.

Rules:
1) Attributes must be directly visible in typical photos.
2) Each attribute should be a short phrase of 2 to 8 words.
3) Do not repeat or slightly rephrase existing attributes.
4) Do not mention behavior, habitat, background, location, season, or actions.
5) Do not mention non-visual facts (diet, distribution, rarity, taxonomy, intelligence, personality).
6) Only return the new attributes.

Output:
Return only a list where each line begins with "- ".
No explanation. No comments.

Begin now.
- 
"""


# ============================================================
# FEATURE PROCESSING (FILTER, DEDUP, CATEGORY)
# ============================================================

STOPWORDS = {
    "the", "a", "an", "of", "with", "and", "in", "on", "for", "to",
    "its", "their", "his", "her", "at", "by", "from", "or", "as",
    "typical", "common", "usually", "often"
}

COLOR_WORDS = {
    "black", "white", "brown", "grey", "gray", "red", "yellow", "orange",
    "blue", "green", "golden", "silver", "cream", "beige", "chestnut",
    "tan", "buff", "rufous", "rusty", "pink", "purple"
}

PATTERN_WORDS = {
    "striped", "spotted", "dotted", "banded", "barred", "streaked",
    "mottled", "speckled", "patched", "checked", "patterned", "ringed",
    "ring", "spot", "stripe", "band"
}

BODY_PART_WORDS = {
    "head", "beak", "bill", "eye", "eyes", "eyering", "eyebrow",
    "crown", "crest", "neck", "chest", "breast", "belly", "back",
    "tail", "wing", "wings", "wingtips", "leg", "legs", "feet",
    "foot", "claw", "talon", "ear", "ears", "snout", "nose", "muzzle"
}

SHAPE_WORDS = {
    "slender", "broad", "narrow", "long", "short", "rounded",
    "compact", "stocky", "tapered", "chunky", "elongated"
}

TEXTURE_WORDS = {
    "smooth", "rough", "glossy", "shiny", "matte", "fluffy",
    "shaggy", "fuzzy", "sleek", "plumage"
}

BANNED_EXACT = {
    "bird class-level visual trait",
    "overall shape and proportion",
    "typical colors and textures",
    "visual feature unavailable",
    "visual class-level attributes",
}

BANNED_SUBSTR = [
    "above rules", "following rules", "do not", "don't",
    "instruction", "guideline", "output format",
    "bullet list", "visual features above",
    "this list", "the list", "attributes above",
    "non-visual facts", "long sentences",
]

GENERIC_CLASS_WORDS = {
    "bird", "animal", "creature", "object", "thing", "species"
}

NON_VISUAL_WORDS = {
    "behavior", "behaviour", "diet", "habitat", "forest", "grassland",
    "desert", "city", "urban", "rural", "country", "continent",
    "sound", "song", "call", "voice", "noisy", "quiet",
    "smell", "taste", "speed", "fast", "slow", "intelligent",
    "friendly", "aggressive", "territorial", "migratory",
    "nocturnal", "diurnal"
}

AIRCRAFT_LIVERY_WORDS = {
    "logo", "brand", "airline", "livery", "flag", "text",
    "company", "lettering", "painted", "embossed"
}

CAR_NON_VISUAL = {
    "horsepower", "engine power", "top speed",
    "fuel", "diesel", "electric", "hybrid",
    "driver", "passenger", "interior", "cabin"
}

DOG_NON_VISUAL = {
    "loyal", "friendly", "aggressive", "playful",
    "guard", "working dog", "intelligent",
    "barks", "barking"
}


def extract_raw_features(text: str) -> List[str]:
    """
    Extract lines starting with '- ' and return cleaned phrases (without '-').
    """
    feats = []
    for line in text.split("\n"):
        l = line.strip()
        if not l.startswith("-"):
            continue
        content = l[1:].strip(" \t-")
        if content:
            feats.append(content)
    return feats


def is_too_generic(feat: str) -> bool:
    """
    Heuristic: mark features that are too generic to be useful.
    """
    f = feat.lower()

    if f in BANNED_EXACT:
        return True

    tokens = f.split()
    if len(tokens) <= 2 and any(t in GENERIC_CLASS_WORDS for t in tokens):
        return True

    if re.match(r"^(small|large|big|medium|tiny|little)\s+(bird|animal|creature|object|species)", f):
        return True

    return False


def normalize_for_dedup(feat: str) -> str:
    """
    Normalize feature string for deduplication.
    Lowercase, remove stopwords and punctuation, sort tokens.
    """
    f = feat.lower()
    f = re.sub(r"[^a-z0-9\s]", " ", f)
    tokens = [t for t in f.split() if t and t not in STOPWORDS]
    tokens = sorted(tokens)
    return " ".join(tokens)


def classify_feature(feat: str) -> str:
    """
    Classify feature into rough buckets to help ordering/diversity.
    Buckets: color, pattern, body, shape, texture, other.
    """
    f = feat.lower()
    tokens = set(f.split())

    if any(c in tokens for c in COLOR_WORDS):
        return "color"
    if any(p in tokens for p in PATTERN_WORDS):
        return "pattern"
    if any(b in tokens for b in BODY_PART_WORDS):
        return "body"
    if any(s in tokens for s in SHAPE_WORDS):
        return "shape"
    if any(t in tokens for t in TEXTURE_WORDS):
        return "texture"
    return "other"


# ============================================================
# CATEGORY-AWARE LIGHT VALIDATION (MODE 1)
# ============================================================

def passes_category_checks(feat: str, category: str, concept: str) -> bool:
    """
    Light, category-aware pruning of obviously wrong or non-visual attributes.
    Mode 1 = moderado: remove só o muito errado.
    """
    f = feat.lower()

    # 1) remover não-visuais genéricos
    if any(word in f for word in NON_VISUAL_WORDS):
        return False

    # 2) checks por categoria
    if category == "bird" or category == "butterfly":
        # Remove partes erradas de mamíferos
        mammal_words = ["fur", "mane", "hooves", "snout", "muzzle", "paws", "four-limbed primate"]
        if any(w in f for w in mammal_words):
            return False

        # "ear" é suspeito em pássaros (exceto ear tufts, mas modo 1 = simples)
        if "ear" in f and "tuft" not in f:
            return False

    if category == "aircraft":
        # Remover coisas de marca ou livery
        if any(w in f for w in AIRCRAFT_LIVERY_WORDS):
            return False
        # Muito suspeito: falar de "voice", "people" etc
        if "people" in f or "passenger" in f or "pilot" in f:
            return False

    if category == "car":
        if any(w in f for w in CAR_NON_VISUAL):
            return False
        if "driver" in f or "passenger" in f:
            return False

    if category == "dog" or category == "pet":
        if any(w in f for w in DOG_NON_VISUAL):
            return False

    if category in {"plant", "flower"}:
        # plantas não têm "fur", "feathers", "paws", etc.
        animal_words = ["fur", "feather", "feathers", "paw", "paws", "claws", "talons"]
        if any(w in f for w in animal_words):
            return False

    # Cena: evitar comportamento/habitat explícito (já coberto em NON_VISUAL_WORDS)

    # 3) evitar anunciar "concept" como palavra solta tipo "bird"
    # (apenas se é o único conteúdo)
    tokens = f.split()
    if len(tokens) <= 2 and any(t in GENERIC_CLASS_WORDS for t in tokens):
        return False

    return True


def filter_and_dedup_features(
    candidates: List[str],
    category: str,
    concept: str,
) -> List[str]:
    """
    Apply quality filters, category-aware checks and deduplicate.
    """
    cleaned = []
    seen_norm = set()

    for feat in candidates:
        feat = feat.strip().rstrip(".")
        if not feat:
            continue

        lower = feat.lower()

        # Remove meta / instruction substrings
        if any(sub in lower for sub in BANNED_SUBSTR):
            continue

        # Remove muito genérico
        if is_too_generic(feat):
            continue

        # Light category validation
        if not passes_category_checks(feat, category, concept):
            continue

        # Limit length
        tokens = feat.split()
        if len(tokens) < 2:
            continue
        if len(tokens) > 10:
            tokens = tokens[:10]
            feat = " ".join(tokens)

        # Dedup
        norm = normalize_for_dedup(feat)
        if not norm:
            continue
        if norm in seen_norm:
            continue
        seen_norm.add(norm)

        cleaned.append(feat)

    return cleaned


def select_diverse_features(features: List[str], max_features: int) -> List[str]:
    """
    Try to select a diverse subset of features with capped size.
    """
    buckets = {"color": [], "pattern": [], "body": [], "shape": [], "texture": [], "other": []}
    for feat in features:
        cat = classify_feature(feat)
        buckets.setdefault(cat, []).append(feat)

    ordered = []

    # Primeiro garantir variedade
    for name in ["color", "pattern", "body", "shape", "texture", "other"]:
        for feat in buckets.get(name, []):
            ordered.append(feat)

    return ordered[:max_features]


# ============================================================
# GENERATOR CLASS
# ============================================================

class Qwen2SOTADescriptorGenerator:

    def __init__(self, pipe, batch_size: int = 4):
        self.pipe = pipe
        self.batch_size = batch_size

    def generate_once(self, prompts: List[str]) -> List[str]:
        """
        Call Qwen2 once with batches of prompts.
        """
        batch_size = self.batch_size
        all_results = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch = prompts[i*batch_size : (i+1)*batch_size]

            # ✅ Aqui mostramos o progresso
            print(f"Processing batch {i+1}/{num_batches}", flush=True)

            results = self.pipe(
                batch,
                max_new_tokens=120,
                do_sample=False,
                batch_size=len(batch)  # passa o tamanho real do batch
            )

            all_results.extend([r[0]["generated_text"] for r in results])

        return all_results


    def maybe_expand(self, concept: str, category: str, base_feats: List[str]) -> List[str]:
        """
        Second pass: try to get additional attributes and merge.
        Only called if we have too few usable features.
        """
        expand_prompt = build_expand_prompt(concept, category, base_feats)
        raw = self.generate_once([expand_prompt])[0]
        new_candidates = extract_raw_features(raw)
        if not new_candidates:
            return base_feats
        merged = base_feats + new_candidates
        merged = filter_and_dedup_features(merged, category, concept)
        return merged

    def process_dataset(self, project_root: pathlib.Path, dataset_name: str, dataset_path: str, output_dir: pathlib.Path):
        print(f"\n📘 Dataset: {dataset_name}")

        category = get_dataset_category(dataset_name)
        dataset_path = project_root / pathlib.Path(dataset_path)

        if not dataset_path.exists():
            print(f"⚠️ Dataset path not found, skipping: {dataset_path}")
            return

        classes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        if not classes:
            print(f"⚠️ No class folders found in: {dataset_path}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{dataset_name}_dclip.json"

        concepts = [sanitize_class_name(c.name) for c in classes]

        print(f"🧩 Classes: {len(classes)}")
        print("📝 Building prompts...")
        prompts = [build_prompt(concept, category) for concept in concepts]

        print(f"⚡ Running Qwen2-7B (batch={self.batch_size})...")
        raw_outputs = self.generate_once(prompts)

        descriptors: Dict[str, List[str]] = {}

        print("🧹 Post-processing features (filters + verification + diversity + fallback)...")
        for i, raw in enumerate(tqdm(raw_outputs)):
            class_name = classes[i].name
            concept = concepts[i]

            # 1) extract raw bullet items
            candidates = extract_raw_features(raw)

            # 2) filter + dedup + light category checks
            feats = filter_and_dedup_features(candidates, category, concept)

            # 3) if too few, try expand (second pass)
            if len(feats) < MIN_FEATURES:
                feats = self.maybe_expand(concept, category, feats)
                feats = filter_and_dedup_features(feats, category, concept)

            # 4) if STILL too few, fallback generic (keep DCLIP structure)
            if len(feats) < MIN_FEATURES:
                fallback = [
                    f"{category} class-level visual trait",
                    "overall shape and proportion",
                    "typical colors and textures",
                ]
                feats = filter_and_dedup_features(feats + fallback, category, concept)

            # 5) enforce min/max & diversity
            if len(feats) > MAX_FEATURES:
                feats = select_diverse_features(feats, MAX_FEATURES)

            # final sanity: if still empty, add very generic
            if not feats:
                feats = [f"{category} visual trait"]

            descriptors[class_name] = feats

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(descriptors, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved descriptors to: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    if not datasets:
        print("❌ No datasets loaded from summary.json")
        return

    # ============================
    # VERIFICAÇÃO DE DATASETS JÁ PROCESSADOS
    # ============================
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets_to_process = {}
    datasets_skipped = []
    
    for dataset_name, dataset_path in datasets.items():
        output_file = OUTPUT_DIR / f"{dataset_name}_dclip.json"
        if output_file.exists():
            print(f"⏭️  Pulando {dataset_name} (descritores já existem)")
            datasets_skipped.append(dataset_name)
        else:
            datasets_to_process[dataset_name] = dataset_path
    
    if datasets_skipped:
        print(f"\n📋 Datasets pulados: {len(datasets_skipped)}")
        print(f"🔨 Datasets a processar: {len(datasets_to_process)}\n")
    
    if not datasets_to_process:
        print("✅ Todos os descritores já foram gerados!")
        return

    # ============================
    # CARREGAMENTO DO MODELO
    # ============================

    print(f"🚀 Loading model: {MODEL_NAME} (device={DEVICE})")

    # Explicit tokenizer/model so we can set padding_side='left'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # pad_token_id fix (important for batching)
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    if getattr(pipe.model.config, "pad_token_id", None) is None:
        pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id

    print("✅ Model loaded.\n")

    # ============================
    # PROCESSAMENTO DOS DATASETS
    # ============================

    gen = Qwen2SOTADescriptorGenerator(pipe, batch_size=BATCH_SIZE)

    print("=" * 70)
    print("🚀 SOTA DCLIP descriptor generation with Qwen2-7B-Instruct (Mode 1 verification)")
    print(f"📊 Total datasets: {len(datasets_to_process)}")
    print(f"📁 Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    for dataset_name, dataset_path in datasets_to_process.items():
        gen.process_dataset(PROJECT_ROOT, dataset_name, dataset_path, OUTPUT_DIR)

    print("\n🎯 All done!")


if __name__ == "__main__":
    main()