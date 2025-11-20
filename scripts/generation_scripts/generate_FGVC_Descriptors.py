"""
SOTA-style DCLIP descriptor generator for aircraft datasets (e.g. FGVC_Aircraft)
using Qwen2-7B-Instruct.

- Especializado em AVIÕES
- Gera atributos curtos, visuais e discriminativos por classe
- Remove alucinações óbvias (engine count impossível, registration, logo, etc.)
- Saída pronta para ser usada como descritores DCLIP (lista de strings)

Formato de saída (JSON):

{
  "A320": [
    "twin underwing jet engines",
    "narrow fuselage with single aisle",
    "swept wings with small winglets",
    "pointed nose profile",
    "single tall vertical tail",
    "row of small rectangular windows"
  ],
  ...
}

Diretório de saída:
    descriptors_dclip_aircraft_qwen2_sota/{dataset_name}_aircraft_qwen2_sota.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
SUMMARY_PATH = Path("outputs/analysis/summary.json")
OUTPUT_DIR = Path("descriptors_dclip_aircraft_qwen2_sota")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4          # bom para RTX 4060 Ti 8GB
MIN_FEATURES = 6        # mínimo desejado por classe
MAX_FEATURES = 10       # máximo de atributos por classe


print(f"🚀 Loading model: {MODEL_NAME} (device={DEVICE})")

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

if pipe.tokenizer.pad_token is None:
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
if getattr(pipe.model.config, "pad_token_id", None) is None:
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id

print("✅ Model loaded.\n")


# ============================================================
# DATASET HELPERS
# ============================================================

def load_datasets_from_summary(summary_path: Path) -> Dict[str, str]:
    """
    Espera summary.json no formato:
    [
      {"dataset": "FGVC_Aircraft", "path": "datasets/FGVC_Aircraft/images"},
      ...
    ]
    """
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


def is_aircraft_dataset(dataset_name: str) -> bool:
    """
    Define quais datasets serão considerados "aviation".
    Aqui focamos em FGVC_Aircraft e nomes relacionados.
    """
    name = dataset_name.lower()
    if "aircraft" in name or "fgvc" in name or "plane" in name:
        return True
    # se quiser adicionar mais depois, só incluir aqui
    return False


def sanitize_class_name(class_name: str) -> str:
    """
    Converte nome da pasta da classe em conceito textual para o LLM.

    Ex:
      'A320' -> 'A320 airliner'
      'Boeing_747' -> 'Boeing 747 airliner'
      'Eurofighter_Typhoon' -> 'Eurofighter Typhoon fighter jet'
    """
    base = class_name
    # remove prefixo numérico tipo '001.A320' se existir
    parts = base.split(".", 1)
    if len(parts) == 2 and parts[0].isdigit():
        base = parts[1]

    # tokens separados por underline
    clean = base.replace("_", " ").strip()

    # heurística: se contém 'Boeing', 'Airbus', 'Embraer', etc, assume airliner
    lower = clean.lower()
    if any(k in lower for k in ["boeing", "airbus", "embraer", "bombardier", "fokker", "cessna", "gulfstream", "falcon", "global", "bae", "saab"]):
        return f"{clean} airliner"
    if any(k in lower for k in ["eurofighter", "typhoon", "f-16", "f 16", "f_a_18", "hawk", "spitfire", "tornado"]):
        return f"{clean} fighter jet"

    # fallback genérico
    return f"{clean} aircraft"


# ============================================================
# PROMPTS ESPECIALIZADOS PARA AVIÕES
# ============================================================

def build_aircraft_prompt(concept: str) -> str:
    """
    Prompt para Qwen focado em atributos visuais de AVIÕES.
    Evita engine count, livery, marca/companhia e comportamento.
    """
    return f"""
You are helping to build a fine-grained aircraft recognition system.

Task:
List short visual attributes that help distinguish the aircraft model "{concept}" in photos.

Rules:
1) Describe ONLY visible external geometry and appearance:
   - nose shape
   - fuselage length and thickness
   - wing position and shape
   - tail configuration (T-tail, low tail, single fin, etc.)
   - engine placement (underwing, tail-mounted, fuselage-mounted)
   - presence of propellers or jet engines
   - winglets or tips
   - landing gear position (nose gear, main gear under wings or fuselage)
2) Each attribute must be a short phrase of 2 to 8 words.
3) Do NOT mention airline logos, company names, text, registration numbers, flags or liveries.
4) Do NOT mention performance, speed, range, year, or country of origin.
5) Do NOT guess the exact number of engines or seats.
6) Do NOT mention background, sky, ground, airport, or environment.
7) Do NOT restate these instructions or talk about the list.

Output:
Return ONLY a list where each line begins with "- ".
No explanation. No extra text.

Begin now.
- 
"""


def build_aircraft_expand_prompt(concept: str, existing: List[str]) -> str:
    """
    Segunda passada: pedir atributos adicionais, diferentes dos já usados.
    """
    joined = "\n".join(f"- {a}" for a in existing[:8])
    return f"""
You are helping to build a fine-grained aircraft recognition system.

The aircraft model is "{concept}".

You already have the following visual attributes:
{joined}

Task:
Propose additional short visual attributes that are clearly different from the ones above.

Rules:
1) Attributes must describe ONLY visible external geometry and appearance.
2) Each attribute must be a short phrase of 2 to 8 words.
3) Do NOT repeat or slightly rephrase existing attributes.
4) Do NOT mention airline logos, company names, text, registration numbers, flags or liveries.
5) Do NOT mention performance, speed, range, year, or country of origin.
6) Do NOT guess the exact number of engines or seats.
7) Do NOT mention background, sky, ground, airport, or environment.
8) Do NOT restate these instructions.

Output:
Return ONLY the new attributes, as a list where each line begins with "- ".
No explanation. No comments.

Begin now.
- 
"""


# ============================================================
# EXTRAÇÃO E LIMPEZA DE FEATURES
# ============================================================

BANNED_SUBSTR = [
    "above rules", "following rules", "do not", "don't",
    "instruction", "guideline", "output format",
    "this list", "the list", "attributes above",
    "non-visual facts", "long sentences",
    "registration number", "registration code", "tail number",
    "company logo", "airline logo", "brand logo", "logo on",
    "airline name", "company name", "text on fuselage", "printed text",
]

AIRLINE_WORDS = [
    "airline", "company", "brand", "logo", "livery",
    "painted letters", "flag", "emblem", "insignia", "text", "writing",
]

NON_VISUAL_WORDS = [
    "speed", "range", "altitude", "performance", "economy", "noise",
    "passenger capacity", "seats", "crew", "cockpit instruments",
    "avionics", "engine power", "fuel", "efficiency",
]

GENERIC_TOO_SHORT = {
    "large aircraft", "small aircraft", "medium aircraft",
    "passenger plane", "passenger aircraft", "jet aircraft",
    "cargo plane", "airplane shape", "aircraft shape"
}


def extract_raw_features(text: str) -> List[str]:
    """
    Pega todas as linhas que começam com "- " e remove o prefixo.
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


def normalize_for_dedup(feat: str) -> str:
    """
    Normaliza string para deduplicação simples: lowercase, sem pontuação.
    """
    f = feat.lower()
    f = re.sub(r"[^a-z0-9\s]", " ", f)
    tokens = f.split()
    tokens = [t for t in tokens if t]
    tokens = sorted(tokens)
    return " ".join(tokens)


def is_bad_aircraft_feature(feat: str) -> bool:
    """
    Filtro moderado para features de aviões: remove óbvias besteiras.
    """
    f = feat.lower()

    # muito genérico
    if f in GENERIC_TOO_SHORT:
        return True

    # substrings proibidos (instruções / meta / registro / logo)
    if any(sub in f for sub in BANNED_SUBSTR):
        return True

    # não-visuais (performance, range, etc)
    if any(word in f for word in NON_VISUAL_WORDS):
        return True

    # airline / companhia / brand / texto
    if any(word in f for word in AIRLINE_WORDS):
        return True

    # tentar evitar engine-count explícito (deixa só engine placement genérico)
    if "engine" in f and (
        re.search(r"\b[0-9]+\b", f)
        or any(w in f for w in ["one engine", "two engines", "three engines", "four engines", "five engines", "six engines"])
    ):
        return True

    # descartar atributos que falam explicitamente de "design", "style", "configuration" sem nada visual
    if f in ["aerodynamic design", "modern design", "distinctive design", "unique style"]:
        return True

    return False


def clean_and_dedup_features(candidates: List[str]) -> List[str]:
    """
    Aplica filtros e deduplicação.
    """
    cleaned: List[str] = []
    seen_norm = set()

    for feat in candidates:
        feat = feat.strip()
        if not feat:
            continue

        # corta ponto final
        if feat.endswith("."):
            feat = feat[:-1].strip()
        if not feat:
            continue

        # tamanho
        tokens = feat.split()
        if len(tokens) < 2:
            continue
        if len(tokens) > 10:
            tokens = tokens[:10]
            feat = " ".join(tokens)

        if is_bad_aircraft_feature(feat):
            continue

        norm = normalize_for_dedup(feat)
        if not norm:
            continue
        if norm in seen_norm:
            continue
        seen_norm.add(norm)

        cleaned.append(feat)

    return cleaned


def select_diverse_subset(features: List[str], max_features: int) -> List[str]:
    """
    Politicamente simples: só corta a lista no máximo N, mantendo a ordem.
    (Se quiser algo mais sofisticado, dá pra classificar por tokens depois.)
    """
    return features[:max_features]


# ============================================================
# GERADOR ESPECIALIZADO PARA AIRCRAFT
# ============================================================

class AircraftDescriptorGenerator:

    def __init__(self, pipe, batch_size: int = 4):
        self.pipe = pipe
        self.batch_size = batch_size

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Chamada única ao modelo em batch.
        """
        results = self.pipe(
            prompts,
            max_new_tokens=120,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
            batch_size=self.batch_size,
        )
        return [r[0]["generated_text"] for r in results]

    def maybe_expand(self, concept: str, base_feats: List[str]) -> List[str]:
        """
        Segunda passada: tenta adicionar mais atributos diferentes dos já existentes.
        """
        expand_prompt = build_aircraft_expand_prompt(concept, base_feats)
        raw = self.generate_batch([expand_prompt])[0]
        extra = extract_raw_features(raw)
        if not extra:
            return base_feats
        merged = base_feats + extra
        merged = clean_and_dedup_features(merged)
        return merged

    def process_dataset(self, dataset_name: str, dataset_path: str, output_dir: Path):
        print(f"\n📘 Aircraft dataset: {dataset_name}")

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"⚠️ Dataset path not found, skipping: {dataset_path}")
            return

        # assumindo que cada subpasta é uma classe (FGVC_Aircraft-style)
        classes = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        if not classes:
            print(f"⚠️ No class folders found in: {dataset_path}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{dataset_name}_aircraft_qwen2_sota.json"

        # conceitos textuais (um por classe)
        concepts = [sanitize_class_name(c.name) for c in classes]

        print(f"🧩 Classes: {len(classes)}")
        print("📝 Building prompts...")
        prompts = [build_aircraft_prompt(concept) for concept in concepts]

        print(f"⚡ Running Qwen2-7B (batch={self.batch_size})...")
        raw_outputs = self.generate_batch(prompts)

        descriptors: Dict[str, List[str]] = {}

        print("🧹 Post-processing aircraft features (filters + expand + truncation)...")
        for i, raw in enumerate(tqdm(raw_outputs)):
            class_name = classes[i].name
            concept = concepts[i]

            # 1) extrai lista de atributos brutos
            candidates = extract_raw_features(raw)

            # 2) limpeza + dedup
            feats = clean_and_dedup_features(candidates)

            # 3) expand se vier muito pouco
            if len(feats) < MIN_FEATURES:
                feats = self.maybe_expand(concept, feats)
                feats = clean_and_dedup_features(feats)

            # 4) se ainda for pouco, fallback genérico leve
            if len(feats) < MIN_FEATURES:
                fallback = [
                    "overall fuselage shape",
                    "wing shape and sweep",
                    "tail fin configuration",
                    "engine placement on airframe",
                    "relative nose shape",
                ]
                feats = clean_and_dedup_features(feats + fallback)

            # 5) força máximo e subset diverso
            if len(feats) > MAX_FEATURES:
                feats = select_diverse_subset(feats, MAX_FEATURES)

            # 6) sanity final
            if not feats:
                feats = ["typical aircraft external shape"]

            # garante que SEMPRE será lista simples de strings
            feats = [str(f) for f in feats]

            descriptors[class_name] = feats

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(descriptors, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved aircraft descriptors to: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    datasets = load_datasets_from_summary(SUMMARY_PATH)
    if not datasets:
        print("❌ No datasets loaded from summary.json")
        return

    gen = AircraftDescriptorGenerator(pipe, batch_size=BATCH_SIZE)

    print("=" * 70)
    print("🚀 SOTA DCLIP aircraft descriptor generation (Qwen2-7B, aircraft-only)")
    print(f"📊 Total datasets in summary: {len(datasets)}")
    print(f"📁 Output dir: {OUTPUT_DIR}")
    print("=" * 70)

    for dataset_name, dataset_path in datasets.items():
        if not is_aircraft_dataset(dataset_name):
            print(f"⏭️ Skipping non-aircraft dataset: {dataset_name}")
            continue
        gen.process_dataset(dataset_name, dataset_path, OUTPUT_DIR)

    print("\n🎯 All done for aircraft datasets!")


if __name__ == "__main__":
    main()
