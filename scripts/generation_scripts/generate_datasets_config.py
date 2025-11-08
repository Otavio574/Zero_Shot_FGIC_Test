import json
from pathlib import Path

# Lista dos datasets
DATASETS = [
    "Bee_Images_Dataset",
    "Birdsnap",
    "CompCars",
    "CUB_200_2011",
    "FGVC_Aircraft",
    "Flavia_Leaves",
    "food-101",
    "FoodX-251",
    "Leaf",
    "Leeds_Butterfly",
    "LZUPSD",
    "MIT_Indoor_Scenes",
    "MNIST_Butterfly",
    "NA_Fish",
    "Oxford-IIIT_Pet_Dataset",
    "Oxford_Flowers_192",
    "PlantCLEF2017Test",
    "SlowSketch",
    "Stanford_Cars",
    "Stanford_Dogs"
]

# Caminho base dos datasets
BASE_PATH = "./datasets"

# Monta a estrutura de configuração
config = {
    "datasets": [
        {"name": name, "path": f"{BASE_PATH}/{name}"} for name in DATASETS
    ]
}

# Caminho de saída
output_file = Path("datasets_config.json")

# Salva o arquivo JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"✅ Arquivo '{output_file}' gerado com sucesso!")
