import json
import pandas as pd
from pathlib import Path

# Caminho onde estão todos os resultados
RESULTS_DIR = Path("all_zero-shot_results")

# Nome dos modelos/variantes a comparar (ajuste conforme seus arquivos)
MODELS = {
    "clip_baseline": "results_zero_shot_baseline/zero_shot_results.json",
    "clip_description": "results_zero_shot_description/zero_shot_results_description.json",
    "clip_comparative": "results_zero_shot_comparative/zero_shot_results_comparative.json",
    "clip_comparative_filtering": "results_zero_shot_comparative_filtering/zero_shot_results_comparative_filtering.json",
    "clip_waffle": "results_zero_shot_waffle/zero_shot_results_waffle.json",
}

# Dicionário geral para acumular resultados
data = {}

# Loop em todos os modelos definidos acima
for model_name, filename in MODELS.items():
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        print(f"⚠️ Arquivo não encontrado: {filename}")
        continue

    with open(file_path, "r") as f:
        content = json.load(f)

    # Cada dataset e sua acurácia
    for dataset_name, values in content["results"].items():
        accuracy = values.get("accuracy", None)
        if dataset_name not in data:
            data[dataset_name] = {}
        data[dataset_name][model_name] = accuracy

# Converter para DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Ordenar colunas e adicionar nome de dataset como coluna
df = df.reset_index().rename(columns={"index": "Nome do dataset"})
df = df[
    ["Nome do dataset"]
    + [col for col in MODELS.keys() if col in df.columns]
]

# Salvar em CSV
output_path = RESULTS_DIR / "accuracy_matrix.csv"
df.to_csv(output_path, index=False, float_format="%.4f")

print("\n✅ Matriz de acurácia gerada com sucesso!")
print(df)
print(f"\n💾 Arquivo salvo em: {output_path}")
