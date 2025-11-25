import json
import pandas as pd
from pathlib import Path

# Caminho onde estão todos os resultados
RESULTS_DIR = Path("all_zero-shot_results")

# Nome dos modelos/variantes (ajuste conforme necessário)
MODELS = {
    "clip_baseline": "results_clip_baseline/clip_baseline_results.json",
    "clip_description": "results_description_clip/description_clip_results.json",
    "clip_comparative": "results_comparative_clip/comparative_clip_results.json",
    "clip_comparative_filtering": "results_comparative_clip_filtered/comparative_clip_filtered_results.json",
    "clip_waffle": "results_waffle_clip/waffle_clip_results.json",
}

# Dicionário geral para acumular resultados
data = {}

# Loop em todos os modelos
for model_name, filename in MODELS.items():
    file_path = RESULTS_DIR / filename
    
    if not file_path.exists():
        print(f"⚠️ Arquivo não encontrado: {filename}")
        continue

    with open(file_path, "r") as f:
        content = json.load(f)

    # json pode ter estrutura direta ou ter "results"
    entries = content["results"] if "results" in content else content

    # Ler cada dataset
    for dataset_name, values in entries.items():

        # Detectar automaticamente a métrica correta
        accuracy = (
            values.get("accuracy_top1") or
            values.get("accuracy") or
            values.get("acc") or
            values.get("top1") or
            None
        )

        if dataset_name not in data:
            data[dataset_name] = {}

        data[dataset_name][model_name] = accuracy

# Converter para DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Adicionar coluna do nome do dataset
df = df.reset_index().rename(columns={"index": "Nome do dataset"})

# Ordenar colunas na ordem definida em MODELS
df = df[["Nome do dataset"] + list(MODELS.keys())]

# ----------- FORMATAÇÃO EM PERCENTUAL (sempre alinhado, 2 casas decimais) -----------
df_percent = df.copy()

for col in df_percent.columns[1:]:
    df_percent[col] = df_percent[col].apply(
        lambda x: f"{x * 100:6.2f}%" if isinstance(x, (float, int)) else ""
    )

# ------------------------------------------------------------------------------------

# Salvar CSV formatado
output_path = RESULTS_DIR / "accuracy_matrix.csv"
df_percent.to_csv(output_path, index=False)

print("\n✅ Matriz de acurácia gerada com sucesso! (percentual, 2 casas decimais)")
print(df_percent)
print(f"\n💾 Arquivo salvo em: {output_path}")
