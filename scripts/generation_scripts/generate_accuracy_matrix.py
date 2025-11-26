import json
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO DE PASTAS
# ============================================================

# Base do projeto (três níveis acima deste script)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "all_zero-shot_results"

# Pastas e arquivos de resultados de cada modelo
MODELS = {
    "clip_baseline": RESULTS_DIR / "results_clip_baseline" / "clip_baseline_results.json",
    "clip_description": RESULTS_DIR / "results_description_clip" / "description_clip_results.json",
    "clip_comparative": RESULTS_DIR / "results_comparative_clip" / "comparative_clip_results.json",
    "clip_comparative_filtering": RESULTS_DIR / "results_comparative_clip_filtered" / "comparative_clip_filtered_results.json",
    "clip_waffle": RESULTS_DIR / "results_waffle_clip" / "waffle_clip_results.json",
}

# ============================================================
# CARREGAR RESULTADOS
# ============================================================

data = {}

for model_name, file_path in MODELS.items():
    if not file_path.exists():
        print(f"⚠️ Arquivo não encontrado: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # Suporta JSON com "results" ou diretamente
    entries = content.get("results", content)

    for dataset_name, values in entries.items():
        accuracy = (
            values.get("accuracy_top1")
            or values.get("accuracy")
            or values.get("acc")
            or values.get("top1")
            or None
        )

        if dataset_name not in data:
            data[dataset_name] = {}

        data[dataset_name][model_name] = accuracy

# Se nenhum arquivo foi encontrado
if not data:
    print("❌ Nenhum resultado encontrado. Verifique os arquivos JSON.")
    exit(1)

# ============================================================
# CONVERTER PARA DATAFRAME
# ============================================================

df = pd.DataFrame.from_dict(data, orient="index")
df = df.reset_index().rename(columns={"index": "Nome do dataset"})

# Garantir que todas as colunas existam
columns_order = ["Nome do dataset"] + list(MODELS.keys())
df = df.reindex(columns=columns_order, fill_value=None)

# ============================================================
# FORMATAÇÃO EM PERCENTUAL (2 casas decimais)
# ============================================================

df_percent = df.copy()
for col in df_percent.columns[1:]:
    df_percent[col] = df_percent[col].apply(
        lambda x: f"{x * 100:6.2f}%" if isinstance(x, (float, int)) else ""
    )

# ============================================================
# SALVAR CSV
# ============================================================

output_path = RESULTS_DIR / "accuracy_matrix.csv"
df_percent.to_csv(output_path, index=False)

print("\n✅ Matriz de acurácia gerada com sucesso! (percentual, 2 casas decimais)")
print(df_percent)
print(f"\n💾 Arquivo salvo em: {output_path}")
