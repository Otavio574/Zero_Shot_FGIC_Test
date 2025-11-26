"""
Analisa e compara os resultados de diferentes variantes zero-shot CLIP.
Gera uma matriz de acurácia consolidada e um gráfico comparativo.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

# ========================
# CONFIGURAÇÕES
# ========================
RESULTS_DIR = "all_zero-shot_results/results_zero_shot_filtering"
OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ========================
# COLETA AUTOMÁTICA
# ========================

def load_all_results(results_dir):
    """Carrega todos os arquivos zero_shot_results_*.json"""
    all_results = {}

    for file in glob(os.path.join(results_dir, "zero_shot_results_*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            method = data.get("method", Path(file).stem.replace("zero_shot_results_", ""))
            model = data.get("model", "unknown")
            all_results[(model, method)] = data["results"]
        except Exception as e:
            print(f"❌ Erro ao carregar {file}: {e}")

    return all_results


def build_accuracy_matrix(all_results):
    """Constrói DataFrame com acurácia por método e dataset"""
    rows = []

    for (model, method), datasets in all_results.items():
        for dataset_name, res in datasets.items():
            acc = res.get("accuracy")
            rows.append({
                "Nome do dataset": dataset_name,
                "Modelo": model,
                "Método": method,
                "Acurácia": acc
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Nome do dataset", columns="Método", values="Acurácia")
    return pivot.reset_index()


def plot_accuracy_matrix(df, out_path):
    """Gera gráfico de barras comparativo"""
    plt.figure(figsize=(12, 6))
    methods = [c for c in df.columns if c != "Nome do dataset"]

    for i, method in enumerate(methods):
        plt.bar(
            df["Nome do dataset"],
            df[method],
            label=method,
            alpha=0.8,
        )

    plt.title("Comparação de Acurácia - Zero-Shot CLIP", fontsize=14)
    plt.xlabel("Dataset")
    plt.ylabel("Acurácia")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Método")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"📊 Gráfico salvo em: {out_path}")


# ========================
# EXECUÇÃO PRINCIPAL
# ========================

def main():
    print("🚀 Analisando resultados Zero-Shot CLIP...\n")

    all_results = load_all_results(RESULTS_DIR)
    if not all_results:
        print("❌ Nenhum resultado encontrado.")
        return

    df = build_accuracy_matrix(all_results)
    print("✅ Matriz de acurácia gerada com sucesso!\n")
    print(df.round(4))

    # salva CSV
    csv_path = os.path.join(OUT_DIR, "zero_shot_accuracy_matrix.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 Matriz salva em: {csv_path}")

    # salva gráfico
    plot_path = os.path.join(OUT_DIR, "zero_shot_accuracy_comparison.png")
    plot_accuracy_matrix(df, plot_path)

    print("\n✅ Análise concluída com sucesso!")


if __name__ == "__main__":
    main()
