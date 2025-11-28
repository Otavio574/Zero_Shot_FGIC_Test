"""
Script para carregar e agregar resultados de acurácia zero-shot de diferentes
métodos CLIP em uma única matriz por modelo.

INCLUI DIAGNÓSTICO: Imprime o caminho da pasta de resultados para verificar a localização.
"""

import json
import pandas as pd
from pathlib import Path
import os
import sys

# ============================================================
# CONFIGURAÇÃO DE PASTAS E MODELOS
# ============================================================

# Tenta encontrar a base do projeto (três níveis acima deste script)
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback para execução em ambientes onde __file__ não está definido
    BASE_DIR = Path.cwd().parent.parent.parent

# Diretorio raiz onde todos os resultados de acurácia estão salvos
ALL_RESULTS_DIR = BASE_DIR / "all_zero-shot_results"
OUTPUT_DIR = BASE_DIR / "accuracy_matrix_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Lista de todos os modelos CLIP a serem considerados
ALL_CLIP_MODELS = [
    'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
    'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]


# Definição das pastas de resultados e dos prefixos de arquivo para cada método
METHODS_CONFIG = {
    "clip_baseline": ("results_clip_baseline", "clip_baseline_results"),
    "clip_description": ("results_description_clip", "description_clip_results"),
    "clip_comparative": ("results_comparative_clip", "comparative_clip_results"),
    "clip_comparative_filtering": ("results_comparative_clip_filtered", "comparative_clip_filtered_results"),
    "clip_waffle": ("results_waffle_clip", "waffle_clip_results"),
}


# ============================================================
# FUNÇÃO PRINCIPAL POR MODELO
# ============================================================

def generate_matrix_for_model(model_name: str):
    """
    Carrega, processa e salva a matriz de acurácia para um modelo CLIP específico.
    """
    print("=" * 70)
    print(f"🔄 PROCESSANDO MATRIZ PARA O MODELO: {model_name}")
    print("=" * 70)
    
    model_safe_name = model_name.replace('/', '-')
    data = {}
    
    # 1. Carregar resultados de todos os métodos para o modelo atual
    for method_key, (folder_name, file_prefix) in METHODS_CONFIG.items():
        
        # O nome do arquivo JSON que ele está procurando:
        file_name = f"{file_prefix}_{model_safe_name}.json"
        file_path = ALL_RESULTS_DIR / folder_name / file_name

        if not file_path.exists():
            # AVISO de arquivo não encontrado (crucial para o nan%)
            # Destaca a falha do clip_comparative para fácil visualização
            if method_key == "clip_comparative":
                 print(f"🚨🚨 FALHA CRÍTICA ({method_key}): ARQUIVO AUSENTE em {file_path}")
            else:
                 print(f"⚠️ ARQUIVO AUSENTE ({method_key}): {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Erro ao decodificar JSON em {file_path}")
            continue

        entries = content 

        for dataset_name, values in entries.items():
            accuracy = (
                values.get("accuracy_top1")
                or values.get("accuracy")
                or values.get("acc")
                or values.get("top1")
                or None
            )

            if accuracy is not None:
                if dataset_name not in data:
                    data[dataset_name] = {}
                data[dataset_name][method_key] = accuracy
    
    # Se nenhum resultado foi carregado para este modelo, pula
    if not data:
        print(f"❌ Nenhum resultado válido encontrado para o modelo {model_name}. Pulando.")
        return

    # 2. CONVERTER PARA DATAFRAME
    df = pd.DataFrame.from_dict(data, orient="index")
    
    # 1. Renomeia o índice (dataset name) para um nome temporário, que será substituído
    df = df.reset_index().rename(columns={"index": "Nome do dataset (Temp)"})

    # 2. RENOMEIA a coluna dos nomes dos datasets para ser o próprio nome do modelo (conforme solicitação)
    # A coluna que era "Nome do dataset" agora terá o nome do modelo (ex: 'ViT-B/32')
    df = df.rename(columns={"Nome do dataset (Temp)": model_name})

    # Garantir que todas as colunas de métodos existam e definir a ordem correta
    # Colunas: [model_name (dataset column)] + [Métodos]
    columns_order = [model_name] + list(METHODS_CONFIG.keys())
    df = df.reindex(columns=columns_order, fill_value=None)
    
    # 3. FORMATAÇÃO EM PERCENTUAL (2 casas decimais)
    df_percent = df.copy()
    
    # A formatação em percentual agora começa no índice 1 (segunda coluna), 
    # pulando apenas o nome do modelo (que é o cabeçalho do dataset)
    for col in df_percent.columns[1:]:
        df_percent[col] = df_percent[col].apply(
            lambda x: f"{x * 100:6.2f}%" if isinstance(x, (float, int)) else 'nan%'
        )

    # 4. SALVAR CSV
    model_safe_name = model_name.replace('/', '-')
    output_path = OUTPUT_DIR / f"accuracy_matrix_{model_safe_name}.csv"
    df_percent.to_csv(output_path, index=False, sep=';')
    
    print("\n✅ Matriz de acurácia gerada com sucesso! (percentual, 2 casas decimais)")
    print(f"\n💾 Arquivo salvo em: {output_path}")
    print("-" * 70)
    print(df_percent.to_string()) # Usando to_string() para melhor formatação no console
    print("-" * 70)

    # NOVO DIAGNÓSTICO FINAL: Avisa se o comparativo falhou.
    if 'clip_comparative' in df_percent.columns:
        if all(df_percent['clip_comparative'].str.strip() == 'nan%'):
            print("\n❌ ATENÇÃO: A coluna 'clip_comparative' está toda 'nan%'.")
            print(">>> Motivo: O arquivo JSON correspondente está AUSENTE ou tem o nome incorreto.")
            print(">>> Procure pela mensagem '🚨🚨 FALHA CRÍTICA' APÓS 'PROCESSANDO MATRIZ' para ver o caminho EXATO que está faltando.")
            print(">>> Você precisa corrigir o script de avaliação que SALVA este arquivo JSON.")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    
    # DIAGNÓSTICO: Mostra o caminho base que o script está usando
    print(f"\nDIAGNÓSTICO: Base do projeto (BASE_DIR): {BASE_DIR}")
    print(f"DIAGNÓSTICO: Pasta de Resultados (ALL_RESULTS_DIR): {ALL_RESULTS_DIR}\n")
    
    if not ALL_RESULTS_DIR.exists():
        print(f"🚫 ERRO FATAL: Diretório de resultados principal não encontrado: {ALL_RESULTS_DIR}")
        print("Certifique-se de que a pasta 'all_zero-shot_results' está no caminho correto,")
        print("ou ajuste a variável BASE_DIR no código.")
        sys.exit(1)

    for model_name in ALL_CLIP_MODELS:
        generate_matrix_for_model(model_name)
    
    print("\n\n*** GERAÇÃO DE TODAS AS MATRIZES CONCLUÍDA ***")