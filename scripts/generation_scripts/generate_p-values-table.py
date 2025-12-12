import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import json
import os

# ============================================================================
# PASSO 1: CARREGAR SEUS DADOS
# ============================================================================

def load_results_from_json(results_dir):
    """
    Carrega os resultados de todos os experimentos a partir dos arquivos JSON.
    
    Assume que você tem arquivos JSON no formato:
    {
        "Dataset1": {"accuracy_top1": 0.65, "method": "vanilla_clip", ...},
        "Dataset2": {"accuracy_top1": 0.72, "method": "vanilla_clip", ...},
        ...
    }
    
    Returns:
        DataFrame com colunas: [dataset, method, model, accuracy_top1]
    """
    all_results = []
    
    # Percorre todos os arquivos JSON no diretório
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(results_dir, filename)
        
        # Extrai informações do nome do arquivo (ajuste conforme sua nomenclatura)
        # Exemplo: "results_RN50_vanilla_clip.json"
        parts = filename.replace('.json', '').split('_')
        model = parts[1]  # RN50, ViT-B-16, etc.
        method = '_'.join(parts[2:])  # vanilla_clip, d_clip, etc.
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Para cada dataset no arquivo
        for dataset_name, metrics in data.items():
            all_results.append({
                'dataset': dataset_name,
                'method': method,
                'model': model,
                'accuracy_top1': metrics['accuracy_top1']
            })
    
    return pd.DataFrame(all_results)


# ============================================================================
# PASSO 2: CALCULAR P-VALUES COM TESTE DE WILCOXON
# ============================================================================

def compute_wilcoxon_matrix(df, methods):
    """
    Calcula matriz de p-values usando teste de Wilcoxon signed-rank.
    
    Args:
        df: DataFrame com colunas [dataset, method, model, accuracy_top1]
        methods: Lista de métodos a comparar
    
    Returns:
        DataFrame com matriz de p-values
    """
    n_methods = len(methods)
    pvalue_matrix = np.ones((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i >= j:  # Pula diagonal e metade inferior (simétrica)
                continue
            
            # Filtra resultados para cada método
            results1 = df[df['method'] == method1].groupby('dataset')['accuracy_top1'].mean()
            results2 = df[df['method'] == method2].groupby('dataset')['accuracy_top1'].mean()
            
            # Garante mesma ordem de datasets
            datasets = sorted(set(results1.index) & set(results2.index))
            acc1 = [results1[d] for d in datasets]
            acc2 = [results2[d] for d in datasets]
            
            # Aplica teste de Wilcoxon
            try:
                stat, pvalue = wilcoxon(acc1, acc2, alternative='two-sided')
                pvalue_matrix[i, j] = pvalue
                pvalue_matrix[j, i] = pvalue  # Matriz simétrica
            except ValueError as e:
                print(f"Erro ao comparar {method1} vs {method2}: {e}")
                pvalue_matrix[i, j] = 1.0
                pvalue_matrix[j, i] = 1.0
    
    # Cria DataFrame para melhor visualização
    pvalue_df = pd.DataFrame(
        pvalue_matrix,
        index=methods,
        columns=methods
    )
    
    return pvalue_df


# ============================================================================
# PASSO 3: GERAR TABELA LATEX
# ============================================================================

def generate_latex_table(pvalue_df, output_file='wilcoxon_table.tex', significance_level=0.05):
    """
    Gera tabela LaTeX formatada com p-values.
    Valores significativos (p < 0.05) aparecem em negrito.
    """
    methods = pvalue_df.index.tolist()
    n = len(methods)
    
    # Inicia código LaTeX
    latex_code = []
    latex_code.append(r'\begin{table}[H]')
    latex_code.append(r'\centering')
    latex_code.append(r'\caption{Valores de p-value do teste de Wilcoxon signed-rank entre pares de métodos. Valores em negrito indicam diferenças estatisticamente significativas ($p < 0.05$).}')
    latex_code.append(r'\label{tab:wilcoxon_pvalues}')
    
    # Define colunas da tabela
    col_format = '|l|' + 'c|' * n
    latex_code.append(r'\begin{tabular}{' + col_format + '}')
    latex_code.append(r'\hline')
    
    # Cabeçalho
    header = r'\textbf{Método}'
    for method in methods:
        # Formata nome do método (substitui underscores)
        method_formatted = method.replace('_', ' ').title()
        header += f' & \\textbf{{{method_formatted}}}'
    header += r' \\'
    latex_code.append(header)
    latex_code.append(r'\hline')
    
    # Linhas com p-values
    for i, method1 in enumerate(methods):
        method1_formatted = method1.replace('_', ' ').title()
        row = f'{method1_formatted}'
        
        for j, method2 in enumerate(methods):
            if i == j:
                row += ' & ---'
            else:
                pval = pvalue_df.iloc[i, j]
                
                # Formata p-value
                if pval < 0.001:
                    pval_str = '< 0.001'
                else:
                    pval_str = f'{pval:.3f}'
                
                # Negrito se significativo
                if pval < significance_level and i != j:
                    row += f' & \\textbf{{{pval_str}}}'
                else:
                    row += f' & {pval_str}'
        
        row += r' \\'
        latex_code.append(row)
    
    latex_code.append(r'\hline')
    latex_code.append(r'\end{tabular}')
    latex_code.append(r'\end{table}')
    
    # Salva em arquivo
    latex_str = '\n'.join(latex_code)
    with open(output_file, 'w') as f:
        f.write(latex_str)
    
    print(f"Tabela LaTeX salva em: {output_file}")
    return latex_str


# ============================================================================
# PASSO 4: EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # OPÇÃO 1: Se você tem seus dados em arquivos JSON
    # results_dir = "path/to/your/results"
    # df = load_results_from_json(results_dir)
    
    # OPÇÃO 2: Se você quer criar dados de exemplo para testar
    # Cria dados fictícios para demonstração
    np.random.seed(42)
    methods = ['clip_baseline', 'd_clip', 'comparative_clip', 'comparative_filtering', 'waffle_clip']
    datasets = [
        'Birdsnap', 'Caltech_256', 'CompCars', 'CUB_200_2011', 'DTD',
        'FGVC_Aircraft', 'Flavia_Leaves', 'food-101', 'FoodX-251', 'Leaf',
        'Leeds_Butterfly', 'LZUPSD', 'MIT_Indoor_Scenes', 'MNIST_Butterfly',
        'NA_Fish', 'Oxford-IIIT_Pet_Dataset', 'Oxford_Flowers_192',
        'SlowSketch', 'Stanford_Cars', 'Stanford_Dogs'
    ]
    models = ['RN50','RN50x4', 'RN50x16', 'RN50x64','RN101', 'ViT-B-16', 'ViT-B-32','ViT-L-14', 'ViT-L-14@336px']
    
    data = []
    for dataset in datasets:
        for model in models:
            # Baseline (vanilla_clip)
            base_acc = np.random.uniform(0.3, 0.6)
            
            for method in methods:
                # Métodos com descrições têm boost variável
                if method == 'clip_baseline':
                    acc = base_acc
                elif method in ['d_clip', 'comparative_clip', 'comparative_filtering']:
                    acc = base_acc + np.random.uniform(0.05, 0.15)
                else:  # waffle_clip
                    acc = base_acc + np.random.uniform(-0.02, 0.03)
                
                # Adiciona ruído
                acc = acc + np.random.normal(0, 0.02)
                acc = np.clip(acc, 0, 1)
                
                data.append({
                    'dataset': dataset,
                    'method': method,
                    'model': model,
                    'accuracy_top1': acc
                })
    
    df = pd.DataFrame(data)
    
    # Calcula matriz de p-values
    print("Calculando p-values com teste de Wilcoxon...")
    pvalue_df = compute_wilcoxon_matrix(df, methods)
    
    # Mostra resultado
    print("\nMatriz de P-values:")
    print(pvalue_df.round(3))
    
    # Gera tabela LaTeX
    latex_table = generate_latex_table(pvalue_df)
    print("\nCódigo LaTeX gerado:")
    print(latex_table)