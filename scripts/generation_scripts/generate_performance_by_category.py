import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

# ============================================================================
# CATEGORIZAÇÃO DOS DATASETS
# ============================================================================

DATASET_CATEGORIES = {
    'Aves': [
        'Birdsnap',
        'CUB_200_2011',
        'Leeds_Butterfly',
        'MNIST_Butterfly'
    ],
    'Veículos': [
        'CompCars',
        'FGVC_Aircraft',
        'Stanford_Cars'
    ],
    'Animais': [
        'NA_Fish',
        'Oxford-IIIT_Pet_Dataset',
        'Stanford_Dogs'
    ],
    'Plantas': [
        'Flavia_Leaves',
        'Leaf',
        'Oxford_Flowers_192',
        'LZUPSD'  # sementes
    ],
    'Alimentos': [
        'food-101',
        'FoodX-251'
    ],
    'Objetos e Cenas': [
        'Caltech_256',
        'DTD',
        'MIT_Indoor_Scenes',
        'SlowSketch'
    ]
}

# ============================================================================
# FUNÇÃO PARA CARREGAR E AGREGAR RESULTADOS
# ============================================================================

def load_all_results(results_base_dir):
    """
    Carrega todos os resultados dos JSONs organizados em subpastas por método.
    Estrutura esperada:
        all_zero-shot_results/
            ├── results_clip_baseline/
            │   ├── clip_baseline_results_RN50.json
            │   └── clip_baseline_results_ViT-B-16.json
            ├── results_d_clip/
            │   └── d_clip_results_RN50.json
            └── ...
    
    Retorna DataFrame com: dataset, method, model, accuracy_top1
    """
    all_results = []
    results_base_dir = Path(results_base_dir)
    
    # Mapeia nomes de pastas para nomes de métodos consistentes
    folder_to_method = {
        'results_clip_baseline': 'vanilla_clip',
        'results_d_clip': 'd_clip',
        'results_comparative_clip': 'comparative_clip',
        'results_comparative_filtering': 'comparative_filtering',
        'results_waffle_clip': 'waffle_clip',
        # Adicione mais mapeamentos se necessário
    }
    
    print(f"Procurando em: {results_base_dir}")
    
    # Percorre todas as subpastas
    for method_folder in results_base_dir.iterdir():
        if not method_folder.is_dir():
            continue
        
        # Identifica o método pela pasta
        method_name = folder_to_method.get(method_folder.name, method_folder.name)
        print(f"\nProcessando pasta: {method_folder.name} -> método: {method_name}")
        
        # Percorre todos os JSONs dentro da pasta do método
        json_files = list(method_folder.glob('*.json'))
        print(f"  Encontrados {len(json_files)} arquivos JSON")
        
        for filepath in json_files:
            filename = filepath.name
            
            # Extrair modelo do nome do arquivo
            # Ex: "clip_baseline_results_RN50.json" -> "RN50"
            # Ex: "d_clip_results_ViT-B-16.json" -> "ViT-B-16"
            
            # Remove extensão e prefixos comuns
            model_part = filename.replace('.json', '')
            
            # Tenta extrair o modelo (última parte após underscore)
            if '_' in model_part:
                model = model_part.split('_')[-1]
            else:
                model = 'unknown'
            
            print(f"    {filename} -> modelo: {model}")
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Para cada dataset no arquivo
                for dataset_name, metrics in data.items():
                    all_results.append({
                        'dataset': dataset_name,
                        'method': method_name,
                        'model': model,
                        'accuracy_top1': metrics['accuracy_top1']
                    })
            except Exception as e:
                print(f"    ⚠️  Erro ao processar {filename}: {e}")
                continue
    
    print(f"\n✅ Total de resultados carregados: {len(all_results)}")
    return pd.DataFrame(all_results)


def assign_categories(df, category_dict):
    """
    Adiciona coluna de categoria ao DataFrame baseado no dicionário.
    """
    def get_category(dataset_name):
        for category, datasets in category_dict.items():
            if dataset_name in datasets:
                return category
        return 'Outros'  # Fallback para datasets não categorizados
    
    df['category'] = df['dataset'].apply(get_category)
    return df


# ============================================================================
# CRIAR GRÁFICO DE BARRAS AGRUPADAS
# ============================================================================

def create_performance_by_category_plot(df, output_path='performance_by_category.png'):
    """
    Cria gráfico de barras agrupadas mostrando performance por categoria.
    """
    # Agrupa por categoria e método (média entre datasets e modelos)
    category_method = df.groupby(['category', 'method'])['accuracy_top1'].agg(['mean', 'std']).reset_index()
    
    # Preparar dados para plotagem
    categories = sorted(category_method['category'].unique())
    methods = sorted(category_method['method'].unique())
    
    # Configurar cores para os métodos
    colors = {
        'vanilla_clip': '#8dd3c7',
        'results_description_clip': '#fb8072',
        'comparative_clip': '#80b1d3',
        'results_comparative_clip_filtered': '#bc80bd',
        'waffle_clip': '#b3de69'
    }
    
    # Mapear nomes legíveis
    method_labels = {
        'vanilla_clip': 'Baseline',
        'results_description_clip': 'D-CLIP',
        'comparative_clip': 'Comparative',
        'results_comparative_clip_filtered': 'Comp+Filter',
        'waffle_clip': 'Waffle'
    }
    
    print("Métodos encontrados no DF:", df['method'].unique())

    # Criar figura
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(categories))
    width = 0.15  # largura das barras
    
    # Plotar barras para cada método
    for i, method in enumerate(methods):
        method_data = category_method[category_method['method'] == method]
        
        means = []
        stds = []
        for cat in categories:
            cat_data = method_data[method_data['category'] == cat]
            if len(cat_data) > 0:
                means.append(cat_data['mean'].values[0])
                stds.append(cat_data['std'].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        offset = width * (i - len(methods)/2 + 0.5)
        bars = ax.bar(x + offset, means, width, 
                     label=method_labels.get(method, method),
                     color=colors.get(method, 'gray'),
                     yerr=stds, capsize=3, alpha=0.8)
    
    # Configurações do gráfico
    ax.set_xlabel('Categoria de Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acurácia Top-1 Média', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Performance por Categoria de Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    # Adicionar linha de referência em 0.5
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo em: {output_path}")
    
    return fig


# ============================================================================
# CRIAR TABELA DE ESTATÍSTICAS
# ============================================================================

def create_statistics_table(df, output_path='category_statistics.tex'):
    """
    Cria tabela LaTeX com estatísticas por categoria.
    """
    # Agrupa por categoria e método
    stats = df.groupby(['category', 'method'])['accuracy_top1'].agg(['mean', 'std', 'count']).reset_index()
    
    # Pivot para ter métodos como colunas
    pivot = stats.pivot(index='category', columns='method', values='mean')
    
    # Formata para LaTeX
    latex_lines = []
    latex_lines.append(r'\begin{table}[H]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Acurácia Top-1 média por categoria de dataset e método.}')
    latex_lines.append(r'\label{tab:category_performance}')
    latex_lines.append(r'\small')
    
    methods = pivot.columns.tolist()
    n_methods = len(methods)
    latex_lines.append(r'\begin{tabular}{|l|' + 'c|' * n_methods + '}')
    latex_lines.append(r'\hline')
    
    # Cabeçalho
    header = r'\textbf{Categoria}'
    method_labels = {
        'vanilla_clip': 'Baseline',
        'd_clip': 'D-CLIP',
        'comparative_clip': 'Comp.',
        'comparative_filtering': 'C+F',
        'waffle_clip': 'Waffle'
    }
    for method in methods:
        header += f' & \\textbf{{{method_labels.get(method, method)}}}'
    header += r' \\'
    latex_lines.append(header)
    latex_lines.append(r'\hline')
    
    # Linhas de dados
    for category in pivot.index:
        row = category
        for method in methods:
            value = pivot.loc[category, method]
            if pd.notna(value):
                row += f' & {value:.3f}'
            else:
                row += ' & ---'
        row += r' \\'
        latex_lines.append(row)
    
    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')
    
    latex_str = '\n'.join(latex_lines)
    
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✅ Tabela LaTeX salva em: {output_path}")
    return latex_str


# ============================================================================
# ANÁLISE DETALHADA
# ============================================================================

def print_category_analysis(df):
    """
    Imprime análise detalhada por categoria.
    """
    print("\n" + "="*80)
    print("ANÁLISE POR CATEGORIA")
    print("="*80)
    
    for category in sorted(df['category'].unique()):
        cat_data = df[df['category'] == category]
        
        print(f"\n📊 {category.upper()}")
        print("-" * 80)
        
        # Datasets nessa categoria
        datasets = cat_data['dataset'].unique()
        print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
        
        # Performance por método
        print("\nPerformance por método:")
        method_perf = cat_data.groupby('method')['accuracy_top1'].agg(['mean', 'std'])
        method_perf = method_perf.sort_values('mean', ascending=False)
        
        for method, row in method_perf.iterrows():
            print(f"  {method:25s}: {row['mean']:.3f} ± {row['std']:.3f}")
        
        # Melhor e pior método
        best_method = method_perf.index[0]
        worst_method = method_perf.index[-1]
        improvement = method_perf.loc[best_method, 'mean'] - method_perf.loc[worst_method, 'mean']
        
        print(f"\n  ✅ Melhor: {best_method} ({method_perf.loc[best_method, 'mean']:.3f})")
        print(f"  ❌ Pior: {worst_method} ({method_perf.loc[worst_method, 'mean']:.3f})")
        print(f"  📈 Ganho: {improvement:.3f} ({improvement*100:.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuração
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RESULTS_DIR = PROJECT_ROOT / "all_zero-shot_results"  # AJUSTE AQUI
    
    print("Carregando resultados...")
    df = load_all_results(RESULTS_DIR)
    
    print("Atribuindo categorias...")
    df = assign_categories(df, DATASET_CATEGORIES)
    
    print(f"\nTotal de registros: {len(df)}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Métodos: {df['method'].nunique()}")
    print(f"Modelos: {df['model'].nunique()}")
    print(f"Categorias: {df['category'].nunique()}")
    
    # Análise detalhada
    print_category_analysis(df)
    
    # Criar visualizações
    print("\n" + "="*80)
    print("GERANDO VISUALIZAÇÕES")
    print("="*80)
    
    create_performance_by_category_plot(df)
    create_statistics_table(df)
    
    print("\n✅ Análise completa!")
    print("\nArquivos gerados:")
    print("  - performance_by_category.png")
    print("  - category_statistics.tex")
    print(df.groupby('method')['accuracy_top1'].mean())