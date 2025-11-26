"""
Verifica a qualidade e diversidade das imagens do dataset
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_images(dataset_path, num_samples=50):
    """Analisa características das imagens"""
    
    print(f"\n{'='*70}")
    print(f"🖼️  ANÁLISE DE IMAGENS: {dataset_path}")
    print(f"{'='*70}\n")
    
    dataset_path = Path(dataset_path)
    
    # Encontra imagens
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_paths.extend(dataset_path.rglob(f'*{ext}'))
    
    if not image_paths:
        print("❌ Nenhuma imagem encontrada!")
        return
    
    print(f"📊 Total de imagens: {len(image_paths)}")
    
    # Amostra aleatória
    import random
    sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # Coleta estatísticas
    sizes = []
    modes = []
    mean_colors = []
    std_colors = []
    
    print(f"\n🔍 Analisando {len(sample_paths)} imagens...\n")
    
    for i, img_path in enumerate(sample_paths[:10]):  # Mostra primeiras 10
        try:
            img = Image.open(img_path)
            
            # Informações básicas
            size = img.size
            mode = img.mode
            
            sizes.append(size)
            modes.append(mode)
            
            # Converte para array
            img_array = np.array(img.convert('RGB'))
            
            # Estatísticas de cor
            mean_color = img_array.mean(axis=(0, 1))
            std_color = img_array.std(axis=(0, 1))
            
            mean_colors.append(mean_color)
            std_colors.append(std_color)
            
            # Mostra info
            print(f"   [{i+1:2d}] {img_path.name[:40]:40s} | "
                  f"Size: {size[0]:4d}x{size[1]:4d} | "
                  f"Mode: {mode:4s} | "
                  f"Mean: [{mean_color[0]:5.1f}, {mean_color[1]:5.1f}, {mean_color[2]:5.1f}]")
            
        except Exception as e:
            print(f"   ❌ Erro ao ler {img_path.name}: {e}")
    
    # Processa todas as amostras
    for img_path in sample_paths[10:]:
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
            modes.append(img.mode)
            
            img_array = np.array(img.convert('RGB'))
            mean_colors.append(img_array.mean(axis=(0, 1)))
            std_colors.append(img_array.std(axis=(0, 1)))
        except:
            pass
    
    # Análise estatística
    print(f"\n📊 ESTATÍSTICAS:")
    
    # Tamanhos
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    
    print(f"\n   Dimensões:")
    print(f"      Largura - Min: {min(widths)}, Max: {max(widths)}, Média: {np.mean(widths):.0f}")
    print(f"      Altura  - Min: {min(heights)}, Max: {max(heights)}, Média: {np.mean(heights):.0f}")
    
    # Modos
    mode_counts = Counter(modes)
    print(f"\n   Modos de cor:")
    for mode, count in mode_counts.most_common():
        print(f"      {mode}: {count}/{len(modes)} ({count/len(modes)*100:.1f}%)")
    
    # Cores médias
    mean_colors_array = np.array(mean_colors)
    std_colors_array = np.array(std_colors)
    
    print(f"\n   Cor média (RGB):")
    print(f"      R: {mean_colors_array[:, 0].mean():.1f} ± {mean_colors_array[:, 0].std():.1f}")
    print(f"      G: {mean_colors_array[:, 1].mean():.1f} ± {mean_colors_array[:, 1].std():.1f}")
    print(f"      B: {mean_colors_array[:, 2].mean():.1f} ± {mean_colors_array[:, 2].std():.1f}")
    
    # Verifica se imagens são muito similares (fundo branco, etc)
    avg_mean = mean_colors_array.mean()
    avg_std = std_colors_array.mean()
    
    print(f"\n   Diversidade:")
    print(f"      Intensidade média: {avg_mean:.1f}/255")
    print(f"      Desvio padrão médio: {avg_std:.1f}")
    
    if avg_mean > 200:
        print(f"      ⚠️  ATENÇÃO: Imagens muito CLARAS (fundo branco?)")
    elif avg_mean < 50:
        print(f"      ⚠️  ATENÇÃO: Imagens muito ESCURAS")
    else:
        print(f"      ✅ Intensidade normal")
    
    if avg_std < 30:
        print(f"      ⚠️  ATENÇÃO: Imagens muito UNIFORMES (pouca variação)")
    else:
        print(f"      ✅ Boa variação de cores")
    
    # Visualiza algumas imagens
    print(f"\n📸 Gerando visualização...")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Amostra de Imagens - {dataset_path.name}', fontsize=14)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(sample_paths[:10]):
            try:
                img = Image.open(sample_paths[idx])
                ax.imshow(img)
                ax.set_title(f"{sample_paths[idx].parent.name[:20]}", fontsize=8)
                ax.axis('off')
            except:
                ax.text(0.5, 0.5, 'Erro', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    output_path = f"image_analysis_{dataset_path.name}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Salvo em: {output_path}")


def check_specific_images(dataset_path):
    """Verifica imagens específicas de classes diferentes"""
    
    print(f"\n{'='*70}")
    print(f"🔬 TESTE: Comparar imagens de classes DIFERENTES")
    print(f"{'='*70}\n")
    
    dataset_path = Path(dataset_path)
    
    # Pega primeira imagem de cada uma das primeiras 3 classes
    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])[:3]
    
    sample_images = []
    
    for class_folder in class_folders:
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend(class_folder.glob(f'*{ext}'))
        
        if images:
            sample_images.append((class_folder.name, images[0]))
    
    print(f"Classes selecionadas:")
    for cls, img_path in sample_images:
        print(f"   - {cls}: {img_path.name}")
    
    # Carrega e mostra
    fig, axes = plt.subplots(1, len(sample_images), figsize=(15, 5))
    
    if len(sample_images) == 1:
        axes = [axes]
    
    for (cls, img_path), ax in zip(sample_images, axes):
        img = Image.open(img_path)
        img_array = np.array(img.convert('RGB'))
        
        ax.imshow(img)
        ax.set_title(f"{cls}\n{img.size[0]}x{img.size[1]}\nMean: {img_array.mean():.0f}", 
                     fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = "sample_classes_comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparação salva em: {output_path}")
    
    # Calcula similaridade visual básica
    print(f"\n📊 Similaridade visual (correlação de pixels):")
    
    if len(sample_images) >= 2:
        img1 = np.array(Image.open(sample_images[0][1]).convert('RGB').resize((224, 224)))
        img2 = np.array(Image.open(sample_images[1][1]).convert('RGB').resize((224, 224)))
        
        # Normaliza
        img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-8)
        img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-8)
        
        # Correlação
        correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        
        print(f"   Correlação entre {sample_images[0][0]} e {sample_images[1][0]}: {correlation:.4f}")
        
        if correlation > 0.7:
            print(f"   ⚠️  Imagens MUITO SIMILARES visualmente!")
        else:
            print(f"   ✅ Imagens suficientemente diferentes")


def main():
    print(f"\n{'#'*70}")
    print(f"# VERIFICAÇÃO DE QUALIDADE DAS IMAGENS")
    print(f"{'#'*70}\n")
    
    dataset_path = "datasets/CUB_200_2011"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return
    
    # Análise geral
    analyze_images(dataset_path, num_samples=100)
    
    # Comparação específica
    check_specific_images(dataset_path)
    
    print(f"\n{'#'*70}")
    print(f"# CONCLUSÃO")
    print(f"{'#'*70}\n")
    
    print("""
    Verifique os gráficos gerados:
    - image_analysis_CUB_200_2011.png
    - sample_classes_comparison.png
    
    Se as imagens tiverem:
    - Fundo branco uniforme
    - Poses muito similares
    - Recortes pequenos/centrados
    
    Isso explica por que os embeddings são tão similares (0.88)!
    
    CLIP funciona melhor com:
    - Imagens naturais/contextualizadas
    - Fundos variados
    - Diferentes poses e ângulos
    
    Se o CUB tiver imagens muito padronizadas (tipo fotos de pássaros
    em fundo branco), o CLIP terá dificuldade em diferenciar.
    """)


if __name__ == "__main__":
    main()