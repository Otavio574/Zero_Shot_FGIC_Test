import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from pathlib import Path

# ============================================================================
# PASSO 1: CARREGAR MODELO E FAZER PREDIÇÕES
# ============================================================================

def load_clip_model(model_name='RN50'):
    """Carrega modelo CLIP"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def predict_single_image(model, preprocess, device, image_path, class_names, method='vanilla'):
    """
    Faz predição para uma única imagem.
    
    Returns:
        predicted_class: classe predita
        true_class: classe verdadeira (do caminho)
        confidence: confiança da predição
        top5_classes: top-5 classes preditas
    """
    # Carrega e preprocessa imagem
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Cria prompts
    if method == 'vanilla':
        texts = [f"a photo of a {c}" for c in class_names]
    # Adicione outros métodos se quiser
    
    # Tokeniza textos
    text_tokens = clip.tokenize(texts).to(device)
    
    # Faz predição
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Normaliza features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calcula similaridade
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Pega top-5 predições
    values, indices = similarity[0].topk(5)
    
    predicted_idx = indices[0].item()
    predicted_class = class_names[predicted_idx]
    confidence = values[0].item()
    
    top5_classes = [(class_names[i.item()], values[j].item()) 
                    for j, i in enumerate(indices)]
    
    return predicted_class, confidence, top5_classes


def find_misclassified_images(model, preprocess, device, dataset_path, class_names, 
                               method='vanilla', num_errors=20):
    """
    Encontra imagens classificadas incorretamente.
    
    Returns:
        List of dicts com: image_path, true_class, predicted_class, confidence
    """
    errors = []
    dataset_path = Path(dataset_path)
    
    # Percorre cada classe
    for class_name in class_names:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
        
        # Pega algumas imagens dessa classe
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        for img_path in images[:10]:  # Limita para não processar tudo
            try:
                predicted, conf, top5 = predict_single_image(
                    model, preprocess, device, img_path, class_names, method
                )
                
                # Se errou
                if predicted != class_name:
                    errors.append({
                        'image_path': str(img_path),
                        'true_class': class_name,
                        'predicted_class': predicted,
                        'confidence': conf,
                        'top5': top5
                    })
                    
                    if len(errors) >= num_errors:
                        return errors
                        
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")
                continue
    
    return errors


# ============================================================================
# PASSO 2: VISUALIZAR ERROS
# ============================================================================

def create_error_grid(errors, output_path='error_analysis.png', grid_size=(4, 5)):
    """
    Cria grid visual de erros.
    
    Args:
        errors: lista de dicts com informações dos erros
        grid_size: (rows, cols) para o grid
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, error in enumerate(errors[:rows*cols]):
        ax = axes[idx]
        
        # Carrega imagem
        img = Image.open(error['image_path'])
        ax.imshow(img)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Título com informações
        true_class = error['true_class'].replace('_', ' ')
        pred_class = error['predicted_class'].replace('_', ' ')
        conf = error['confidence']
        
        title = f"True: {true_class}\n"
        title += f"Pred: {pred_class}\n"
        title += f"Conf: {conf:.2%}"
        
        ax.set_title(title, fontsize=10, pad=10)
        
        # Borda vermelha para indicar erro
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    
    # Remove axes extras
    for idx in range(len(errors), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Análise Qualitativa de Erros - Exemplos de Classificações Incorretas',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figura salva em: {output_path}")
    return fig


def create_confusion_examples(errors, output_path='confusion_examples.png'):
    """
    Cria visualização focada em pares de confusão comuns.
    Mostra exemplos lado a lado de classes frequentemente confundidas.
    """
    # Identifica pares mais confundidos
    confusion_pairs = {}
    for error in errors:
        pair = tuple(sorted([error['true_class'], error['predicted_class']]))
        if pair not in confusion_pairs:
            confusion_pairs[pair] = []
        confusion_pairs[pair].append(error)
    
    # Pega top 6 pares mais confundidos
    top_pairs = sorted(confusion_pairs.items(), 
                      key=lambda x: len(x[1]), 
                      reverse=True)[:6]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for idx, (pair, pair_errors) in enumerate(top_pairs):
        row = idx // 2
        col_start = (idx % 2) * 2
        
        # Mostra 2 exemplos desse par de confusão
        for i in range(min(2, len(pair_errors))):
            ax = axes[row, col_start + i]
            error = pair_errors[i]
            
            img = Image.open(error['image_path'])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            
            true_class = error['true_class'].replace('_', ' ')
            pred_class = error['predicted_class'].replace('_', ' ')
            
            title = f"True: {true_class}\nPredicted: {pred_class}"
            ax.set_title(title, fontsize=9)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
    
    plt.suptitle('Pares de Classes Frequentemente Confundidos', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figura de confusão salva em: {output_path}")
    return fig


# ============================================================================
# PASSO 3: ANÁLISE DE PADRÕES DE ERRO
# ============================================================================

def analyze_error_patterns(errors):
    """
    Analisa padrões nos erros e gera estatísticas.
    """
    print("\n" + "="*80)
    print("ANÁLISE DE PADRÕES DE ERRO")
    print("="*80)
    
    # 1. Pares mais confundidos
    confusion_pairs = {}
    for error in errors:
        pair = (error['true_class'], error['predicted_class'])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    print("\nTop 10 Pares Mais Confundidos:")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for (true_c, pred_c), count in sorted_pairs[:10]:
        print(f"  {true_c:30s} -> {pred_c:30s} : {count} vezes")
    
    # 2. Classes mais problemáticas (como true class)
    true_class_errors = {}
    for error in errors:
        tc = error['true_class']
        true_class_errors[tc] = true_class_errors.get(tc, 0) + 1
    
    print("\nClasses com Mais Erros (difíceis de classificar):")
    sorted_true = sorted(true_class_errors.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_true[:10]:
        print(f"  {class_name:40s} : {count} erros")
    
    # 3. Confiança média nas predições erradas
    avg_confidence = np.mean([e['confidence'] for e in errors])
    print(f"\nConfiança Média nas Predições Erradas: {avg_confidence:.2%}")
    print("  (Confiança alta em erros indica confusão real, não incerteza)")
    
    return {
        'confusion_pairs': confusion_pairs,
        'problematic_classes': true_class_errors,
        'avg_confidence': avg_confidence
    }


# ============================================================================
# PASSO 4: USO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Configuração
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / "datasets/CUB_200_2011/"  # AJUSTE AQUI
    MODEL_NAME = "RN50"  # ou "ViT-B/32", etc.
    METHOD = "vanilla"
    
    # Classes do dataset (carregue do seu arquivo ou liste aqui)
    # Para CUB-200-2011:
    class_names = sorted([d.name for d in Path(DATASET_PATH).iterdir() if d.is_dir()])
    print(f"Classes encontradas: {len(class_names)}")
    
    # 1. Carrega modelo
    print("Carregando modelo CLIP...")
    model, preprocess, device = load_clip_model(MODEL_NAME)
    
    # 2. Encontra erros
    print("Buscando exemplos de classificações incorretas...")
    errors = find_misclassified_images(
        model, preprocess, device, 
        DATASET_PATH, class_names, 
        method=METHOD, 
        num_errors=20
    )
    
    print(f"\nEncontrados {len(errors)} erros")
    
    if len(errors) == 0:
        print("Nenhum erro encontrado! Modelo está muito bom ou há problema no código.")
    else:
        # 3. Cria visualizações
        print("\nCriando visualizações...")
        create_error_grid(errors, 'error_analysis.png')
        create_confusion_examples(errors, 'confusion_examples.png')
        
        # 4. Analisa padrões
        patterns = analyze_error_patterns(errors)
        
        # 5. Salva resultados em JSON
        with open('error_analysis.json', 'w') as f:
            json.dump({
                'errors': errors,
                'patterns': {
                    'confusion_pairs': {f"{k[0]} -> {k[1]}": v 
                                       for k, v in patterns['confusion_pairs'].items()},
                    'problematic_classes': patterns['problematic_classes'],
                    'avg_confidence': patterns['avg_confidence']
                }
            }, f, indent=2)
        
        print("\n✅ Análise completa! Arquivos gerados:")
        print("  - error_analysis.png (grid de erros)")
        print("  - confusion_examples.png (pares confundidos)")
        print("  - error_analysis.json (dados completos)")