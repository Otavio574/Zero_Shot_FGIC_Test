import subprocess
import sys
from pathlib import Path

# =================================================================
# 1. ORDEM DE EXECUÇÃO DA PIPELINE (COM CAMINHOS EXPLÍCITOS)
# =================================================================
# Define o caminho base do projeto (onde este main.py está localizado)
PROJECT_ROOT = Path(__file__).parent.parent

print(PROJECT_ROOT)

# Define a lista de caminhos completos dos scripts em relação à raiz do projeto.
SCRIPTS = [
    # 1. Geração: generate_datasets_config.py
    PROJECT_ROOT / "scripts" / "generation_scripts" / "generate_datasets_config.py",
    
    # 2. Análise (Assumindo que este está na raiz):
    PROJECT_ROOT / "analyze_dataset.py", 
    
    # 3. Geração: generate_embeddings.py
    PROJECT_ROOT / "scripts" / "generation_scripts" / "generate_embeddings.py",
    
    # 4. Geração: generate_descriptors_dclip.py
    PROJECT_ROOT / "scripts" / "generation_scripts" / "generate_descriptors_dclip.py",
    
    # 5. Geração (Comparisons):
    PROJECT_ROOT / "scripts" / "generation_scripts" / "generate_comparisons.py",
    
    # 6. Geração (Filtering):
    PROJECT_ROOT / "scripts" / "generation_scripts" / "generate_comparison_filtering.py",
    
    # 7. Avaliação (Baseline - Assumindo na raiz):
    PROJECT_ROOT / "scripts" / "evaluation_scripts" /"evaluate_clip_zero-shot.py",
    
    # 8. Avaliação (Description - Assumindo na raiz):
    PROJECT_ROOT / "scripts" / "evaluation_scripts" / "evaluate_clip_zero-shot_description.py",
    
    # 9. Avaliação (Comparative - Assumindo na raiz):
    PROJECT_ROOT / "scripts" / "evaluation_scripts" / "evaluate_clip_zero-shot_comparative.py",
    
    # 10. Avaliação (Comparative Filtering - Assumindo na raiz):
    PROJECT_ROOT / "scripts" / "evaluation_scripts" / "evaluate_clip_zero-shot_comparative_filtering.py",
    
    # 11. Avaliação (Waffle - Assumindo na raiz):
    PROJECT_ROOT / "scripts" / "evaluation_scripts" / "evaluate_clip_zero-shot_waffle.py",
    
    # 12. Finalização (Matrix - Assumindo na raiz):
    PROJECT_ROOT / "scripts" / "generation_scripts" / "generate_accuracy_matrix.py"
]

def run_pipeline():
    """Roda todos os scripts sequencialmente. Para a execução se um script falhar."""
    print("--- 🚀 Iniciando a Pipeline Completa de Avaliação CLIP ---")

    python_exec = sys.executable 

    for i, script_path in enumerate(SCRIPTS, 1):
        
        # O nome do script é apenas para exibição
        script_name = script_path.name
        
        print(f"\n[{i}/{len(SCRIPTS)}] Executando: {script_name} (Caminho: {script_path.relative_to(PROJECT_ROOT)})")
        
        # 1. Verificação de existência
        if not script_path.exists():
            print(f"❌ ERRO: Script não encontrado no caminho esperado: {script_path}")
            sys.exit(1)

        # 2. Comando de execução
        command = [python_exec, str(script_path)]
        
        try:
            # check=True garante que a execução pare se houver um erro no script
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8' 
            )
            
            output_snippet = result.stdout.strip()
            print(f"✔️ Concluído.")
            if output_snippet:
                last_lines = '\n'.join(output_snippet.splitlines()[-5:])
                print(f"   Últimas linhas de output:\n{last_lines}")
            
        except subprocess.CalledProcessError as e:
            # Captura o erro e interrompe o pipeline
            print(f"❌ ERRO FATAL no Passo {i}: {script_name}")
            print(f"Detalhes do Erro (stderr):\n{e.stderr}")
            print("\n🚨 Pipeline interrompida. Corrija o erro e reinicie.")
            sys.exit(1) 

    print("\n--- ✅ Pipeline Concluída com Sucesso! (Todos os 12 passos) ---")

if __name__ == "__main__":
    run_pipeline()