"""
Executa automaticamente toda a pipeline de análise e avaliação CLIP
na ordem correta, com logs e detecção de falhas.

Este script executa cada método de avaliação (os 5 scripts listados)
para CADA MODELO na lista ALL_MODELS, centralizando o controle do modelo aqui.
"""

import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path

# FIX para encoding no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# ==========================
# CONFIGURAÇÕES CENTRAIS
# ==========================

# 📢 NOVO: Lista centralizada de modelos. Para adicionar ou remover um modelo,
# edite SOMENTE esta lista.
ALL_MODELS = [
    'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 
    'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
]

# Lista dos 5 scripts de avaliação (métodos)
SCRIPTS = [
    "evaluate_clip_zero-shot.py",
    "evaluate_clip_zero-shot_description.py",
    "evaluate_clip_zero-shot_comparative.py",
    "evaluate_clip_zero-shot_comparative_filtering.py",
    "evaluate_clip_zero-shot_waffle.py",
    # Adicione a matriz de acurácia aqui se ela também precisar rodar
    # "scripts/generation_scripts/generate_accuracy_matrix.py", 
]

PYTHON_CMD = sys.executable  # garante que use o mesmo interpretador do ambiente virtual
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ==========================
# FUNÇÕES
# ==========================

def run_script(script_name, model_name=None):
    """
    Executa um script Python e registra o log.
    Recebe o nome do modelo e o passa como argumento.
    """
    
    # Prepara o comando
    command = [PYTHON_CMD, script_name]
    display_name = script_name
    log_file_base = f"{Path(script_name).stem}"
    
    if model_name:
        # Adiciona o nome do modelo ao comando
        command.append(model_name)
        display_name = f"{script_name} [MODELO: {model_name}]"
        # Adiciona o nome do modelo ao nome do arquivo de log
        log_file_base = f"{log_file_base}_{model_name.replace('/', '-')}"

    print(f"\n{'='*70}")
    print(f"[>] Iniciando: {display_name}")
    print(f"{'='*70}")

    log_file = LOGS_DIR / f"{log_file_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configura variáveis de ambiente para encoding UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    if not Path(script_name).exists():
        print(f"\n[X] ERRO: Script não encontrado em {script_name}")
        sys.exit(1)

    try:
        with open(log_file, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )

            # Printa em tempo real e salva no log
            for line in iter(process.stdout.readline, ""):
                if line:
                    print(line.rstrip())
                    log.write(line)
            
            process.stdout.close()
            return_code = process.wait()

        if return_code != 0:
            print(f"\n[X] Erro ao executar {script_name} com modelo {model_name if model_name else 'N/A'}")
            print(f"[!] Verifique o log em: {log_file}\n")
            sys.exit(return_code)

        print(f"[OK] Finalizado com sucesso: {display_name}")
        print(f"[i] Log salvo em: {log_file}")
        
    except Exception as e:
        print(f"\n[X] Excecao ao executar {script_name}: {e}")
        sys.exit(1)


def main():
    total_executions = len(SCRIPTS) * len(ALL_MODELS)
    
    print(f"\n{'='*70}")
    print(f"Pipeline de Avaliação com {len(ALL_MODELS)} Modelos ({total_executions} execuções)")
    print(f"{'='*70}")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logs: {LOGS_DIR.resolve()}\n")

    start_time = datetime.now()
    execution_count = 0

    # NOVO LOOP: Itera sobre modelos e scripts
    for model_name in ALL_MODELS:
        print(f"\n*** EXECUTANDO TODOS OS MÉTODOS PARA O MODELO: {model_name} ***")
        for script in SCRIPTS:
            execution_count += 1
            print(f"\n[{execution_count}/{total_executions}] Executando {script}...")
            # Passa o nome do modelo para a função run_script
            run_script(script, model_name=model_name)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*70}")
    print(f"Avaliação completa executada com sucesso!")
    print(f"{'='*70}")
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duracao: {duration}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()