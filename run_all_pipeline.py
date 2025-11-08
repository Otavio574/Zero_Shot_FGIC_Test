"""
Executa automaticamente toda a pipeline de análise e avaliação CLIP
na ordem correta, com logs e detecção de falhas.

Ordem dos scripts:
1. analyze_datasets.py
2. generate_embeddings.py
3. generate_descriptors.py
4. generate_comparisons.py
5. generate_waffle_descriptors.py
6. analyze_similarity.py
7. evaluate_clip_zero-shot.py (baseline)
8. evaluate_clip_zero-shot_description.py
9. evaluate_clip_zero-shot_comparative.py
10. evaluate_clip_zero-shot_comparative_filtering.py
11. evaluate_clip_zero-shot_waffle.py
12. generate_accuracy_matrix.py
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
# CONFIGURAÇÕES
# ==========================

"""
SCRIPTS = [
    "scripts/generation_scripts/generate_datasets_config.py",
    "analyze_dataset.py",
    "scripts/generation_scripts/generate_embeddings.py",
    "scripts/generation_scripts/generate_descriptors.py",
    "scripts/generation_scripts/generate_comparisons.py",
    "scripts/generation_scripts/generate_waffle_descriptors.py",
    "scripts/analyze_scripts/analyze_similarity.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_description.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_comparative.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_comparative_filtering.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_waffle.py",
    "scripts/generation_scripts/generate_accuracy_matrix.py",
]
"""


SCRIPTS = [
    "scripts/evaluation_scripts/evaluate_clip_zero-shot.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_description.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_comparative.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_comparative_filtering.py",
    "scripts/evaluation_scripts/evaluate_clip_zero-shot_waffle.py",
    "scripts/generation_scripts/generate_accuracy_matrix.py",
]

PYTHON_CMD = sys.executable  # garante que use o mesmo interpretador do ambiente virtual
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ==========================
# FUNÇÕES
# ==========================

def run_script(script_name):
    """Executa um script Python e registra o log"""
    print(f"\n{'='*70}")
    print(f"[>] Iniciando: {script_name}")
    print(f"{'='*70}")

    log_file = LOGS_DIR / f"{Path(script_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Configura variáveis de ambiente para encoding UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        with open(log_file, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [PYTHON_CMD, script_name],
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
            print(f"\n[X] Erro ao executar {script_name}")
            print(f"[!] Verifique o log em: {log_file}\n")
            sys.exit(return_code)

        print(f"[OK] Finalizado com sucesso: {script_name}")
        print(f"[i] Log salvo em: {log_file}")
        
    except Exception as e:
        print(f"\n[X] Excecao ao executar {script_name}: {e}")
        sys.exit(1)


def main():
    print(f"\n{'='*70}")
    print(f"Pipeline CLIP Completa ({len(SCRIPTS)} etapas)")
    print(f"{'='*70}")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logs: {LOGS_DIR.resolve()}\n")

    start_time = datetime.now()

    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[{i}/{len(SCRIPTS)}] Executando {script}...")
        run_script(script)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*70}")
    print(f"Pipeline completa executada com sucesso!")
    print(f"{'='*70}")
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duracao: {duration}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()