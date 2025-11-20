import os
from pathlib import Path

def clean_dataset_folders(dataset_root: str):
    root = Path(dataset_root)

    if not root.exists():
        print(f"❌ Caminho não encontrado: {root}")
        return

    for folder in root.iterdir():
        if folder.is_dir():
            name = folder.name

            # exemplo: 001.ak47 → ['001', 'ak47']
            if "." in name:
                prefix, rest = name.split(".", 1)

                # valida que o prefixo é numérico (ex: "001")
                if prefix.isdigit():
                    new_name = rest.strip()

                    old_path = folder
                    new_path = root / new_name

                    # evita sobrescrever caso já exista
                    if new_path.exists():
                        print(f"⚠️ Pasta já existe, pulando: {new_path}")
                        continue

                    print(f"🔄 Renomeando: {old_path.name} → {new_name}")
                    old_path.rename(new_path)

    print("\n✅ Finalizado! Todas as pastas foram corrigidas.")


# ============================
# USO
# ============================

if __name__ == "__main__":
    # coloque aqui o caminho da pasta do dataset
    clean_dataset_folders("datasets/CUB_200_2011")
