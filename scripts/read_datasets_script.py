import os

def list_datasets(base_path="datasets"):
    if not os.path.exists(base_path):
        print(f"❌ A pasta '{base_path}' não existe.")
        return

    datasets = [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ]

    if not datasets:
        print(f"⚠️ Nenhum dataset encontrado em '{base_path}'.")
    else:
        print(f"📁 Datasets encontrados em '{base_path}':\n")
        for ds in datasets:
            print(f" - {ds}")

if __name__ == "__main__":
    list_datasets()
