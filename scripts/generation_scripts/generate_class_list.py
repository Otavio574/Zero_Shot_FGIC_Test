import json
import os

# 1. Carrega JSON existente
with open("descriptors_dclip_github/CUB_200_2011_dclip.json", "r", encoding="utf8") as f:
    data = json.load(f)

# 2. Carrega lista de classes do seu dataset
classes = []
for c in sorted(os.listdir("datasets/CUB_200_2011")):
    if os.path.isdir(os.path.join("datasets/CUB_200_2011", c)):
        classes.append(c)

# 3. Normaliza nomes (remove underscore, lower etc)
def normalize(s):
    return s.lower().replace("_", "").replace("-", "").replace(" ", "")

json_keys = list(data.keys())

# 4. Cria novo JSON alinhado
aligned = {}

for class_name in classes:
    n_class = normalize(class_name)
    match = None
    for key in json_keys:
        if normalize(key) == n_class:
            match = key
            break

    if match is None:
        print("❌ Sem correspondência para:", class_name)
    else:
        aligned[class_name] = data[match]

# 5. Salva o JSON corrigido
with open("descriptors_dclip_github/CUB_200_2011_dclip_aligned.json", "w", encoding="utf8") as f:
    json.dump(aligned, f, indent=4, ensure_ascii=False)

print("\n✅ JSON alinhado gerado com sucesso!")
