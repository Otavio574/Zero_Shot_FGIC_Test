# NO SEU SCRIPT DE AVALIAÇÃO:
# Adicione este bloco após a linha: image_embeds, text_embeds, labels, class_names = result

descriptor_coverage = 0
fallback_count = 0
for class_name in class_names:
    description = match_descriptor_to_class(class_name, descriptors)
    # Verifica se o descriptor é o fallback genérico
    if description.startswith("a photo of a "):
        fallback_count += 1
    else:
        descriptor_coverage += 1

print(f"📊 Descriptors: {descriptor_coverage} específicos encontrados, {fallback_count} usando fallback genérico.")