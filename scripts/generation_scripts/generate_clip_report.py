import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# Caminhos
RESULTS_PATH = Path("results_zero_shot/zero_shot_results.json")
OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT_DIR / "clip_baseline_report.pdf"

# Carregar resultados
with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

results = data["results"]
model_name = data["model"]

# Criar gráfico de acurácia
datasets = list(results.keys())
accuracies = [v["accuracy"] for v in results.values()]

plt.figure(figsize=(10, 5))
plt.barh(datasets, accuracies, color="skyblue")
plt.xlabel("Acurácia")
plt.title("Acurácia por Dataset – CLIP Zero-Shot")
plt.tight_layout()
chart_path = OUTPUT_DIR / "clip_accuracy_chart.png"
plt.savefig(chart_path)
plt.close()

# Criar PDF
doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4)
styles = getSampleStyleSheet()
content = []

# Cabeçalho
content.append(Paragraph("<b>Relatório Técnico – Avaliação Zero-Shot com CLIP</b>", styles["Title"]))
content.append(Spacer(1, 12))
content.append(Paragraph(f"<b>Modelo Avaliado:</b> {model_name}", styles["Normal"]))
content.append(Paragraph(f"<b>Total de Datasets:</b> {data['total_datasets']}", styles["Normal"]))
content.append(Paragraph(f"<b>Datasets Avaliados com Sucesso:</b> {data['successful']}", styles["Normal"]))
content.append(Spacer(1, 12))

# Tabela de resultados
table_data = [["Dataset", "Acurácia", "Classes", "Imagens"]]
for name, vals in results.items():
    table_data.append([
        name,
        f"{vals['accuracy']*100:.2f}%",
        vals["num_classes"],
        vals["num_images"]
    ])

table = Table(table_data, repeatRows=1)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ('BOX', (0, 0), (-1, -1), 0.25, colors.black)
]))
content.append(table)
content.append(Spacer(1, 20))

# Inserir gráfico
content.append(Paragraph("<b>Gráfico de Acurácia</b>", styles["Heading2"]))
content.append(Spacer(1, 10))
content.append(RLImage(str(chart_path), width=450, height=250))
content.append(Spacer(1, 20))

# Insights automáticos
best = max(results.items(), key=lambda x: x[1]["accuracy"])
worst = min(results.items(), key=lambda x: x[1]["accuracy"])
content.append(Paragraph("<b>Insights Automáticos</b>", styles["Heading2"]))
content.append(Spacer(1, 6))
content.append(Paragraph(
    f"O modelo CLIP apresentou melhor desempenho em <b>{best[0]}</b> "
    f"(acurácia {best[1]['accuracy']*100:.2f}%) e pior desempenho em "
    f"<b>{worst[0]}</b> (acurácia {worst[1]['accuracy']*100:.2f}%).", styles["Normal"]
))
content.append(Spacer(1, 12))
content.append(Paragraph(
    "Observa-se que datasets com menor granularidade ou poucas classes tendem a "
    "obter desempenho mais alto, o que é consistente com limitações conhecidas do CLIP em tarefas fine-grained.",
    styles["Normal"]
))
content.append(Spacer(1, 20))

# Placeholder de conclusão
content.append(Paragraph("<b>Conclusão (Preencher Manualmente)</b>", styles["Heading2"]))
content.append(Spacer(1, 6))
content.append(Paragraph(
    "[Espaço reservado para discussão final, implicações dos resultados e próximos passos do projeto.]",
    styles["Italic"]
))

doc.build(content)
print(f"✅ Relatório gerado com sucesso: {PDF_PATH}")
