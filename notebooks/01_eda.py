"""
LUCÍA-MVD | Notebook 01 — Exploración de Datos (EDA)
======================================================
Este script puede ejecutarse directamente o convertirse a Jupyter.

Analiza los datasets crudos y genera visualizaciones exploratorias.
"""

# %% [markdown]
# # LUCÍA-MVD | EDA Inicial
# ## Exploración de los datos de Violencia Doméstica, STM y Censo

# %%
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Configuración visual
plt.rcParams.update({
    "figure.facecolor": "#0D0F1A",
    "axes.facecolor":   "#151827",
    "axes.edgecolor":   "#252A42",
    "axes.labelcolor":  "#E8ECF8",
    "text.color":       "#E8ECF8",
    "xtick.color":      "#7C85A8",
    "ytick.color":      "#7C85A8",
    "grid.color":       "#252A42",
    "grid.linewidth":   0.5,
})

COLORS = ["#22C55E", "#F59E0B", "#EF4444", "#7C3AED"]

# %% [markdown]
# ## 1. Cargar datasets

# %%
SYN_DIR = BASE_DIR / "data" / "synthetic"

vda      = pd.read_parquet(SYN_DIR / "vda.parquet")
stm      = pd.read_parquet(SYN_DIR / "stm.parquet")
equip    = pd.read_parquet(SYN_DIR / "equipamiento.parquet")
censo    = pd.read_parquet(SYN_DIR / "censo.parquet")

print(f"VDA:         {len(vda):,} denuncias")
print(f"STM:         {len(stm):,} paradas")
print(f"Equipamiento:{len(equip):,} puntos")
print(f"Censo:       {len(censo):,} barrios")

# %% [markdown]
# ## 2. VDA — Distribución temporal y por tipo

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 2.1 Por tipo de delito
ax = axes[0, 0]
counts_tipo = vda["tipo_delito"].value_counts()
bars = ax.barh(counts_tipo.index, counts_tipo.values,
               color=["#6C63FF"] * len(counts_tipo))
ax.set_title("Denuncias por Tipo", fontweight="bold", color="#E8ECF8")
ax.set_xlabel("Cantidad")
for bar, v in zip(bars, counts_tipo.values):
    ax.text(v + 100, bar.get_y() + bar.get_height()/2,
            f"{v:,}", va="center", color="#7C85A8", fontsize=9)

# 2.2 Por sexo
ax = axes[0, 1]
counts_sexo = vda["sexo_victima"].value_counts()
wedge_colors = ["#FF6B9D", "#6C63FF"]
wedges, texts, autotexts = ax.pie(
    counts_sexo.values,
    labels=counts_sexo.index,
    autopct="%1.1f%%",
    colors=wedge_colors,
    startangle=90,
    wedgeprops={"edgecolor": "#0D0F1A", "linewidth": 2},
)
for at in autotexts:
    at.set_color("#E8ECF8")
    at.set_fontweight("bold")
ax.set_title("Víctimas por Sexo", fontweight="bold", color="#E8ECF8")

# 2.3 Por franja horaria
ax = axes[1, 0]
counts_franja = vda["franja_horaria"].value_counts().reindex([
    "Mañana (6-12h)", "Tarde (12-19h)", "Noche (19-24h)", "Madrugada (0-6h)"
])
bars = ax.bar(range(len(counts_franja)), counts_franja.values, color=COLORS)
ax.set_xticks(range(len(counts_franja)))
ax.set_xticklabels([c.split(" ")[0] for c in counts_franja.index], rotation=15)
ax.set_title("Distribución por Franja Horaria", fontweight="bold", color="#E8ECF8")
ax.set_ylabel("Denuncias")
for bar, v in zip(bars, counts_franja.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 50,
            f"{v:,}", ha="center", va="bottom", color="#7C85A8", fontsize=9)

# 2.4 Por día de semana
ax = axes[1, 1]
dow_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
counts_dow = vda["dia_semana"].value_counts().sort_index()
bar_colors = ["#EF4444" if d >= 4 else "#6C63FF" for d in counts_dow.index]
bars = ax.bar(dow_labels[:len(counts_dow)], counts_dow.values, color=bar_colors)
ax.set_title("Distribución por Día de Semana", fontweight="bold", color="#E8ECF8")
ax.set_ylabel("Denuncias")
ax.axhline(y=counts_dow.mean(), color="#F59E0B", linestyle="--", alpha=0.7,
           label=f"Media: {counts_dow.mean():.0f}")
ax.legend(facecolor="#1C2035", edgecolor="#252A42")

plt.tight_layout()
out_path = BASE_DIR / "outputs" / "maps" / "eda_vda.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D0F1A")
print(f"Guardado: {out_path}")
plt.show()

# %% [markdown]
# ## 3. Distribución geográfica de denuncias

# %%
# Top 15 barrios por densidad de denuncias (mujeres)
vda_mujeres = vda[vda["sexo_victima"] == "Mujer"]
top_barrios = vda_mujeres["barrio"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(
    range(len(top_barrios)),
    top_barrios.values,
    color=[f"#{int(255*(1-i/14)):02x}63FF" for i in range(len(top_barrios))],
)
ax.set_yticks(range(len(top_barrios)))
ax.set_yticklabels(top_barrios.index)
ax.set_title("Top 15 Barrios por Denuncias VDA (Mujeres víctimas) — 2023",
             fontweight="bold", color="#E8ECF8", fontsize=13)
ax.set_xlabel("Número de Denuncias")
for bar, v in zip(bars, top_barrios.values):
    ax.text(v + 10, bar.get_y() + bar.get_height()/2,
            f"{v:,}", va="center", color="#7C85A8", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(BASE_DIR / "outputs" / "maps" / "eda_barrios.png",
            dpi=150, bbox_inches="tight", facecolor="#0D0F1A")
plt.show()

# %% [markdown]
# ## 4. Análisis del dataset de features

# %%
PROC_DIR = BASE_DIR / "data" / "processed"
df = pd.read_parquet(PROC_DIR / "dataset_scored.parquet")

print(f"\nDataset scored: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print(f"\nDistribución de riesgo:")
print(df.nivel_riesgo.value_counts())
print(f"\nScore stats:")
print(df.score_riesgo.describe().round(2))

# %%
# Score por franja horaria
fig, ax = plt.subplots(figsize=(10, 5))
for franja, color in zip(
    ["Mañana (6-12h)", "Tarde (12-19h)", "Noche (19-24h)", "Madrugada (0-6h)"],
    ["#22C55E", "#F59E0B", "#EF4444", "#7C3AED"],
):
    data = df[df["franja_horaria"] == franja]["score_riesgo"]
    ax.hist(data, bins=30, alpha=0.65, color=color,
            label=franja.split(" ")[0], edgecolor="none")

ax.set_title("Distribución de Score de Riesgo por Franja Horaria",
             fontweight="bold", color="#E8ECF8", fontsize=12)
ax.set_xlabel("Score de Riesgo (0-100)")
ax.set_ylabel("Frecuencia (celdas)")
ax.legend(facecolor="#1C2035", edgecolor="#252A42")
ax.axvline(x=35, color="#F59E0B", linestyle="--", alpha=0.6, label="Umbral Medio")
ax.axvline(x=55, color="#EF4444", linestyle="--", alpha=0.6, label="Umbral Alto")
ax.axvline(x=75, color="#7C3AED", linestyle="--", alpha=0.6, label="Umbral Crítico")
plt.tight_layout()
plt.savefig(BASE_DIR / "outputs" / "maps" / "eda_scores.png",
            dpi=150, bbox_inches="tight", facecolor="#0D0F1A")
plt.show()

print("\n✓ EDA completado. Gráficos guardados en outputs/maps/")
