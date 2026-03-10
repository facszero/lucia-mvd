"""
LUCÍA-MVD | Dashboard Principal
================================
Localizador Urbano de Condiciones de Inseguridad y Alerta – Montevideo
ONU Mujeres DAT4CCIÓN 2026

Autor: Equipo LUCÍA-MVD
Contacto: dat4ccion@lucia-mvd.uy
"""

import math
import sys
import os
import warnings
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
PROC_DIR  = DATA_DIR / "processed"
SYN_DIR   = DATA_DIR / "synthetic"
MODEL_DIR = BASE_DIR / "outputs" / "models"

sys.path.insert(0, str(BASE_DIR / "src"))

# ── Paleta de colores LUCÍA ───────────────────────────────────────────────────
COLORS = {
    "bg":        "#0D0F1A",
    "surface":   "#151827",
    "card":      "#1C2035",
    "border":    "#252A42",
    "primary":   "#6C63FF",
    "accent":    "#FF6B9D",
    "text":      "#E8ECF8",
    "muted":     "#7C85A8",
    "bajo":      "#22C55E",
    "medio":     "#F59E0B",
    "alto":      "#EF4444",
    "critico":   "#7C3AED",
}

RIESGO_COLORS = {
    "Bajo":    "#22C55E",
    "Medio":   "#F59E0B",
    "Alto":    "#EF4444",
    "Crítico": "#7C3AED",
}

FRANJAS = ["Mañana (6-12h)", "Tarde (12-19h)", "Noche (19-24h)", "Madrugada (0-6h)"]
FRANJAS_ICONS = {
    "Mañana (6-12h)":    "🌅",
    "Tarde (12-19h)":    "☀️",
    "Noche (19-24h)":    "🌆",
    "Madrugada (0-6h)":  "🌙",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LUCÍA-MVD | Mapa de Riesgo Invisible",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0D0F1A;
    color: #E8ECF8;
  }

  .stApp { background-color: #0D0F1A; }

  /* Header hero */
  .hero-header {
    background: linear-gradient(135deg, #151827 0%, #1a1235 50%, #151827 100%);
    border: 1px solid #252A42;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF 0%, #FF6B9D 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
  }
  .hero-subtitle {
    color: #7C85A8;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 400;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #6C63FF;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  /* KPI Cards */
  .kpi-card {
    background: #1C2035;
    border: 1px solid #252A42;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: #6C63FF; }
  .kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
  }
  .kpi-label {
    color: #7C85A8;
    font-size: 0.82rem;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
  }

  /* Risk badge */
  .risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.03em;
  }

  /* Section header */
  .section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E8ECF8;
    padding-bottom: 10px;
    border-bottom: 1px solid #252A42;
    margin-bottom: 16px;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background-color: #151827;
    border-right: 1px solid #252A42;
  }
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #6C63FF;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
  }

  /* Metric overrides */
  [data-testid="metric-container"] {
    background: #1C2035;
    border: 1px solid #252A42;
    border-radius: 12px;
    padding: 16px;
  }
  [data-testid="metric-container"] label {
    color: #7C85A8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* Tables */
  .dataframe { font-size: 0.88rem; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #151827; }
  ::-webkit-scrollbar-thumb { background: #252A42; border-radius: 3px; }

  /* Alert boxes */
  .alert-critico {
    background: rgba(124,58,237,0.15);
    border-left: 4px solid #7C3AED;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
  }
  .alert-alto {
    background: rgba(239,68,68,0.12);
    border-left: 4px solid #EF4444;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #151827;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #252A42;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #7C85A8;
    font-weight: 500;
    padding: 8px 18px;
  }
  .stTabs [aria-selected="true"] {
    background: #6C63FF !important;
    color: white !important;
  }

  /* Simulation panel */
  .sim-card {
    background: linear-gradient(135deg, #1C2035, #1a1235);
    border: 1px solid #252A42;
    border-radius: 12px;
    padding: 20px;
  }
  .sim-result-up {
    color: #22C55E;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }
  .sim-result-down {
    color: #EF4444;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: #3D4465;
    font-size: 0.78rem;
    padding: 20px 0 10px;
    border-top: 1px solid #1C2035;
    margin-top: 40px;
  }
</style>
""", unsafe_allow_html=True)


# ── Data loading con cache ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_scored_data():
    path = PROC_DIR / "dataset_scored.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Pipeline completo automático si no existen los datos
    st.info("⚙️ Generando datos por primera vez... (~15 segundos)")
    from ingest.ingest import run_ingestion
    from features.features import run_feature_engineering
    from modeling.model import run_modeling, score_full_dataset
    run_ingestion()
    df = run_feature_engineering()
    results = run_modeling()
    return score_full_dataset(df, results["model_xgb"])


@st.cache_resource
def load_model():
    path = MODEL_DIR / "xgb_lucia.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data(ttl=3600)
def load_vda():
    path = SYN_DIR / "vda.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_feature_importance():
    path = MODEL_DIR / "feature_importance.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def get_risk_color_hex(nivel: str) -> str:
    return RIESGO_COLORS.get(nivel, "#7C85A8")


def score_to_color(score: float) -> str:
    """Convierte score 0-100 a color hex del gradiente."""
    if score >= 75:
        return "#7C3AED"
    elif score >= 55:
        return "#EF4444"
    elif score >= 35:
        return "#F59E0B"
    else:
        return "#22C55E"


def score_to_opacity(score: float) -> float:
    """
    Opacidad continua basada en score 0-100.
    Evita saltos visuales bruscos al cambiar de franja horaria.
    Usa curva sigmoide centrada en score=45: rango 0.06..0.82
    """
    normalized = (float(score) - 45.0) / 18.0
    sigmoid = 1.0 / (1.0 + math.exp(-normalized))
    return round(0.06 + sigmoid * 0.76, 3)


# ── Componentes de visualización ──────────────────────────────────────────────
def build_risk_map(df: pd.DataFrame, franja: str, niveles_filtro: list = None) -> folium.Map:
    """Construye mapa Folium con grilla de riesgo enmascarada al territorio de Montevideo."""
    df_f = df[df["franja_horaria"] == franja].copy()
    # Aplicar filtro de niveles del sidebar (todos si no se especifica)
    if niveles_filtro:
        df_f = df_f[df_f["nivel_riesgo"].isin(niveles_filtro)]

    # Centro y zoom calibrados para mostrar Montevideo completa
    m = folium.Map(
        location=[-34.872, -56.165],
        tiles="CartoDB.DarkMatter",
        prefer_canvas=True,
        min_zoom=11,
        max_zoom=16,
    )

    # fit_bounds define el encuadre inicial (controla zoom, zoom_start sería ignorado)
    m.fit_bounds([[-34.940, -56.300], [-34.820, -56.030]])

    # Opacidad continua basada en score (evita saltos visuales entre franjas horarias)
    # score_to_opacity usa sigmoide centrada en 45: score=10→0.09, 35→0.27, 55→0.62, 75→0.80
    for _, row in df_f.iterrows():
        color = score_to_color(row["score_riesgo"])
        opacity = score_to_opacity(row["score_riesgo"])
        cell_half = 0.003

        folium.Rectangle(
            bounds=[
                [row["lat_cen"] - cell_half, row["lon_cen"] - cell_half],
                [row["lat_cen"] + cell_half, row["lon_cen"] + cell_half],
            ],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=0,
            tooltip=folium.Tooltip(
                f"""
                <div style='font-family:sans-serif;font-size:13px;background:#1C2035;
                     color:#E8ECF8;padding:10px;border-radius:8px;min-width:200px;'>
                  <b style='color:{color};font-size:15px;'>⚠ {row['nivel_riesgo']}</b><br>
                  <b>Barrio:</b> {row.get('barrio','–')}<br>
                  <b>Score:</b> {row['score_riesgo']:.0f}/100<br>
                  <b>Franja:</b> {franja}<br>
                  <hr style='border-color:#252A42;margin:6px 0;'>
                  <b>Top factores:</b><br>
                  {('<br>').join(['• ' + f.split(' (')[0] for f in row.get('top_factores','').split(' | ')[:3]])}
                  <hr style='border-color:#252A42;margin:6px 0;'>
                  <b>Intervención:</b><br>
                  {('<br>').join(['• ' + r for r in row.get('recomendacion','').split(' | ')[:2]])}
                </div>
                """,
                sticky=True,
            ),
        ).add_to(m)

    # Zonas críticas con marcador especial
    df_critico = df_f[df_f["nivel_riesgo"] == "Crítico"].head(15)
    for _, row in df_critico.iterrows():
        folium.CircleMarker(
            location=[row["lat_cen"], row["lon_cen"]],
            radius=8,
            color="#7C3AED",
            fill=True,
            fill_color="#7C3AED",
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(
                f"""<div style='font-family:sans-serif;font-size:12px;min-width:180px;'>
                <b style='color:#7C3AED;'>🔴 ZONA CRÍTICA</b><br>
                {row.get('barrio','–')} | Score: {row['score_riesgo']:.0f}<br>
                {row.get('recomendacion','').split(' | ')[0]}
                </div>""",
                max_width=250,
            ),
        ).add_to(m)

    return m


def kpi_card(value, label, color=None, prefix="", suffix=""):
    color_style = f"color:{color};" if color else "color:#6C63FF;"
    st.markdown(
        f"""<div class='kpi-card'>
        <div class='kpi-value' style='{color_style}'>{prefix}{value}{suffix}</div>
        <div class='kpi-label'>{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def risk_distribution_chart(df_franja: pd.DataFrame) -> go.Figure:
    counts = df_franja["nivel_riesgo"].value_counts()
    order = ["Bajo", "Medio", "Alto", "Crítico"]
    vals = [counts.get(n, 0) for n in order]
    colors = [RIESGO_COLORS[n] for n in order]

    fig = go.Figure(go.Bar(
        x=order,
        y=vals,
        marker_color=colors,
        text=vals,
        textposition="outside",
        textfont=dict(color="#E8ECF8", size=13),
    ))
    fig.update_layout(
        plot_bgcolor="#151827",
        paper_bgcolor="#151827",
        font=dict(color="#E8ECF8", family="Space Grotesk"),
        margin=dict(l=10, r=10, t=20, b=30),
        height=220,
        xaxis=dict(showgrid=False, color="#7C85A8"),
        yaxis=dict(showgrid=True, gridcolor="#252A42", color="#7C85A8"),
        showlegend=False,
    )
    return fig


def temporal_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap barrio × franja horaria con score promedio."""
    top_barrios = (
        df.groupby("barrio")["score_riesgo"].mean()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )

    pivot = (
        df[df["barrio"].isin(top_barrios)]
        .groupby(["barrio", "franja_horaria"])["score_riesgo"]
        .mean()
        .unstack("franja_horaria")
        .reindex(columns=FRANJAS)
        .reindex(top_barrios)
        .round(1)
    )

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f.split(" ")[0] for f in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0.00, "#22C55E"],
            [0.35, "#F59E0B"],
            [0.65, "#EF4444"],
            [1.00, "#7C3AED"],
        ],
        text=pivot.values.round(0).astype(int),
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        showscale=True,
        colorbar=dict(
            title="Score",
            tickfont=dict(color="#7C85A8"),
            title_font=dict(color="#7C85A8"),
            bgcolor="#1C2035",
            bordercolor="#252A42",
        ),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="#151827",
        paper_bgcolor="#151827",
        font=dict(color="#E8ECF8", family="Space Grotesk"),
        margin=dict(l=150, r=20, t=20, b=60),
        height=500,
        xaxis=dict(side="bottom", color="#7C85A8"),
        yaxis=dict(color="#E8ECF8", tickfont=dict(size=11)),
    )
    return fig


def feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    fi_sorted = fi_df.sort_values("shap_importance")
    fig = go.Figure(go.Bar(
        x=fi_sorted["shap_importance"],
        y=fi_sorted["feature_es"],
        orientation="h",
        marker=dict(
            color=fi_sorted["shap_importance"],
            colorscale=[[0, "#252A42"], [0.5, "#6C63FF"], [1.0, "#FF6B9D"]],
        ),
        text=fi_sorted["shap_importance"].round(3),
        textposition="outside",
        textfont=dict(color="#7C85A8", size=11),
    ))
    fig.update_layout(
        plot_bgcolor="#151827",
        paper_bgcolor="#151827",
        font=dict(color="#E8ECF8", family="Space Grotesk"),
        margin=dict(l=10, r=60, t=10, b=10),
        height=380,
        xaxis=dict(showgrid=True, gridcolor="#252A42", color="#7C85A8", title="Importancia SHAP"),
        yaxis=dict(color="#E8ECF8", tickfont=dict(size=11)),
    )
    return fig


def trend_by_barrio_chart(df: pd.DataFrame, barrio: str) -> go.Figure:
    df_b = df[df["barrio"] == barrio].copy()
    scores = df_b.groupby("franja_horaria")["score_riesgo"].mean().reindex(FRANJAS)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f.split(" ")[0] for f in FRANJAS],
        y=scores.values,
        mode="lines+markers+text",
        line=dict(color="#6C63FF", width=3),
        marker=dict(
            size=12,
            color=[score_to_color(s) for s in scores.values],
            line=dict(color="#E8ECF8", width=2),
        ),
        text=[f"{s:.0f}" for s in scores.values],
        textposition="top center",
        textfont=dict(color="#E8ECF8", size=12),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
    ))
    fig.add_hline(y=55, line_dash="dot", line_color="#EF4444",
                  annotation_text="Umbral Alto", annotation_font_color="#EF4444")
    fig.add_hline(y=75, line_dash="dot", line_color="#7C3AED",
                  annotation_text="Umbral Crítico", annotation_font_color="#7C3AED")
    fig.update_layout(
        plot_bgcolor="#151827",
        paper_bgcolor="#151827",
        font=dict(color="#E8ECF8", family="Space Grotesk"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=240,
        xaxis=dict(showgrid=False, color="#7C85A8"),
        yaxis=dict(showgrid=True, gridcolor="#252A42", color="#7C85A8",
                   range=[0, 105], title="Score 0-100"),
        showlegend=False,
    )
    return fig


def simulation_chart(actual: dict, proyectado: dict) -> go.Figure:
    niveles = ["Bajo", "Medio", "Alto", "Crítico"]
    v_actual = [actual.get(n, 0) * 100 for n in niveles]
    v_proy   = [proyectado.get(n, 0) * 100 for n in niveles]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual",
        x=niveles,
        y=v_actual,
        marker_color="#EF4444",
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name="Con intervención",
        x=niveles,
        y=v_proy,
        marker_color="#22C55E",
        opacity=0.85,
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#151827",
        paper_bgcolor="#151827",
        font=dict(color="#E8ECF8", family="Space Grotesk"),
        margin=dict(l=10, r=10, t=10, b=30),
        height=250,
        xaxis=dict(showgrid=False, color="#7C85A8"),
        yaxis=dict(showgrid=True, gridcolor="#252A42", color="#7C85A8",
                   title="Probabilidad %"),
        legend=dict(
            bgcolor="#252A42",
            bordercolor="#3D4465",
            font=dict(color="#E8ECF8"),
        ),
    )
    return fig


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    # ── Cargar datos ──────────────────────────────────────────────────────────
    with st.spinner("Cargando datos..."):
        df       = load_scored_data()
        model    = load_model()
        vda_df   = load_vda()
        fi_df    = load_feature_importance()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-badge'>🛡️ ONU Mujeres · DAT4CCIÓN 2026</div>
        <h1 class='hero-title'>LUCÍA-MVD</h1>
        <p class='hero-subtitle'>
            <b>L</b>ocalizador <b>U</b>rbano de <b>C</b>ondiciones de <b>I</b>nseguridad y <b>A</b>lerta
            &nbsp;·&nbsp; Montevideo, Uruguay &nbsp;·&nbsp;
            Mapa de Riesgo Invisible para Mujeres
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🕐 Franja Horaria")
        franja = st.radio(
            "Seleccioná la franja",
            FRANJAS,
            format_func=lambda x: f"{FRANJAS_ICONS[x]}  {x}",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📍 Explorador de Barrio")
        barrios_disponibles = sorted(df["barrio"].unique())
        barrio_sel = st.selectbox(
            "Seleccionar barrio",
            barrios_disponibles,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🔍 Filtros de riesgo")
        niveles_sel = st.multiselect(
            "Mostrar niveles",
            ["Bajo", "Medio", "Alto", "Crítico"],
            default=["Alto", "Crítico"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Sobre LUCÍA-MVD")
        st.markdown("""
        <div style='color:#7C85A8;font-size:0.82rem;line-height:1.6;'>
        Sistema de estimación de riesgo urbano diferencial para mujeres.<br><br>
        <b style='color:#6C63FF;'>Datos:</b> DNPG/Ministerio del Interior, STM, INE, IMM<br>
        <b style='color:#6C63FF;'>Modelo:</b> XGBoost + SHAP (Accuracy: 93.4%)<br>
        <b style='color:#6C63FF;'>Período:</b> 2023 (calibrado con 43.245 denuncias)<br>
        <b style='color:#6C63FF;'>Celdas:</b> 608 (~560m × 560m, solo tierra)<br><br>
        <i>Los datos de denuncias son sintéticos calibrados con estadísticas oficiales reales del DNPG.</i>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs principales ──────────────────────────────────────────────────────
    tab_mapa, tab_barrios, tab_modelo, tab_simulador, tab_datos = st.tabs([
        "🗺️  Mapa de Riesgo",
        "📊  Análisis de Barrios",
        "🤖  Modelo IA",
        "⚡  Simulador",
        "📂  Datos y Metodología",
    ])

    # ── Tab 1: Mapa ────────────────────────────────────────────────────────────
    with tab_mapa:
        df_franja = df[df["franja_horaria"] == franja]

        # KPIs
        n_alto    = (df_franja["nivel_riesgo"] == "Alto").sum()
        n_critico = (df_franja["nivel_riesgo"] == "Crítico").sum()
        score_avg = df_franja["score_riesgo"].mean()
        pct_riesgo = (df_franja["nivel_riesgo"].isin(["Alto", "Crítico"])).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            kpi_card(f"{n_critico}", "Zonas Críticas", color="#7C3AED")
        with col2:
            kpi_card(f"{n_alto}", "Zonas Alto Riesgo", color="#EF4444")
        with col3:
            kpi_card(f"{score_avg:.1f}", "Score Promedio MVD", color="#F59E0B", suffix="/100")
        with col4:
            kpi_card(f"{pct_riesgo:.0f}%", "Celdas en Riesgo Elevado", color="#6C63FF")

        st.markdown("<br>", unsafe_allow_html=True)

        # Layout: mapa + panel lateral
        col_map, col_panel = st.columns([3, 1])

        with col_map:
            st.markdown(f"""<div class='section-title'>
            {FRANJAS_ICONS[franja]} Mapa de Riesgo — {franja}</div>""",
            unsafe_allow_html=True)
            mapa = build_risk_map(df, franja, niveles_filtro=niveles_sel)
            st_folium(mapa, width=None, height=520, returned_objects=[])

        with col_panel:
            st.markdown("<div class='section-title'>Distribución</div>",
                        unsafe_allow_html=True)
            fig_dist = risk_distribution_chart(df_franja)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div class='section-title' style='margin-top:16px;'>🔴 Top Zonas Críticas</div>",
                        unsafe_allow_html=True)
            top_criticas = (
                df_franja[df_franja["nivel_riesgo"].isin(["Crítico", "Alto"])]
                .sort_values("score_riesgo", ascending=False)
                .drop_duplicates("barrio")
                .head(8)[["barrio", "score_riesgo", "nivel_riesgo"]]
            )
            for _, row in top_criticas.iterrows():
                color = get_risk_color_hex(row["nivel_riesgo"])
                st.markdown(
                    f"""<div style='display:flex;justify-content:space-between;
                    align-items:center;padding:7px 10px;margin:4px 0;
                    background:#1C2035;border-radius:8px;border-left:3px solid {color};'>
                    <span style='color:#E8ECF8;font-size:0.88rem;'>{row['barrio']}</span>
                    <span style='color:{color};font-weight:700;font-family:monospace;
                    font-size:0.9rem;'>{row['score_riesgo']:.0f}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Leyenda
        st.markdown("""
        <div style='display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;
        background:#1C2035;padding:12px 18px;border-radius:10px;border:1px solid #252A42;'>
            <span style='font-size:0.82rem;color:#7C85A8;font-weight:600;text-transform:uppercase;
            letter-spacing:0.08em;align-self:center;'>Niveles:</span>
            <span style='background:rgba(34,197,94,0.15);color:#22C55E;padding:4px 12px;
            border-radius:6px;font-size:0.85rem;font-weight:600;'>⬤ Bajo (0–34)</span>
            <span style='background:rgba(245,158,11,0.15);color:#F59E0B;padding:4px 12px;
            border-radius:6px;font-size:0.85rem;font-weight:600;'>⬤ Medio (35–54)</span>
            <span style='background:rgba(239,68,68,0.15);color:#EF4444;padding:4px 12px;
            border-radius:6px;font-size:0.85rem;font-weight:600;'>⬤ Alto (55–74)</span>
            <span style='background:rgba(124,58,237,0.15);color:#7C3AED;padding:4px 12px;
            border-radius:6px;font-size:0.85rem;font-weight:600;'>⬤ Crítico (75–100)</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: Análisis de Barrios ─────────────────────────────────────────────
    with tab_barrios:
        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown(f"<div class='section-title'>📍 {barrio_sel}</div>",
                        unsafe_allow_html=True)

            df_barrio_franja = df[
                (df["barrio"] == barrio_sel) & (df["franja_horaria"] == franja)
            ]

            if len(df_barrio_franja) > 0:
                # Usar la celda de mayor score como representativa del barrio
                row = df_barrio_franja.loc[df_barrio_franja["score_riesgo"].idxmax()]
                score = row["score_riesgo"]
                nivel = row["nivel_riesgo"]
                color = get_risk_color_hex(nivel)

                st.markdown(f"""
                <div style='text-align:center;background:#1C2035;border-radius:12px;
                padding:24px;border:1px solid {color}44;margin-bottom:16px;'>
                    <div style='font-size:3.5rem;font-weight:700;color:{color};
                    font-family:monospace;'>{score:.0f}</div>
                    <div style='font-size:0.82rem;color:#7C85A8;text-transform:uppercase;
                    letter-spacing:0.1em;'>Score de Riesgo</div>
                    <div style='margin-top:10px;'>
                    <span class='risk-badge' style='background:{color}22;color:{color};
                    border:1px solid {color}55;'>⚠ {nivel}</span>
                    </div>
                    <div style='color:#7C85A8;font-size:0.8rem;margin-top:8px;'>{franja}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='section-title'>🔍 Factores de Riesgo</div>",
                            unsafe_allow_html=True)
                factores = row.get("top_factores", "").split(" | ")
                for f in factores[:3]:
                    partes = f.split(" (")
                    nombre = partes[0].strip()
                    valor = partes[1].rstrip(")") if len(partes) > 1 else "–"
                    try:
                        barra = int(float(valor) * 100)
                    except:
                        barra = 50
                    barra = min(barra, 100)
                    st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;
                    font-size:0.85rem;margin-bottom:4px;'>
                    <span style='color:#E8ECF8;'>{nombre}</span>
                    <span style='color:#6C63FF;font-family:monospace;'>{valor}</span>
                    </div>
                    <div style='background:#252A42;border-radius:4px;height:6px;'>
                    <div style='background:linear-gradient(90deg,#6C63FF,#FF6B9D);
                    width:{barra}%;height:6px;border-radius:4px;'></div>
                    </div></div>
                    """, unsafe_allow_html=True)

                st.markdown("<div class='section-title' style='margin-top:16px;'>💡 Intervención Sugerida</div>",
                            unsafe_allow_html=True)
                recomendaciones = row.get("recomendacion", "").split(" | ")
                for rec in recomendaciones[:3]:
                    st.markdown(
                        f"<div style='background:#1C2035;border-radius:8px;padding:9px 12px;"
                        f"margin:5px 0;color:#E8ECF8;font-size:0.88rem;border-left:3px solid #6C63FF;'>"
                        f"{rec}</div>",
                        unsafe_allow_html=True,
                    )

        with col_r:
            st.markdown("<div class='section-title'>📈 Evolución por Franja Horaria</div>",
                        unsafe_allow_html=True)
            fig_trend = trend_by_barrio_chart(df, barrio_sel)
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div class='section-title'>🌡️ Mapa de Calor: Top 20 Barrios × Franja</div>",
                        unsafe_allow_html=True)
            fig_heat = temporal_heatmap(df)
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 3: Modelo IA ───────────────────────────────────────────────────────
    with tab_modelo:
        st.markdown("<div class='section-title'>🤖 Arquitectura del Modelo LUCÍA</div>",
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("93.4%", "Accuracy XGBoost", color="#22C55E")
        with col2:
            kpi_card("93.4%", "F1-Score Weighted", color="#22C55E")
        with col3:
            kpi_card("90.1%", "Accuracy RF Baseline", color="#F59E0B")

        st.markdown("<br>", unsafe_allow_html=True)

        col_fi, col_exp = st.columns([2, 1])

        with col_fi:
            st.markdown("<div class='section-title'>📊 Importancia de Features (SHAP)</div>",
                        unsafe_allow_html=True)
            if not fi_df.empty:
                fig_fi = feature_importance_chart(fi_df)
                st.plotly_chart(fig_fi, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Ejecutar pipeline completo para ver importancia SHAP")

        with col_exp:
            st.markdown("<div class='section-title'>📖 ¿Cómo funciona?</div>",
                        unsafe_allow_html=True)
            st.markdown("""
            <div style='color:#7C85A8;font-size:0.87rem;line-height:1.7;'>

            <b style='color:#6C63FF;'>1. Score Compuesto (base)</b><br>
            Índice ponderado de 5 dimensiones que combina datos históricos, movilidad y entorno urbano.<br><br>

            <b style='color:#6C63FF;'>2. XGBoost (clasificador)</b><br>
            Clasifica cada celda en 4 niveles de riesgo usando 13 features. Entrenado sobre 2.432 observaciones celda×franja (608 celdas × 4 franjas).<br><br>

            <b style='color:#6C63FF;'>3. SHAP (explicabilidad)</b><br>
            Para cada celda se calcula la contribución de cada feature a la predicción. Esto permite identificar QUÉ intervención reducirá el riesgo.<br><br>

            <b style='color:#6C63FF;'>4. Ajuste temporal</b><br>
            Cada franja horaria aplica multiplicadores calibrados: Noche ×1.45, Madrugada ×1.30.<br><br>

            <b style='color:#FF6B9D;'>⚠ Importante:</b><br>
            El modelo estima riesgo potencial. No determina causalidad. Para toma de decisiones, se recomienda revisión humana.

            </div>
            """, unsafe_allow_html=True)

        # Ponderaciones del score
        st.markdown("<div class='section-title' style='margin-top:24px;'>⚖️ Ponderaciones del Score Compuesto</div>",
                    unsafe_allow_html=True)
        pesos_data = {
            "Componente": [
                "Violencia Observada (IVO)",
                "Aislamiento de Transporte (IAT)",
                "Vulnerabilidad Social (IVS)",
                "Aislamiento Espacial",
                "Déficit de Equipamiento",
            ],
            "Peso": ["35%", "25%", "20%", "12%", "8%"],
            "Descripción": [
                "Denuncias VDA y delitos sexuales georreferenciados, ajustados por franja",
                "Distancia a paradas STM, frecuencia nocturna, paradas activas de noche",
                "NBI, desempleo, informalidad, jefatura femenina de hogar",
                "Distancia al centro, conectividad, fragmentación territorial",
                "Luminarias, refugios VBG, comisarías en radio de influencia",
            ],
        }
        df_pesos = pd.DataFrame(pesos_data)
        st.dataframe(
            df_pesos,
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 4: Simulador ───────────────────────────────────────────────────────
    with tab_simulador:
        st.markdown("""
        <div class='section-title'>⚡ Simulador de Intervenciones</div>
        <p style='color:#7C85A8;font-size:0.9rem;margin-bottom:20px;'>
        Seleccioná una celda y simulá el impacto de intervenciones concretas sobre el score de riesgo.
        La reducción se calcula usando el modelo XGBoost entrenado.
        </p>
        """, unsafe_allow_html=True)

        if model is None:
            st.error("Modelo no disponible. Ejecutar pipeline completo.")
        else:
            col_config, col_result = st.columns([1, 1])

            with col_config:
                st.markdown("<div class='section-title'>🔧 Configurar Intervención</div>",
                            unsafe_allow_html=True)

                barrio_sim = st.selectbox(
                    "Barrio objetivo",
                    sorted(df["barrio"].unique()),
                    key="sim_barrio",
                )
                franja_sim = st.selectbox(
                    "Franja horaria",
                    FRANJAS,
                    format_func=lambda x: f"{FRANJAS_ICONS[x]} {x}",
                    key="sim_franja",
                )

                # Obtener celda representativa del barrio
                df_sim = df[
                    (df["barrio"] == barrio_sim) &
                    (df["franja_horaria"] == franja_sim)
                ].sort_values("score_riesgo", ascending=False)

                if len(df_sim) > 0:
                    celda_ref = df_sim.iloc[0].copy()
                    nivel_ref = celda_ref["nivel_riesgo"]
                    score_ref = celda_ref["score_riesgo"]

                    st.markdown(f"""
                    <div style='background:#1C2035;border-radius:8px;padding:12px 16px;
                    margin-bottom:16px;border:1px solid #252A42;'>
                    <span style='color:#7C85A8;font-size:0.82rem;'>Celda de referencia: </span>
                    <span style='color:#E8ECF8;font-weight:600;'>{barrio_sim}</span>
                    <span style='color:#7C85A8;'> | </span>
                    <span style='color:{get_risk_color_hex(nivel_ref)};font-weight:700;'>
                    {nivel_ref} ({score_ref:.0f}/100)</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Modificar condiciones:**")

                    nueva_frec = st.slider(
                        "🚌 Frecuencia STM nocturna (min)",
                        min_value=5, max_value=90,
                        value=int(celda_ref.get("frec_nocturna_avg_min", 30)),
                        step=5,
                    )
                    nuevas_luminarias = st.slider(
                        "💡 Luminarias en 200m",
                        min_value=0, max_value=20,
                        value=int(celda_ref.get("n_luminarias_200m", 2)),
                        step=1,
                    )
                    nueva_dist_refugio = st.slider(
                        "🏥 Distancia refugio VBG (km)",
                        min_value=0.1, max_value=5.0,
                        value=float(min(celda_ref.get("dist_refugio_km", 2.0), 5.0)),
                        step=0.1,
                    )
                    nueva_dist_pol = st.slider(
                        "🚓 Distancia comisaría (km)",
                        min_value=0.1, max_value=5.0,
                        value=float(min(celda_ref.get("dist_comisaria_km", 1.5), 5.0)),
                        step=0.1,
                    )

                    btn_simular = st.button("⚡ Simular intervención", type="primary",
                                            use_container_width=True)
                else:
                    btn_simular = False

            with col_result:
                st.markdown("<div class='section-title'>📈 Resultado de la Simulación</div>",
                            unsafe_allow_html=True)

                if len(df_sim) > 0 and btn_simular:
                    from modeling.model import simulate_intervention, FEATURE_COLS

                    intervenciones = {
                        "frec_nocturna_avg_min": nueva_frec,
                        "n_luminarias_200m":      nuevas_luminarias,
                        "dist_refugio_km":        nueva_dist_refugio,
                        "dist_comisaria_km":      nueva_dist_pol,
                    }

                    try:
                        resultado = simulate_intervention(celda_ref, intervenciones, model)

                        color_nuevo = get_risk_color_hex(resultado["nivel_proyectado"])
                        color_viejo = get_risk_color_hex(nivel_ref)
                        reduccion = resultado["reduccion_riesgo_pct"]

                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,#1C2035,#1a1235);
                        border-radius:12px;padding:24px;border:1px solid #252A42;
                        text-align:center;margin-bottom:16px;'>
                            <div style='font-size:3rem;font-weight:700;color:#22C55E;
                            font-family:monospace;'>↓ {reduccion:.0f}%</div>
                            <div style='color:#7C85A8;font-size:0.85rem;'>Reducción de riesgo proyectada</div>
                            <div style='display:flex;justify-content:center;gap:24px;margin-top:16px;'>
                                <div>
                                    <div style='color:{color_viejo};font-size:1.1rem;font-weight:700;'>
                                    {nivel_ref}</div>
                                    <div style='color:#7C85A8;font-size:0.78rem;'>Actual</div>
                                </div>
                                <div style='color:#7C85A8;font-size:1.5rem;align-self:center;'>→</div>
                                <div>
                                    <div style='color:{color_nuevo};font-size:1.1rem;font-weight:700;'>
                                    {resultado["nivel_proyectado"]}</div>
                                    <div style='color:#7C85A8;font-size:0.78rem;'>Con intervención</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        fig_sim = simulation_chart(
                            resultado["prob_actual"],
                            resultado["prob_proyectado"],
                        )
                        st.plotly_chart(fig_sim, use_container_width=True,
                                        config={"displayModeBar": False})

                        st.markdown("**Cambios aplicados:**")
                        cambios = {
                            "🚌 Frecuencia STM nocturna": f"{celda_ref.get('frec_nocturna_avg_min',30):.0f} → {nueva_frec} min",
                            "💡 Luminarias": f"{celda_ref.get('n_luminarias_200m',2):.0f} → {nuevas_luminarias}",
                            "🏥 Dist. refugio": f"{celda_ref.get('dist_refugio_km',2):.1f} → {nueva_dist_refugio:.1f} km",
                            "🚓 Dist. comisaría": f"{celda_ref.get('dist_comisaria_km',1.5):.1f} → {nueva_dist_pol:.1f} km",
                        }
                        for k, v in cambios.items():
                            st.markdown(
                                f"<div style='font-size:0.85rem;color:#7C85A8;padding:4px 0;'>"
                                f"<b style='color:#E8ECF8;'>{k}:</b> {v}</div>",
                                unsafe_allow_html=True,
                            )
                    except Exception as e:
                        st.error(f"Error en simulación: {e}")

                elif len(df_sim) > 0:
                    # Estado inicial sin simular
                    st.markdown("""
                    <div style='text-align:center;padding:60px 20px;color:#7C85A8;'>
                        <div style='font-size:3rem;'>⚡</div>
                        <div style='margin-top:12px;font-size:0.95rem;'>
                        Ajustá los parámetros de intervención y hacé click en<br>
                        <b style='color:#6C63FF;'>Simular intervención</b> para ver el impacto
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Tab 5: Datos ───────────────────────────────────────────────────────────
    with tab_datos:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-title'>📂 Fuentes de Datos</div>",
                        unsafe_allow_html=True)
            fuentes = [
                ("🏛️ Ministerio del Interior",
                 "Denuncias VDA, delitos sexuales, femicidios (DNPG-DIAE). 43.245 denuncias 2023.",
                 "catalogodatos.gub.uy"),
                ("🚌 STM Montevideo",
                 "Paradas, recorridos y frecuencias de ómnibus urbanos (~2.800 paradas).",
                 "montevideo.gub.uy"),
                ("📊 INE Uruguay",
                 "Cartografía censal, indicadores socioeconómicos por zona (Censo 2011/est. 2023).",
                 "ine.gub.uy"),
                ("🏙️ IMM Montevideo",
                 "Equipamiento urbano, espacios públicos, alumbrado.",
                 "montevideo.gub.uy/datos"),
                ("🌐 Catálogo Datos Abiertos",
                 "Portal nacional de datos abiertos del Gobierno de Uruguay.",
                 "catalogodatos.gub.uy"),
            ]
            for icon_name, desc, url in fuentes:
                st.markdown(
                    f"""<div style='background:#1C2035;border-radius:10px;padding:14px 16px;
                    margin:8px 0;border:1px solid #252A42;'>
                    <div style='font-weight:600;color:#E8ECF8;margin-bottom:4px;'>{icon_name}</div>
                    <div style='color:#7C85A8;font-size:0.85rem;'>{desc}</div>
                    <div style='color:#6C63FF;font-size:0.78rem;margin-top:4px;'>{url}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='section-title' style='margin-top:20px;'>⚠️ Limitaciones y Advertencias</div>",
                        unsafe_allow_html=True)
            advertencias = [
                "Los datos de denuncias usados en esta demo son **sintéticos calibrados** con estadísticas reales del DNPG.",
                "El modelo estima **riesgo potencial**, no causalidad. No certifica peligrosidad de personas ni lugares.",
                "El **subregistro** de VBG es una limitación inherente: las zonas con menos denuncias no son necesariamente más seguras.",
                "Las intervenciones simuladas son **escenarios hipotéticos** basados en correlaciones del modelo.",
                "Se recomienda **revisión humana** y validación con expertos antes de decisiones operativas.",
            ]
            for adv in advertencias:
                st.markdown(
                    f"<div style='padding:8px 12px;margin:4px 0;background:#1C2035;border-radius:6px;"
                    f"border-left:3px solid #F59E0B;color:#7C85A8;font-size:0.85rem;'>{adv}</div>",
                    unsafe_allow_html=True,
                )

        with col_b:
            st.markdown("<div class='section-title'>📈 Estadísticas del Dataset</div>",
                        unsafe_allow_html=True)

            stats = {
                "Total registros (celda×franja)": f"{len(df):,}",
                "Celdas de análisis": f"{df['cell_id'].nunique():,}",
                "Barrios cubiertos": f"{df['barrio'].nunique()}",
                "Franjas horarias": "4",
                "Denuncias VDA calibradas": "43.245",
                "Paradas STM modeladas": "793",
                "Puntos equipamiento": "1.361",
                "Período de referencia": "2023",
            }
            for k, v in stats.items():
                st.markdown(
                    f"""<div style='display:flex;justify-content:space-between;
                    padding:9px 14px;margin:4px 0;background:#1C2035;border-radius:8px;
                    border:1px solid #252A42;'>
                    <span style='color:#7C85A8;font-size:0.88rem;'>{k}</span>
                    <span style='color:#6C63FF;font-weight:600;font-family:monospace;
                    font-size:0.9rem;'>{v}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='section-title' style='margin-top:20px;'>🏗️ Arquitectura Técnica</div>",
                        unsafe_allow_html=True)
            st.markdown("""
            <div style='background:#1C2035;border-radius:10px;padding:16px;
            border:1px solid #252A42;font-family:monospace;font-size:0.82rem;
            color:#7C85A8;line-height:1.9;'>
            <span style='color:#6C63FF;'>Pipeline:</span> Python · Pandas · GeoPandas<br>
            <span style='color:#6C63FF;'>Geo:</span> Shapely · PyProj · SciPy KDTree<br>
            <span style='color:#6C63FF;'>ML:</span> Scikit-learn · XGBoost · SHAP<br>
            <span style='color:#6C63FF;'>Viz:</span> Folium · Plotly · Streamlit<br>
            <span style='color:#6C63FF;'>Datos:</span> Parquet · GeoJSON<br>
            <span style='color:#6C63FF;'>Deploy:</span> Streamlit Cloud / Docker<br>
            <span style='color:#6C63FF;'>Repo:</span> github.com/facszero/lucia-mvd
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title' style='margin-top:20px;'>🌍 Consideraciones Éticas</div>",
                        unsafe_allow_html=True)
            st.markdown("""
            <div style='color:#7C85A8;font-size:0.87rem;line-height:1.8;'>
            LUCÍA-MVD fue diseñado con enfoque de <b style='color:#6C63FF;'>prevención, no control</b>.<br><br>
            ✅ No publica microdatos individuales<br>
            ✅ Trabaja con agregación espacial anónima<br>
            ✅ Orientado a intervención, no vigilancia<br>
            ✅ Incorpora perspectiva de subregistro VBG<br>
            ✅ Revisión humana requerida antes de decisiones operativas<br>
            ✅ Acceso público para ONGs, municipios y ciudadanía
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='footer'>
        🛡️ <b>LUCÍA-MVD</b> &nbsp;·&nbsp; ONU Mujeres DAT4CCIÓN 2026 &nbsp;·&nbsp;
        Montevideo, Uruguay &nbsp;·&nbsp;
        Datos: DNPG/Ministerio del Interior · STM · INE · IMM &nbsp;·&nbsp;
        Modelo: XGBoost + SHAP (Accuracy 93.4%) &nbsp;·&nbsp;
        <a href='https://github.com/facszero/lucia-mvd' style='color:#6C63FF;text-decoration:none;'>
        GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
