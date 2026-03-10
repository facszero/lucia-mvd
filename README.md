# 🛡️ LUCÍA-MVD

**Localizador Urbano de Condiciones de Inseguridad y Alerta — Montevideo**

> Sistema de estimación de riesgo urbano diferencial para mujeres en Montevideo, Uruguay.
> Desarrollado para el **DAT4CCIÓN 2026** de ONU Mujeres – América Latina y el Caribe.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ONU Mujeres](https://img.shields.io/badge/ONU%20Mujeres-DAT4CCIÓN%202026-purple)](https://lac.unwomen.org/es/dat4ccion)

---

## 📋 Índice

1. [Problema que resuelve](#-el-problema)
2. [¿Qué es LUCÍA-MVD?](#-qué-es-lucía-mvd)
3. [Demo](#-demo)
4. [Instalación](#-instalación)
5. [Uso rápido](#-uso-rápido)
6. [Arquitectura técnica](#-arquitectura-técnica)
7. [Datos y fuentes](#-datos-y-fuentes)
8. [Metodología](#-metodología)
9. [Resultados del modelo](#-resultados-del-modelo)
10. [Estructura del proyecto](#-estructura-del-proyecto)
11. [Consideraciones éticas](#-consideraciones-éticas)
12. [Escalabilidad](#-escalabilidad)
13. [Contribuir](#-contribuir)
14. [Equipo](#-equipo)
15. [Licencia](#-licencia)

---

## 🚨 El Problema

En 2023, Uruguay registró **43.245 denuncias de violencia doméstica y asociados**, con el **72% de las víctimas siendo mujeres** (DNPG/Ministerio del Interior). Pero los sistemas de respuesta actuales tienen un problema estructural: **reaccionan cuando ya hubo daño**.

Los mapas de delitos convencionales muestran solo lo que ya fue denunciado. Esto significa que:

- El **subregistro de VBG** hace que las zonas con menos denuncias no sean necesariamente más seguras
- Los factores de riesgo **urbanos y de movilidad** (falta de transporte nocturno, aislamiento, déficit de luminarias) no están siendo incorporados en la planificación preventiva
- Los recursos de intervención se priorizan de forma reactiva, no predictiva

> **Una denuncia se hace después de que ocurrió la violencia. LUCÍA-MVD busca identificar el riesgo antes.**

---

## 🛡️ ¿Qué es LUCÍA-MVD?

LUCÍA-MVD es un **sistema de análisis de riesgo urbano diferencial para mujeres** que combina:

| Capa | Datos | Propósito |
|------|-------|-----------|
| 🔴 Violencia observada | Denuncias VDA, delitos sexuales (DNPG/MI) | Ancla empírica del riesgo |
| 🚌 Movilidad urbana | Paradas STM, frecuencias, conectividad | Riesgo de aislamiento nocturno |
| 📊 Vulnerabilidad social | NBI, desempleo, informalidad (INE) | Exposición estructural |
| 🏙️ Equipamiento urbano | Luminarias, refugios VBG, comisarías (IMM) | Capacidad de respuesta |
| 🗺️ Aislamiento espacial | Distancia al centro, conectividad vial | Fragmentación territorial |

El sistema genera:

- **Mapa de calor interactivo** con riesgo por celda (~560m²) y franja horaria
- **Score de riesgo (0-100)** en 4 niveles: Bajo / Medio / Alto / Crítico
- **Explicabilidad SHAP** para cada zona: qué factores dominan el riesgo
- **Recomendaciones de intervención** con costo estimado
- **Simulador de políticas**: qué pasa si mejoro el transporte, agrego luminarias, instalo un refugio

---

## 🎯 Demo

```bash
# Clonar el repositorio
git clone https://github.com/facszero/lucia-mvd.git
cd lucia-mvd

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (~15 segundos)
python run_pipeline.py

# Iniciar dashboard
streamlit run app.py
```

El dashboard se abre en `http://localhost:8501`

---

## ⚙️ Instalación

### Requisitos

- Python 3.10 o superior
- pip 23+
- ~500MB espacio en disco
- Conexión a internet (opcional, para descarga de datos reales)

### Paso a paso

```bash
# 1. Clonar
git clone https://github.com/facszero/lucia-mvd.git
cd lucia-mvd

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar pipeline
python run_pipeline.py

# 5. Iniciar app
streamlit run app.py
```

### Con Docker

```bash
docker build -t lucia-mvd .
docker run -p 8501:8501 lucia-mvd
```

---

## 🚀 Uso Rápido

### Pipeline completo

```bash
python run_pipeline.py
```

### Solo ingesta de datos

```bash
python run_pipeline.py --ingest
```

### Solo modelado

```bash
python run_pipeline.py --model
```

### Dashboard

```bash
streamlit run app.py
```

### API Python

```python
from src.ingest.ingest import run_ingestion
from src.features.features import run_feature_engineering
from src.modeling.model import run_modeling, score_full_dataset, simulate_intervention

# Pipeline
run_ingestion()
df = run_feature_engineering()
results = run_modeling()

# Scoring
df_scored = score_full_dataset(df, results["model_xgb"])

# Simulación de intervención
celda = df_scored[
    (df_scored["barrio"] == "Cerro") &
    (df_scored["franja_horaria"] == "Noche (19-24h)")
].iloc[0]

resultado = simulate_intervention(
    celda,
    intervenciones={
        "frec_nocturna_avg_min": 15,    # Mejor frecuencia STM
        "n_luminarias_200m": 8,          # Más luminarias
        "dist_refugio_km": 0.5,          # Refugio más cercano
    },
    model=results["model_xgb"],
)

print(f"Riesgo actual: {resultado['nivel_actual']}")
print(f"Con intervención: {resultado['nivel_proyectado']}")
print(f"Reducción: {resultado['reduccion_riesgo_pct']}%")
```

---

## 🏗️ Arquitectura Técnica

```
LUCÍA-MVD Architecture
═══════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────┐
│                      DATA LAYER                         │
│                                                         │
│  MI/DNPG    STM Mvd    INE/Censo    IMM Equipamiento    │
│  (VDA CSV)  (GTFS)     (Shapefile)  (GeoJSON)           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   INGESTION LAYER                       │
│                                                         │
│  Real data download (requests) + Synthetic fallback     │
│  Calibrated with official DNPG 2023 statistics          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                 GEO PROCESSING LAYER                    │
│                                                         │
│  Grid creation (900 cells, ~560m²)                      │
│  Spatial join (GeoPandas + SciPy KDTree)                │
│  Feature extraction per cell × time slot                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                 │
│                                                         │
│  IVO (Violence Index)        IAT (Transport Isolation)  │
│  IVS (Social Vulnerability)  IRI (Invisible Risk Index) │
│  Equipment deficit index     Spatial isolation index    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   MODELING LAYER                        │
│                                                         │
│  Composite Score (weighted index) → Level 0-3           │
│  XGBoost Classifier (Accuracy: 95.0%, F1: 95.0%)        │
│  SHAP Explainer (global + local feature importance)     │
│  Intervention Simulator (what-if counterfactuals)       │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER                      │
│                                                         │
│  Streamlit Dashboard                                    │
│  ├── Tab 1: Interactive Risk Map (Folium)                │
│  ├── Tab 2: Neighborhood Analysis (Plotly)              │
│  ├── Tab 3: AI Model (SHAP, metrics)                    │
│  ├── Tab 4: Intervention Simulator                      │
│  └── Tab 5: Data & Methodology                          │
└─────────────────────────────────────────────────────────┘

Stack: Python · Pandas · GeoPandas · Shapely · Scikit-learn
       XGBoost · SHAP · Folium · Plotly · Streamlit
```

---

## 📂 Datos y Fuentes

| Fuente | Dataset | Formato | Período | URL |
|--------|---------|---------|---------|-----|
| DNPG/Ministerio del Interior | Denuncias VDA | CSV/XLSX | 2017-2024 | [catalogodatos.gub.uy](https://catalogodatos.gub.uy/dataset/violencia-domestica-y-asociados) |
| DNPG/Ministerio del Interior | Delitos sexuales | CSV | 2017-2024 | [catalogodatos.gub.uy](https://catalogodatos.gub.uy) |
| DNPG/Ministerio del Interior | Femicidios/Homicidios VBG | CSV | 2017-2024 | [catalogodatos.gub.uy](https://catalogodatos.gub.uy) |
| STM Montevideo | Paradas y recorridos | GTFS/SHP | Actual | [montevideo.gub.uy](https://montevideo.gub.uy) |
| INE Uruguay | Censo 2011 + est. 2023 | SHP/CSV | 2011/2023 | [ine.gub.uy](https://ine.gub.uy) |
| IMM / IDE | Equipamiento urbano | GeoJSON | Actual | [geoportal.montevideo.gub.uy](https://geoportal.montevideo.gub.uy) |

> ⚠️ **Nota**: En esta versión de demo, los datos de denuncias georreferenciadas son **sintéticos calibrados** con las estadísticas oficiales reales publicadas por el DNPG para 2023. Los datos de transporte y equipamiento también son sintéticos. Para producción, integrar los datasets reales disponibles en el catálogo de datos abiertos.

---

## 🔬 Metodología

### 1. Unidad de análisis

- **Espacio**: Grilla cuadrada de celdas de ~560m × 560m sobre Montevideo (900 celdas totales)
- **Tiempo**: 4 franjas horarias — Mañana (6-12h), Tarde (12-19h), Noche (19-24h), Madrugada (0-6h)
- **Unidad**: Celda × Franja → 3.600 registros de análisis

### 2. Score compuesto de riesgo

```
Score(celda, franja) = 
    0.35 × IVO_ajustado(franja)     # Violencia Observada
  + 0.25 × IAT_ajustado(franja)     # Aislamiento Transporte
  + 0.20 × IVS                      # Vulnerabilidad Social
  + 0.12 × Aislamiento_espacial     # Distancia/conectividad
  + 0.08 × Déficit_equipamiento     # Luminarias/refugios/policiales
```

**Ajuste temporal**: cada franja aplica un multiplicador calibrado:
- Mañana: ×0.70
- Tarde: ×1.00 (base)
- Noche: ×1.45 (mayor riesgo)
- Madrugada: ×1.30

### 3. Modelo predictivo XGBoost

- **Input**: 13 features (ver tabla)
- **Output**: Nivel de riesgo (Bajo / Medio / Alto / Crítico)
- **Validación**: 80/20 train-test split, estratificado
- **Métricas**: Accuracy 95.0%, F1-weighted 95.0%
- **Explicabilidad**: SHAP TreeExplainer

### 4. Índice de Riesgo Invisible (IRI)

El diferencial entre el riesgo contextual estimado y las denuncias observadas identifica zonas con potencial subregistro:

```
IRI(celda) = max(0, Riesgo_contextual - Violencia_observada)
```

Zonas con IRI alto son prioritarias para campañas de sensibilización y mejora del acceso a denuncia.

### 5. Simulador de intervenciones

Basado en contrafactuales del modelo XGBoost:

```python
simulate_intervention(celda, {
    "frec_nocturna_avg_min": 15,
    "n_luminarias_200m": 8,
    "dist_refugio_km": 0.5,
})
# → Reducción proyectada de riesgo
```

---

## 📊 Resultados del Modelo

| Métrica | XGBoost | Random Forest (baseline) |
|---------|---------|--------------------------|
| Accuracy | **95.0%** | 91.8% |
| F1-Score Weighted | **95.0%** | 91.8% |

### Importancia de features (SHAP)

| Feature | Importancia SHAP |
|---------|-----------------|
| Aislamiento espacial | 2.78 |
| Vulnerabilidad social | 2.08 |
| Violencia histórica | 2.07 |
| Luminarias en 200m | 1.56 |
| Déficit equipamiento | 1.39 |
| Distancia refugio VBG | 1.39 |
| Frecuencia nocturna STM | 1.19 |
| Factor horario | 0.81 |

### Distribución de riesgo en Montevideo (franja nocturna)

| Nivel | Celdas | % |
|-------|--------|---|
| 🟢 Bajo | 268 | 29.8% |
| 🟡 Medio | 296 | 32.9% |
| 🔴 Alto | 223 | 24.8% |
| 🟣 Crítico | 113 | 12.6% |

---

## 📁 Estructura del Proyecto

```
lucia-mvd/
│
├── 📄 app.py                    # Dashboard Streamlit (aplicación principal)
├── 📄 run_pipeline.py           # Pipeline maestro (end-to-end)
├── 📄 requirements.txt          # Dependencias Python
├── 📄 README.md                 # Este documento
├── 📄 Dockerfile                # Containerización
├── 📄 .gitignore
│
├── 📂 src/                      # Código fuente
│   ├── 📂 ingest/
│   │   └── ingest.py            # Descarga y generación de datos
│   ├── 📂 features/
│   │   └── features.py          # Feature engineering geoespacial
│   ├── 📂 modeling/
│   │   └── model.py             # XGBoost + SHAP + simulador
│   ├── 📂 geo/                  # (Módulos geoespaciales extendidos)
│   ├── 📂 scoring/              # (Scoring en producción)
│   └── 📂 app/                  # (Componentes de dashboard)
│
├── 📂 data/
│   ├── raw/                     # Datos descargados sin procesar
│   ├── processed/               # Datasets procesados (.parquet)
│   ├── synthetic/               # Datos sintéticos calibrados
│   └── external/                # Cartografía y datos externos
│
├── 📂 outputs/
│   ├── models/                  # Modelos entrenados (.pkl)
│   ├── maps/                    # Mapas exportados
│   └── reports/                 # Reportes de análisis
│
├── 📂 notebooks/
│   ├── 01_ingesta.ipynb         # Exploración de datos
│   ├── 02_eda_geo.ipynb         # Análisis geoespacial
│   ├── 03_feature_engineering.ipynb
│   └── 04_modelado.ipynb        # Entrenamiento y evaluación
│
├── 📂 docs/
│   ├── metodologia.md           # Metodología detallada
│   └── datos.md                 # Diccionario de datos
│
├── 📂 tests/
│   └── test_pipeline.py         # Tests unitarios
│
└── 📂 .github/
    └── workflows/
        └── ci.yml               # CI/CD GitHub Actions
```

---

## ⚖️ Consideraciones Éticas

LUCÍA-MVD fue diseñado desde el principio con un enfoque de **prevención, no control** y **transparencia metodológica**.

| Principio | Implementación |
|-----------|---------------|
| **No microdatos** | Solo datos agregados a nivel de celda (~560m²) |
| **Transparencia** | Score explicado con SHAP, ponderaciones públicas |
| **Subregistro** | El IRI reconoce explícitamente el subregistro como limitación |
| **No causalidad** | El modelo estima correlación, no determina responsabilidad |
| **Revisión humana** | Las recomendaciones requieren validación operativa |
| **No vigilancia** | Orientado a intervención urbana, no seguimiento de personas |
| **Acceso abierto** | Código y metodología públicos (MIT License) |

---

## 🌍 Escalabilidad

La arquitectura de LUCÍA-MVD está diseñada para replicarse en otras ciudades de América Latina:

| Ciudad | Datos STM | Datos VDA | Estado |
|--------|-----------|-----------|--------|
| 🇺🇾 Montevideo | STM/MTOP | DNPG/MI | ✅ Implementado |
| 🇦🇷 Buenos Aires | SUBE/GCBA | INDEC | 🔄 Adaptable |
| 🇧🇷 São Paulo | GTFS/SPTrans | SSP-SP | 🔄 Adaptable |
| 🇨🇱 Santiago | GTFS/DTPM | Sernameg | 🔄 Adaptable |
| 🇨🇴 Bogotá | SITP/DTDM | DANE | 🔄 Adaptable |
| 🇲🇽 Ciudad de México | GTFS/STCM | INEGI | 🔄 Adaptable |

**Requisitos mínimos para nueva ciudad**:
- Cartografía de barrios/zonas (SHP/GeoJSON)
- Datos de paradas de transporte público (CSV/GTFS)
- Datos de violencia doméstica por territorio (CSV/XLSX)
- Variables socioeconómicas básicas (censo)

---

## 🤝 Contribuir

```bash
# Fork y clone
git clone https://github.com/facszero/lucia-mvd.git
cd lucia-mvd
git checkout -b feature/mi-mejora

# Hacer cambios y tests
python -m pytest tests/
git commit -m "feat: descripción del cambio"
git push origin feature/mi-mejora
# → Abrir Pull Request
```

### Áreas prioritarias para contribución

- [ ] Integración de datos STM reales (GTFS de montevideo.gub.uy)
- [ ] Incorporar datos de iluminación real de la IMM
- [ ] Módulo de trayectos (riesgo en recorridos casa-trabajo)
- [ ] API REST para integración con apps móviles
- [ ] Tests unitarios y de integración completos
- [ ] Módulo de percepción ciudadana (crowdsourcing)

---

## 👥 Equipo

| Rol | Persona |
|-----|---------|
| Análisis de datos y arquitectura | Fernando Acosta ([@facszero](https://github.com/facszero)) |
| Ciencia de datos y modelado | LUCÍA-MVD Team |
| Diseño de metodología | ONU Mujeres Concepts |

**Proyecto desarrollado para**: DAT4CCIÓN 2026 – ONU Mujeres América Latina y el Caribe

---

## 📄 Licencia

MIT License — Ver [LICENSE](LICENSE) para detalles.

Los datos utilizados son de fuentes abiertas del Gobierno de Uruguay (Ministerio del Interior, STM, INE, IMM) bajo licencias Creative Commons. Los datos sintéticos son calibrados con estadísticas oficiales y son de libre uso para fines académicos y de política pública.

---

## 📚 Referencias

- DNPG/Ministerio del Interior Uruguay. *Estadísticas de Violencia Doméstica y de Género 2023*. Montevideo, 2024.
- OPS/OMS. *Violencia contra la mujer. Informe regional Américas 2025*.
- ONU Mujeres. *Ciudades y Espacios Públicos Seguros para las Mujeres y las Niñas*. 2013.
- Lundberg, S.M., Lee, S.I. *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
- Chen, T., Guestrin, C. *XGBoost: A Scalable Tree Boosting System*. KDD 2016.

---

<div align="center">
<br>
🛡️ <strong>LUCÍA-MVD</strong> &nbsp;·&nbsp; No esperamos a que el riesgo se convierta en tragedia.<br>
Usamos datos para anticiparlo y orientar intervenciones concretas donde más se necesitan.
<br><br>
<a href="https://lac.unwomen.org/es/dat4ccion">ONU Mujeres DAT4CCIÓN 2026</a> &nbsp;·&nbsp;
<a href="https://github.com/facszero/lucia-mvd">GitHub</a>
</div>
