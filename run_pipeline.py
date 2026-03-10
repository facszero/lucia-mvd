"""
LUCÍA-MVD | Pipeline Maestro
==============================
Ejecuta el pipeline completo de extremo a extremo:
1. Ingesta de datos
2. Feature engineering
3. Modelado predictivo
4. Exportación de reportes

Uso:
    python run_pipeline.py           # Pipeline completo
    python run_pipeline.py --ingest  # Solo ingesta
    python run_pipeline.py --model   # Solo modelado
    python run_pipeline.py --report  # Solo reporte
"""

import sys
import time
import logging
import argparse
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lucia.pipeline")


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🛡️  LUCÍA-MVD  |  Pipeline de Análisis de Riesgo Urbano   ║
║   Localizador Urbano de Condiciones de Inseguridad y Alerta  ║
║                                                              ║
║   ONU Mujeres · DAT4CCIÓN 2026 · Montevideo, Uruguay        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def run_full_pipeline():
    banner()
    start = time.time()

    # ── Etapa 1: Ingesta ─────────────────────────────────────────────────────
    log.info("▶ ETAPA 1/3: INGESTA DE DATOS")
    from ingest.ingest import run_ingestion
    datasets = run_ingestion()
    log.info(f"✓ Ingesta completa: {len(datasets)} datasets")

    # ── Etapa 2: Feature Engineering ─────────────────────────────────────────
    log.info("▶ ETAPA 2/3: FEATURE ENGINEERING")
    from features.features import run_feature_engineering
    df_features = run_feature_engineering()
    log.info(f"✓ Features: {df_features.shape} (filas × columnas)")

    # ── Etapa 3: Modelado ─────────────────────────────────────────────────────
    log.info("▶ ETAPA 3/3: MODELADO PREDICTIVO")
    from modeling.model import run_modeling, score_full_dataset
    results = run_modeling()
    log.info(f"✓ XGBoost Accuracy: {results['metrics']['xgb_accuracy']:.3f}")
    log.info(f"✓ XGBoost F1:       {results['metrics']['xgb_f1']:.3f}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    df_scored = pd.read_parquet(BASE_DIR / "data" / "processed" / "dataset_scored.parquet")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    PIPELINE COMPLETADO                       ║
╠══════════════════════════════════════════════════════════════╣
║  Tiempo total:    {elapsed:.1f}s                                   
║  Celdas analizadas:  {df_scored['cell_id'].nunique():,}                               
║  Registros totales:  {len(df_scored):,}                              
║  Barrios cubiertos:  {df_scored['barrio'].nunique()}                                  
║                                                              
║  Distribución de Riesgo:                                     
║    🟢 Bajo:    {(df_scored.nivel_riesgo=='Bajo').sum():,} celdas               
║    🟡 Medio:   {(df_scored.nivel_riesgo=='Medio').sum():,} celdas              
║    🔴 Alto:    {(df_scored.nivel_riesgo=='Alto').sum():,} celdas               
║    🟣 Crítico: {(df_scored.nivel_riesgo=='Crítico').sum():,} celdas             
║                                                              
║  Modelo: XGBoost | Accuracy: {results['metrics']['xgb_accuracy']:.1%}            
╠══════════════════════════════════════════════════════════════╣
║  Para iniciar dashboard:                                     
║  $ streamlit run app.py                                      
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LUCÍA-MVD Pipeline")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--model",  action="store_true")
    args = parser.parse_args()

    if args.ingest:
        from ingest.ingest import run_ingestion
        run_ingestion()
    elif args.model:
        from modeling.model import run_modeling
        run_modeling()
    else:
        run_full_pipeline()
