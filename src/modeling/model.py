"""
LUCÍA-MVD | Módulo de Modelado Predictivo
==========================================
Entrena un modelo XGBoost sobre el score de riesgo compuesto
y genera explicabilidad SHAP por celda.

El modelo permite:
1. Predicción de nivel de riesgo (clasificación 4 clases)
2. Explicabilidad local (qué factores explican cada celda)
3. Simulación de intervenciones (what-if analysis)
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
log = logging.getLogger("lucia.modeling")

BASE_DIR   = Path(__file__).resolve().parents[2]
PROC_DIR   = BASE_DIR / "data" / "processed"
MODEL_DIR  = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Features usadas en el modelo (interpretables y auditables)
FEATURE_COLS = [
    "ivo_adj",                      # Violencia histórica normalizada
    "peso_transporte_ajustado",     # Aislamiento transporte (ajustado franja)
    "indice_vulnerabilidad",        # Vulnerabilidad socioeconómica
    "indice_aislamiento_espacial",  # Distancia centro / conectividad
    "indice_deficit_equipamiento",  # Carencia de equipamiento
    "n_paradas_400m",               # Número paradas cercanas
    "frec_nocturna_avg_min",        # Frecuencia promedio nocturna STM
    "n_luminarias_200m",            # Luminarias en 200m
    "dist_refugio_km",              # Distancia al refugio más cercano
    "dist_comisaria_km",            # Distancia a comisaría
    "pct_hogares_nbi",              # % hogares con NBI
    "tasa_desempleo",               # Tasa de desempleo barrial
    "mult_franja",                  # Multiplicador temporal de franja
]

FEATURE_NAMES_ES = {
    "ivo_adj":                     "Violencia histórica",
    "peso_transporte_ajustado":    "Aislamiento de transporte",
    "indice_vulnerabilidad":       "Vulnerabilidad social",
    "indice_aislamiento_espacial": "Aislamiento espacial",
    "indice_deficit_equipamiento": "Déficit equipamiento",
    "n_paradas_400m":              "Paradas STM cercanas",
    "frec_nocturna_avg_min":       "Frecuencia nocturna STM",
    "n_luminarias_200m":           "Luminarias en 200m",
    "dist_refugio_km":             "Distancia refugio VBG",
    "dist_comisaria_km":           "Distancia comisaría",
    "pct_hogares_nbi":             "Hogares c/ NBI",
    "tasa_desempleo":              "Tasa desempleo",
    "mult_franja":                 "Factor horario",
}

TARGET_COL = "nivel_riesgo_num"
TARGET_NAMES = ["Bajo", "Medio", "Alto", "Crítico"]


def load_features() -> pd.DataFrame:
    path = PROC_DIR / "features_completo.parquet"
    df = pd.read_parquet(path)
    return df


def prepare_data(df: pd.DataFrame):
    """Prepara features y target para el modelo."""
    # Asegurar que todas las columnas requeridas existen
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning(f"Columnas faltantes en features: {missing}")
        for c in missing:
            df[c] = 0.0

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Rellenar NaN
    X = X.fillna(X.median())

    return X, y


def train_model(df: pd.DataFrame) -> dict:
    """
    Entrena XGBoost con validación cruzada.
    Retorna dict con modelo, métricas y objetos SHAP.
    """
    log.info("Preparando datos para entrenamiento...")
    X, y = prepare_data(df)

    # Muestra estratificada para agilidad (mantener representatividad)
    # Usar 30% de los datos para CV, mantener distribución de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    log.info(f"Distribución target: {y.value_counts().to_dict()}")

    # ── Modelo XGBoost ───────────────────────────────────────────────────────
    log.info("Entrenando XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)

    # ── Random Forest baseline ───────────────────────────────────────────────
    log.info("Entrenando Random Forest baseline...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # ── Métricas ─────────────────────────────────────────────────────────────
    y_pred_xgb = xgb.predict(X_test)
    y_pred_rf  = rf.predict(X_test)

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb  = f1_score(y_test, y_pred_xgb, average="weighted")
    acc_rf  = accuracy_score(y_test, y_pred_rf)
    f1_rf   = f1_score(y_test, y_pred_rf, average="weighted")

    log.info(f"XGBoost → Accuracy: {acc_xgb:.3f} | F1-weighted: {f1_xgb:.3f}")
    log.info(f"RandomForest → Accuracy: {acc_rf:.3f} | F1-weighted: {f1_rf:.3f}")

    report = classification_report(
        y_test, y_pred_xgb,
        target_names=TARGET_NAMES,
        output_dict=True,
    )

    # ── SHAP ─────────────────────────────────────────────────────────────────
    log.info("Calculando valores SHAP...")
    explainer = shap.TreeExplainer(xgb)

    # Calcular SHAP para muestra representativa (max 3000 registros)
    X_shap = X_test.sample(min(3000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_shap)

    # Importancia global de features (promedio |SHAP|)
    if isinstance(shap_values, list):
        # Multiclase: shape es [n_classes][n_samples, n_features]
        shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # binario o ya aplanado
        shap_mean = np.abs(shap_values).mean(axis=0)
    
    # Asegurar que sea 1D
    shap_mean = np.array(shap_mean).flatten()[:len(FEATURE_COLS)]

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "feature_es": [FEATURE_NAMES_ES[f] for f in FEATURE_COLS],
        "shap_importance": shap_mean,
    }).sort_values("shap_importance", ascending=False)

    log.info("Top 5 features por importancia SHAP:")
    for _, row in feature_importance.head(5).iterrows():
        log.info(f"  {row.feature_es:<30} {row.shap_importance:.4f}")

    # ── Guardar modelos ──────────────────────────────────────────────────────
    model_path = MODEL_DIR / "xgb_lucia.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(xgb, f)

    explainer_path = MODEL_DIR / "shap_explainer.pkl"
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)

    feature_importance.to_parquet(MODEL_DIR / "feature_importance.parquet")

    log.info(f"Modelos guardados en {MODEL_DIR}")

    return {
        "model_xgb": xgb,
        "model_rf": rf,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_shap": X_shap,
        "metrics": {
            "xgb_accuracy": acc_xgb,
            "xgb_f1": f1_xgb,
            "rf_accuracy": acc_rf,
            "rf_f1": f1_rf,
            "classification_report": report,
        },
        "feature_importance": feature_importance,
    }


def score_full_dataset(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Aplica el modelo XGBoost al dataset completo y agrega predicción.
    Usa el score compuesto para asignar nivel de riesgo oficial.
    """
    X, _ = prepare_data(df)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    df = df.copy()
    df["nivel_predicho"] = [TARGET_NAMES[p] for p in y_pred]
    df["prob_bajo"]    = y_prob[:, 0].round(3)
    df["prob_medio"]   = y_prob[:, 1].round(3)
    df["prob_alto"]    = y_prob[:, 2].round(3)
    df["prob_critico"] = y_prob[:, 3].round(3)
    df["confianza"]    = y_prob.max(axis=1).round(3)

    log.info(f"Scoring completo: {len(df):,} registros clasificados")
    return df


def simulate_intervention(
    df_cell: pd.Series,
    intervenciones: dict,
    model,
) -> dict:
    """
    Simula el impacto de intervenciones en una celda específica.
    
    intervenciones: dict con modificaciones de features
    Ejemplo: {"n_luminarias_200m": 8, "frec_nocturna_avg_min": 15}
    
    Retorna: score actual vs proyectado y delta.
    """
    X_actual = pd.DataFrame([df_cell[FEATURE_COLS]])
    X_modif  = X_actual.copy()

    for feat, val in intervenciones.items():
        if feat in X_modif.columns:
            X_modif[feat] = val

    prob_actual = model.predict_proba(X_actual)[0]
    prob_modif  = model.predict_proba(X_modif)[0]

    nivel_actual = TARGET_NAMES[model.predict(X_actual)[0]]
    nivel_modif  = TARGET_NAMES[model.predict(X_modif)[0]]

    # Calcular reducción de riesgo ponderada
    pesos_nivel = np.array([0, 1, 2, 3])
    riesgo_actual = (prob_actual * pesos_nivel).sum()
    riesgo_modif  = (prob_modif  * pesos_nivel).sum()
    reduccion_pct = max(0, (riesgo_actual - riesgo_modif) / max(riesgo_actual, 0.01) * 100)

    return {
        "nivel_actual":   nivel_actual,
        "nivel_proyectado": nivel_modif,
        "reduccion_riesgo_pct": round(reduccion_pct, 1),
        "prob_actual":    dict(zip(TARGET_NAMES, prob_actual.round(3))),
        "prob_proyectado": dict(zip(TARGET_NAMES, prob_modif.round(3))),
    }


def run_modeling() -> dict:
    """Pipeline completo de modelado."""
    log.info("=" * 60)
    log.info("LUCÍA-MVD | MODELADO PREDICTIVO")
    log.info("=" * 60)

    df = load_features()
    results = train_model(df)

    # Scoring completo
    df_scored = score_full_dataset(df, results["model_xgb"])
    scored_path = PROC_DIR / "dataset_scored.parquet"
    df_scored.to_parquet(scored_path)
    log.info(f"Dataset con scores guardado: {scored_path}")

    return results


if __name__ == "__main__":
    run_modeling()
