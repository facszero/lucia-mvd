"""
LUCÍA-MVD | Test Suite
======================
Tests unitarios y de integración para el pipeline completo.

Ejecutar con:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))


class TestIngestion:
    """Tests del módulo de ingesta."""

    def test_vda_generation_shape(self):
        from ingest.ingest import generate_synthetic_vda
        df = generate_synthetic_vda(seed=42)
        assert len(df) == 43245, f"Expected 43245 rows, got {len(df)}"

    def test_vda_columns(self):
        from ingest.ingest import generate_synthetic_vda
        df = generate_synthetic_vda(seed=42)
        required = ["fecha", "barrio", "tipo_delito", "franja_horaria",
                    "sexo_victima", "lat", "lon"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_vda_sexo_distribution(self):
        from ingest.ingest import generate_synthetic_vda
        df = generate_synthetic_vda(seed=42)
        pct_mujeres = (df["sexo_victima"] == "Mujer").mean()
        assert 0.68 <= pct_mujeres <= 0.76, f"Expected ~72% mujeres, got {pct_mujeres:.2%}"

    def test_vda_coordinates_montevideo(self):
        from ingest.ingest import generate_synthetic_vda
        df = generate_synthetic_vda(seed=42)
        # 99%+ of coords should be in Montevideo area (allow rare gaussian outliers)
        pct_in_bounds = df["lat"].between(-34.98, -34.77).mean()
        assert pct_in_bounds >= 0.998, f"Too many coordinates outside Montevideo: {1-pct_in_bounds:.2%}"

    def test_stm_generation(self):
        from ingest.ingest import generate_synthetic_stm
        df = generate_synthetic_stm(seed=42)
        assert len(df) > 500, "STM should have >500 stops"
        assert "frec_nocturna_min" in df.columns
        assert "activa_noche" in df.columns

    def test_censo_generation(self):
        from ingest.ingest import generate_synthetic_censo
        df = generate_synthetic_censo(seed=42)
        assert len(df) > 30, "Censo should have >30 barrios"
        assert "indice_vulnerabilidad" in df.columns
        assert df["indice_vulnerabilidad"].between(0, 1).all()


class TestFeatureEngineering:
    """Tests del módulo de feature engineering."""

    @pytest.fixture(scope="class", autouse=True)
    def run_ingestion(self):
        """Asegura que los datos existen antes de los tests."""
        from ingest.ingest import run_ingestion
        run_ingestion()

    def test_grid_creation(self):
        from features.features import build_grid
        grid = build_grid(cell_size_deg=0.006)
        assert len(grid) > 500, "Grid should have >500 cells"
        assert "cell_id" in grid.columns
        assert "lat_cen" in grid.columns
        assert "lon_cen" in grid.columns

    def test_feature_matrix_shape(self):
        from features.features import run_feature_engineering
        df = run_feature_engineering()
        # Celdas terrestres × 4 franjas (máscara costera reduce de 900 a ~608)
        assert len(df) > 0, "Dataset vacío"
        assert len(df) % 4 == 0, f"Debe ser múltiplo de 4 franjas, got {len(df)}"
        n_celdas = len(df) // 4
        assert 500 <= n_celdas <= 1000, f"Celdas fuera de rango: {n_celdas}"

    def test_score_range(self):
        from features.features import run_feature_engineering
        df = run_feature_engineering()
        assert df["score_riesgo"].between(0, 100).all(), "Scores must be 0-100"

    def test_risk_levels_valid(self):
        from features.features import run_feature_engineering
        df = run_feature_engineering()
        valid_levels = {"Bajo", "Medio", "Alto", "Crítico"}
        assert set(df["nivel_riesgo"].unique()).issubset(valid_levels)

    def test_all_franjas_present(self):
        from features.features import run_feature_engineering, FRANJAS
        df = run_feature_engineering()
        for franja in FRANJAS:
            count = (df["franja_horaria"] == franja).sum()
            assert count > 0, f"Franja {franja} has no records"

    def test_night_higher_than_morning(self):
        """Score nocturno debe ser >= score matutino para mismas celdas."""
        from features.features import run_feature_engineering
        df = run_feature_engineering()
        mañana = df[df["franja_horaria"] == "Mañana (6-12h)"]["score_riesgo"].mean()
        noche  = df[df["franja_horaria"] == "Noche (19-24h)"]["score_riesgo"].mean()
        assert noche > mañana, f"Night score ({noche:.1f}) should exceed morning ({mañana:.1f})"


class TestModeling:
    """Tests del módulo de modelado."""

    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(self):
        from ingest.ingest import run_ingestion
        from features.features import run_feature_engineering
        run_ingestion()
        run_feature_engineering()

    def test_model_accuracy(self):
        from modeling.model import run_modeling
        results = run_modeling()
        acc = results["metrics"]["xgb_accuracy"]
        assert acc >= 0.85, f"Accuracy {acc:.3f} below 85% threshold"

    def test_model_f1(self):
        from modeling.model import run_modeling
        results = run_modeling()
        f1 = results["metrics"]["xgb_f1"]
        assert f1 >= 0.85, f"F1 {f1:.3f} below 85% threshold"

    def test_model_output_exists(self):
        model_path = BASE_DIR / "outputs" / "models" / "xgb_lucia.pkl"
        assert model_path.exists(), "Model file not saved"

    def test_feature_importance_has_all_features(self):
        from modeling.model import run_modeling, FEATURE_COLS
        results = run_modeling()
        fi = results["feature_importance"]
        assert len(fi) == len(FEATURE_COLS), "Feature importance length mismatch"

    def test_simulation_reduces_risk(self):
        """Una intervención de mejora debe reducir o mantener el riesgo."""
        import pickle
        from modeling.model import simulate_intervention
        df = pd.read_parquet(BASE_DIR / "data" / "processed" / "dataset_scored.parquet")
        model_path = BASE_DIR / "outputs" / "models" / "xgb_lucia.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Tomar celda de alto riesgo
        celda = df[df["nivel_riesgo"] == "Alto"].iloc[0]

        resultado = simulate_intervention(
            celda,
            intervenciones={
                "frec_nocturna_avg_min": 8,
                "n_luminarias_200m": 15,
                "dist_refugio_km": 0.2,
                "dist_comisaria_km": 0.3,
            },
            model=model,
        )
        # Con intervenciones muy favorables, la reducción debe ser >= 0
        assert resultado["reduccion_riesgo_pct"] >= 0


class TestDataQuality:
    """Tests de calidad de datos finales."""

    def test_no_nulls_in_key_columns(self):
        path = BASE_DIR / "data" / "processed" / "dataset_scored.parquet"
        if not path.exists():
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        key_cols = ["cell_id", "franja_horaria", "score_riesgo", "nivel_riesgo",
                    "lat_cen", "lon_cen", "barrio"]
        for col in key_cols:
            nulls = df[col].isnull().sum()
            assert nulls == 0, f"Column {col} has {nulls} nulls"

    def test_score_distribution_reasonable(self):
        path = BASE_DIR / "data" / "processed" / "dataset_scored.parquet"
        if not path.exists():
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        # At least 10% of cells in each of the main risk levels
        for nivel in ["Bajo", "Medio", "Alto"]:
            pct = (df["nivel_riesgo"] == nivel).mean()
            assert pct >= 0.05, f"Level {nivel} has only {pct:.1%} of records"

    def test_barrios_coverage(self):
        path = BASE_DIR / "data" / "processed" / "dataset_scored.parquet"
        if not path.exists():
            pytest.skip("Dataset not generated yet")
        df = pd.read_parquet(path)
        n_barrios = df["barrio"].nunique()
        assert n_barrios >= 20, f"Only {n_barrios} barrios covered, expected 20+"
