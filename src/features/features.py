"""
LUCÍA-MVD | Feature Engineering
=================================
Construye la grilla hexagonal de Montevideo y calcula todas las
features por celda y franja horaria.

Índices calculados:
  - Índice de Violencia Observada (IVO)
  - Índice de Aislamiento de Transporte (IAT)
  - Índice de Vulnerabilidad Social (IVS)
  - Índice de Riesgo Invisible (IRI)
  - Score de Riesgo Final (SRF) 0-100
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger("lucia.features")

BASE_DIR = Path(__file__).resolve().parents[2]
SYN_DIR  = BASE_DIR / "data" / "synthetic"
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box de Montevideo
MVD_BOUNDS = {
    "lat_min": -34.940, "lat_max": -34.820,
    "lon_min": -56.300, "lon_max": -56.030,
}

# Polígono de tierra de Montevideo (excluye Río de la Plata y Bahía)
# Coordenadas: (lon, lat) - borde costero aproximado del departamento
MVD_LAND_COORDS = [
    # Límite norte (departamental) W → E
    (-56.300, -34.820),
    (-56.030, -34.820),
    # Costa este - bajando hacia Carrasco
    (-56.030, -34.858),
    (-56.045, -34.872),  # Punta Gorda / Carrasco
    (-56.065, -34.888),
    (-56.090, -34.898),  # Buceo / Malvín
    (-56.120, -34.908),  # Pocitos
    (-56.150, -34.915),  # Punta Carretas
    (-56.175, -34.918),  # Playa Ramírez
    (-56.195, -34.915),  # Parque Rodó
    (-56.215, -34.910),  # Puerto / Ciudad Vieja
    (-56.235, -34.905),
    (-56.255, -34.900),  # Cerro interior
    (-56.270, -34.895),
    (-56.290, -34.892),  # Cerro oeste
    (-56.300, -34.890),  # Límite oeste
    # Cierre al norte
    (-56.300, -34.820),
]
MVD_LAND_POLYGON = Polygon(MVD_LAND_COORDS)

# Ponderaciones del score final (suma = 1.0)
PESOS = {
    "violencia_observada":     0.35,
    "aislamiento_transporte":  0.25,
    "vulnerabilidad_social":   0.20,
    "aislamiento_espacial":    0.12,
    "deficit_equipamiento":    0.08,
}

FRANJAS = ["Mañana (6-12h)", "Tarde (12-19h)", "Noche (19-24h)", "Madrugada (0-6h)"]


def build_grid(cell_size_deg: float = 0.006) -> gpd.GeoDataFrame:
    """
    Construye grilla cuadrada sobre Montevideo, enmascarada al polígono de tierra.
    cell_size_deg ≈ 0.006° ≈ 560m en latitud (resolución análisis urbano).
    Las celdas cuyo centroide cae en el Río de la Plata o fuera del límite
    departamental son excluidas.
    """
    lats = np.arange(MVD_BOUNDS["lat_min"], MVD_BOUNDS["lat_max"], cell_size_deg)
    lons = np.arange(MVD_BOUNDS["lon_min"], MVD_BOUNDS["lon_max"], cell_size_deg)

    cells = []
    cell_id = 0
    for lat in lats:
        for lon in lons:
            poly = Polygon([
                (lon, lat),
                (lon + cell_size_deg, lat),
                (lon + cell_size_deg, lat + cell_size_deg),
                (lon, lat + cell_size_deg),
            ])
            centroid = Point(lon + cell_size_deg / 2, lat + cell_size_deg / 2)
            # Excluir celdas fuera del polígono de tierra de Montevideo
            if not MVD_LAND_POLYGON.contains(centroid):
                continue
            cells.append({
                "cell_id": cell_id,
                "geometry": poly,
                "lat_cen": lat + cell_size_deg / 2,
                "lon_cen": lon + cell_size_deg / 2,
            })
            cell_id += 1

    gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    # Re-indexar cell_id secuencialmente
    gdf["cell_id"] = range(len(gdf))
    log.info(f"Grilla creada: {len(gdf):,} celdas terrestres ({cell_size_deg*111:.0f}m × {cell_size_deg*111:.0f}m)")
    return gdf


def assign_vda_to_grid(vda_df: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Asigna cada denuncia VDA a su celda de la grilla."""
    # Solo mujeres víctimas para el análisis de VBG
    vda_mujeres = vda_df[vda_df["sexo_victima"] == "Mujer"].copy()

    gdf_vda = gpd.GeoDataFrame(
        vda_mujeres,
        geometry=gpd.points_from_xy(vda_mujeres["lon"], vda_mujeres["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(gdf_vda, grid[["cell_id", "geometry"]], how="left", predicate="within")

    # Agregar por celda y franja
    agg = (
        joined.groupby(["cell_id", "franja_horaria"])
        .agg(
            n_denuncias=("fecha", "count"),
            n_vda_pura=("tipo_delito", lambda x: (x == "Violencia doméstica").sum()),
            n_fin_semana=("es_finde", "sum"),
        )
        .reset_index()
    )

    # Crear tabla completa (todas las celdas × todas las franjas)
    idx = pd.MultiIndex.from_product(
        [grid["cell_id"].unique(), FRANJAS],
        names=["cell_id", "franja_horaria"],
    )
    full = agg.set_index(["cell_id", "franja_horaria"]).reindex(idx, fill_value=0).reset_index()
    log.info(f"VDA asignado a grilla: {len(full):,} registros celda×franja")
    return full


def compute_transport_features(stm_df: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calcula features de transporte por celda:
    - Número de paradas en radio de 400m
    - Frecuencia promedio diurna/nocturna
    - Índice de aislamiento nocturno
    """
    stm_coords = stm_df[["lat", "lon"]].values
    tree = cKDTree(stm_coords)

    records = []
    for _, row in grid.iterrows():
        center = np.array([[row.lat_cen, row.lon_cen]])
        # Radio ≈ 400m en grados
        radio = 0.004
        idx = tree.query_ball_point(center[0], r=radio)

        if len(idx) > 0:
            paradas_cercanas = stm_df.iloc[idx]
            n_paradas = len(paradas_cercanas)
            frec_diurna = paradas_cercanas["frec_diurna_min"].mean()
            frec_nocturna = paradas_cercanas["frec_nocturna_min"].mean()
            n_nocturnas = paradas_cercanas["activa_noche"].sum()
            n_lineas = paradas_cercanas["num_lineas"].sum()
        else:
            n_paradas = 0
            frec_diurna = 60.0  # Sin servicio = espera muy alta
            frec_nocturna = 90.0
            n_nocturnas = 0
            n_lineas = 0

        # Índice de aislamiento nocturno (0=bien conectado, 1=aislado)
        aislamiento_nocturno = 0.0
        if n_paradas == 0:
            aislamiento_nocturno = 1.0
        else:
            # Penalizar frecuencias altas (esperas largas) y pocas paradas
            ais_frec = min(frec_nocturna / 60.0, 1.0)
            ais_paradas = max(0, 1 - n_paradas / 6)
            aislamiento_nocturno = 0.6 * ais_frec + 0.4 * ais_paradas

        records.append({
            "cell_id": row.cell_id,
            "n_paradas_400m": n_paradas,
            "frec_diurna_avg_min": round(frec_diurna, 1),
            "frec_nocturna_avg_min": round(frec_nocturna, 1),
            "n_paradas_nocturnas": n_nocturnas,
            "n_lineas_cercanas": n_lineas,
            "indice_aislamiento_nocturno": round(aislamiento_nocturno, 3),
        })

    df = pd.DataFrame(records)
    log.info(f"Features transporte calculadas: {len(df):,} celdas")
    return df


def compute_social_features(censo_df: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Asigna features socioeconómicas a la grilla por proximidad al centroide de barrio."""
    censo_coords = censo_df[["lat_centroide", "lon_centroide"]].values
    tree = cKDTree(censo_coords)

    records = []
    for _, row in grid.iterrows():
        dist, idx = tree.query([row.lat_cen, row.lon_cen], k=1)
        barrio_data = censo_df.iloc[idx]

        records.append({
            "cell_id": row.cell_id,
            "barrio_asignado": barrio_data["barrio"],
            "indice_vulnerabilidad": barrio_data["indice_vulnerabilidad"],
            "pct_jefas_hogar": barrio_data["pct_jefas_hogar_sin_pareja"],
            "densidad_pob_km2": barrio_data["densidad_pob_km2"],
            "pct_hogares_nbi": barrio_data["pct_hogares_nbi"],
            "tasa_desempleo": barrio_data["tasa_desempleo"],
            "pct_informalidad": barrio_data["pct_informalidad_laboral"],
        })

    df = pd.DataFrame(records)
    log.info(f"Features sociales asignadas: {len(df):,} celdas")
    return df


def compute_equipment_features(equip_df: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calcula acceso a equipamiento urbano relevante para seguridad."""
    # Separar por tipo relevante
    luminarias = equip_df[equip_df["tipo"] == "Luminaria funcional"][["lat", "lon"]].values
    refugios   = equip_df[equip_df["tipo"] == "Refugio/Centro atención VBG"][["lat", "lon"]].values
    policiales = equip_df[equip_df["tipo"] == "Comisaría/Seccional"][["lat", "lon"]].values
    salud      = equip_df[equip_df["tipo"] == "Centro de salud"][["lat", "lon"]].values
    plazas     = equip_df[equip_df["tipo"] == "Plaza/Espacio público"][["lat", "lon"]].values

    tree_lum = cKDTree(luminarias) if len(luminarias) > 0 else None
    tree_ref = cKDTree(refugios)   if len(refugios) > 0 else None
    tree_pol = cKDTree(policiales) if len(policiales) > 0 else None

    records = []
    for _, row in grid.iterrows():
        pt = [row.lat_cen, row.lon_cen]

        # Luminarias en 200m
        n_lum = len(tree_lum.query_ball_point(pt, r=0.002)) if tree_lum else 0

        # Distancia al refugio más cercano (grados → km aprox)
        dist_ref = tree_ref.query(pt)[0] * 111 if tree_ref else 15.0

        # Distancia a comisaría más cercana
        dist_pol = tree_pol.query(pt)[0] * 111 if tree_pol else 10.0

        # Índice de déficit de equipamiento (0=bien equipado, 1=carente)
        deficit_lum = max(0, 1 - n_lum / 5)
        deficit_ref = min(dist_ref / 3.0, 1.0)  # 3km+ = máximo déficit
        deficit_pol = min(dist_pol / 2.0, 1.0)  # 2km+ = máximo déficit

        deficit_total = 0.4 * deficit_lum + 0.35 * deficit_ref + 0.25 * deficit_pol

        records.append({
            "cell_id": row.cell_id,
            "n_luminarias_200m": n_lum,
            "dist_refugio_km": round(dist_ref, 2),
            "dist_comisaria_km": round(dist_pol, 2),
            "indice_deficit_equipamiento": round(deficit_total, 3),
        })

    df = pd.DataFrame(records)
    log.info(f"Features equipamiento calculadas: {len(df):,} celdas")
    return df


def compute_spatial_isolation(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Índice de aislamiento espacial basado en:
    - Distancia al centro (Palacio Legislativo como referencia)
    - Conectividad vial aproximada por posición
    """
    centro_lat, centro_lon = -34.9058, -56.1888  # Centro de Montevideo

    records = []
    for _, row in grid.iterrows():
        dist_centro = (
            (row.lat_cen - centro_lat) ** 2 + (row.lon_cen - centro_lon) ** 2
        ) ** 0.5 * 111  # en km

        # Normalizar: 0km=centro (0), 15km+=periferia (1)
        iso_dist = min(dist_centro / 12.0, 1.0)

        records.append({
            "cell_id": row.cell_id,
            "dist_centro_km": round(dist_centro, 2),
            "indice_aislamiento_espacial": round(iso_dist, 3),
        })

    df = pd.DataFrame(records)
    return df


def build_feature_matrix(
    vda_agg: pd.DataFrame,
    transport: pd.DataFrame,
    social: pd.DataFrame,
    equipment: pd.DataFrame,
    spatial: pd.DataFrame,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Une todas las features en una única tabla celda×franja."""
    # Base: todas las celdas × todas las franjas
    base_cells = pd.MultiIndex.from_product(
        [grid["cell_id"].values, FRANJAS], names=["cell_id", "franja_horaria"]
    )
    df = pd.DataFrame(index=base_cells).reset_index()

    # Merge VDA (ya tiene franja_horaria)
    df = df.merge(vda_agg, on=["cell_id", "franja_horaria"], how="left")

    # Merge features sin franja (se replican para cada franja)
    for extra, key in [
        (transport, "cell_id"),
        (social, "cell_id"),
        (equipment, "cell_id"),
        (spatial, "cell_id"),
    ]:
        df = df.merge(extra, on=key, how="left")

    # Merge coords de la grilla
    df = df.merge(grid[["cell_id", "lat_cen", "lon_cen"]], on="cell_id", how="left")

    # Rellenar NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # ── Ajuste por franja: multiplicador de riesgo nocturno ─────────────────
    franja_mult = {
        "Mañana (6-12h)":    0.70,
        "Tarde (12-19h)":    1.00,
        "Noche (19-24h)":    1.45,
        "Madrugada (0-6h)":  1.30,
    }
    df["mult_franja"] = df["franja_horaria"].map(franja_mult)

    # La frecuencia nocturna aplica extra peso en franjas nocturnas
    df["peso_transporte_ajustado"] = np.where(
        df["franja_horaria"].isin(["Noche (19-24h)", "Madrugada (0-6h)"]),
        df["indice_aislamiento_nocturno"],
        df["indice_aislamiento_nocturno"] * 0.6,  # menor peso de noche en horario diurno
    )

    log.info(f"Matriz de features: {len(df):,} filas, {len(df.columns)} columnas")
    return df


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Score de Riesgo Final (0-100) como índice ponderado.
    
    Componentes:
    1. Violencia Observada      (35%): tasa de denuncias normalizada
    2. Aislamiento Transporte   (25%): índice nocturno ajustado por franja
    3. Vulnerabilidad Social    (20%): índice socioeconómico
    4. Aislamiento Espacial     (12%): distancia/conectividad
    5. Déficit Equipamiento      (8%): luminarias, refugios, policiales
    """
    scaler = MinMaxScaler()

    # Normalizar denuncias (suavizado logarítmico)
    df["n_den_log"] = np.log1p(df["n_denuncias"])
    df["ivo"] = scaler.fit_transform(df[["n_den_log"]]).flatten()

    # Ajustar por multiplicador de franja
    df["ivo_adj"] = df["ivo"] * df["mult_franja"]
    df["ivo_adj"] = scaler.fit_transform(df[["ivo_adj"]]).flatten()

    # Score compuesto
    df["score_raw"] = (
        PESOS["violencia_observada"]    * df["ivo_adj"]
        + PESOS["aislamiento_transporte"] * df["peso_transporte_ajustado"]
        + PESOS["vulnerabilidad_social"]  * df["indice_vulnerabilidad"]
        + PESOS["aislamiento_espacial"]   * df["indice_aislamiento_espacial"]
        + PESOS["deficit_equipamiento"]   * df["indice_deficit_equipamiento"]
    )

    # Llevar a escala 0-100
    df["score_riesgo"] = (
        scaler.fit_transform(df[["score_raw"]]).flatten() * 100
    ).round(1)

    # Clasificación en 4 niveles
    def classify_risk(score):
        if score >= 75:
            return "Crítico"
        elif score >= 55:
            return "Alto"
        elif score >= 35:
            return "Medio"
        else:
            return "Bajo"

    df["nivel_riesgo"] = df["score_riesgo"].apply(classify_risk)

    # Categoría numérica para modelo
    cat_map = {"Bajo": 0, "Medio": 1, "Alto": 2, "Crítico": 3}
    df["nivel_riesgo_num"] = df["nivel_riesgo"].map(cat_map)

    # Índice de Riesgo Invisible (gap entre contexto y denuncia)
    df["riesgo_contextual"] = (
        0.4 * df["peso_transporte_ajustado"]
        + 0.35 * df["indice_vulnerabilidad"]
        + 0.25 * df["indice_deficit_equipamiento"]
    )
    df["riesgo_invisible"] = np.maximum(0, df["riesgo_contextual"] - df["ivo_adj"])
    df["riesgo_invisible"] = (
        scaler.fit_transform(df[["riesgo_invisible"]]).flatten() * 100
    ).round(1)

    log.info(
        f"Score calculado. Distribución: "
        f"Bajo={( df.nivel_riesgo=='Bajo').sum():,} | "
        f"Medio={(df.nivel_riesgo=='Medio').sum():,} | "
        f"Alto={(df.nivel_riesgo=='Alto').sum():,} | "
        f"Crítico={(df.nivel_riesgo=='Crítico').sum():,}"
    )
    return df


def top_3_factors(row: pd.Series) -> str:
    """Retorna los 3 principales factores de riesgo para una celda."""
    factores = {
        "Violencia histórica":      row.get("ivo_adj", 0),
        "Aislamiento transporte":   row.get("peso_transporte_ajustado", 0),
        "Vulnerabilidad social":    row.get("indice_vulnerabilidad", 0),
        "Aislamiento espacial":     row.get("indice_aislamiento_espacial", 0),
        "Déficit equipamiento":     row.get("indice_deficit_equipamiento", 0),
    }
    top = sorted(factores.items(), key=lambda x: x[1], reverse=True)[:3]
    return " | ".join([f"{k} ({v:.2f})" for k, v in top])


def intervention_recommendation(row: pd.Series) -> str:
    """Genera recomendación de intervención basada en factores dominantes."""
    recs = []
    if row.get("indice_aislamiento_nocturno", 0) > 0.6:
        recs.append("🚌 Reforzar frecuencia STM nocturna")
    if row.get("n_luminarias_200m", 5) < 2:
        recs.append("💡 Ampliar alumbrado público")
    if row.get("dist_refugio_km", 0) > 2.5:
        recs.append("🏥 Instalar punto de atención VBG cercano")
    if row.get("indice_vulnerabilidad", 0) > 0.65:
        recs.append("🤝 Programas sociales focalizados")
    if row.get("dist_comisaria_km", 0) > 2:
        recs.append("🚓 Patrullaje preventivo en zona")
    if not recs:
        recs.append("✅ Mantener monitoreo preventivo")
    return " | ".join(recs[:3])


def run_feature_engineering() -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    log.info("=" * 60)
    log.info("LUCÍA-MVD | FEATURE ENGINEERING")
    log.info("=" * 60)

    # Cargar datos
    vda      = pd.read_parquet(SYN_DIR / "vda.parquet")
    stm      = pd.read_parquet(SYN_DIR / "stm.parquet")
    equip    = pd.read_parquet(SYN_DIR / "equipamiento.parquet")
    censo    = pd.read_parquet(SYN_DIR / "censo.parquet")

    # Construir grilla
    grid = build_grid(cell_size_deg=0.006)

    # Calcular features
    log.info("Asignando VDA a grilla...")
    vda_agg  = assign_vda_to_grid(vda, grid)

    log.info("Calculando features de transporte...")
    transport = compute_transport_features(stm, grid)

    log.info("Asignando features sociales...")
    social   = compute_social_features(censo, grid)

    log.info("Calculando features de equipamiento...")
    equipment = compute_equipment_features(equip, grid)

    log.info("Calculando aislamiento espacial...")
    spatial  = compute_spatial_isolation(grid)

    # Unir todas las features
    log.info("Construyendo matriz de features...")
    df = build_feature_matrix(vda_agg, transport, social, equipment, spatial, grid)

    # Calcular score de riesgo
    log.info("Calculando score de riesgo...")
    df = compute_risk_score(df)

    # Agregar factores y recomendaciones
    log.info("Generando factores y recomendaciones...")
    df["top_factores"] = df.apply(top_3_factors, axis=1)
    df["recomendacion"] = df.apply(intervention_recommendation, axis=1)

    # Agregar info de barrio (join por cercanía centroide)
    from scipy.spatial import cKDTree as KDTree
    censo_coords = censo[["lat_centroide", "lon_centroide"]].values
    tree = KDTree(censo_coords)
    cell_coords = df[["lat_cen", "lon_cen"]].values
    _, indices = tree.query(cell_coords, k=1)
    df["barrio"] = censo.iloc[indices]["barrio"].values

    # Guardar
    out_path = PROC_DIR / "features_completo.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Dataset final guardado: {out_path}")
    log.info(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    run_feature_engineering()
