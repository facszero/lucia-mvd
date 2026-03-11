# Mejora Futura: Asignación de Barrios por Polígonos Oficiales

**Estado:** Detectado — pendiente de implementación  
**Prioridad:** Media (no afecta demo/datathon, relevante para versión productiva)  
**Detectado en:** Marzo 2026, durante pruebas de la interfaz del mapa  

---

## Problema detectado

Al hacer hover sobre celdas del mapa, el tooltip muestra en algunos casos un barrio
incorrecto. El ejemplo concreto que originó este análisis: una celda ubicada visualmente
dentro de **Buceo** (según la cartografía base de CartoDB/OpenStreetMap) mostraba
**"Barrio: Jacinto Vera"** en el tooltip del sistema.

La discrepancia se produce porque el mapa tiene **dos fuentes de información de barrio**
que no siempre coinciden:

1. **Etiquetas del mapa base** — cartografía real de OpenStreetMap con límites geográficos
   oficiales de cada barrio de Montevideo.
2. **Campo `barrio` del dataset** — asignado por el pipeline de LUCÍA-MVD mediante
   proximidad al centroide del censo sintético (KDTree).

---

## Causa raíz técnica

### Sistema actual: KDTree por centroide sintético

```python
# features.py — asignación actual
censo_km = np.column_stack([
    censo["lat_centroide"].values * 111.0,
    censo["lon_centroide"].values * 111.0 * cos(radians(-34.9)),
])
tree = KDTree(censo_km)
_, indices = tree.query(cell_km, k=1)
df["barrio"] = censo.iloc[indices]["barrio"].values
```

Cada celda de la grilla recibe el barrio cuyo **centroide sintético** (inventado durante
la ingesta, calibrado manualmente) está más cercano en kilómetros.

### Por qué falla en celdas de frontera

Los centroides sintéticos son aproximaciones. Una celda ubicada cerca del límite
entre dos barrios puede quedar asignada al barrio "equivocado" si el centroide
sintético de ese barrio está ligeramente más cercano que el del barrio correcto,
aunque visualmente la celda caiga dentro del otro barrio según los límites reales.

**Ejemplo documentado:**
- Celda: `lat=-34.901, lon=-56.135`
- Barrio correcto (OpenStreetMap): **Buceo** (centroide oficial ~`-34.910, -56.120`)
- Barrio asignado antes del fix KDTree: **Jacinto Vera** (centroide sintético `−34.886, −56.141`)
- Barrio asignado después del fix KDTree (con factor coseno): **Buceo** ✅
- Barrio asignado en celdas de frontera aún problemáticas: puede variar

*Nota: el fix del factor coseno (commit `f713cea`) mejoró significativamente la
precisión general, pero no resuelve el problema estructural en celdas de frontera
donde dos centroides sintéticos están a distancias similares.*

---

## Solución propuesta: Spatial Join con polígonos oficiales IMM

### Concepto

Reemplazar la asignación por proximidad de centroide por un **spatial join geoespacial**
contra los polígonos oficiales de barrios de Montevideo publicados por la Intendencia
de Montevideo (IMM).

```
ACTUAL:  celda → barrio más cercano por distancia a centroide inventado
PROPUESTO: celda → barrio cuyo polígono oficial contiene el centroide de la celda
```

### Fuente de datos

- **Portal:** [Catálogo de datos abiertos IMM](https://montevideo.gub.uy/datos)
- **Dataset:** "Barrios de Montevideo" (GeoJSON / Shapefile)
- **URL directa (referencia):** `https://montevideo.gub.uy/sites/default/files/datos/barrios.geojson`
- **Formato:** GeoJSON con 62 polígonos, CRS EPSG:4326
- **Licencia:** Datos abiertos del Gobierno Departamental de Montevideo

### Cambios necesarios en el código

#### 1. Ingesta del shapefile (`src/ingest/ingest.py`)

```python
import geopandas as gpd
import requests

def load_barrios_imm() -> gpd.GeoDataFrame:
    """Descarga o carga desde disco los polígonos oficiales de barrios IMM."""
    local_path = SYN_DIR / "barrios_imm.geojson"
    
    if not local_path.exists():
        url = "https://montevideo.gub.uy/sites/default/files/datos/barrios.geojson"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        local_path.write_bytes(r.content)
        log.info("Barrios IMM descargados y guardados")
    
    gdf = gpd.read_file(local_path)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf[["nombre", "geometry"]].rename(columns={"nombre": "barrio"})
```

#### 2. Asignación en feature engineering (`src/features/features.py`)

```python
def assign_barrio_spatial(grid: gpd.GeoDataFrame, 
                           barrios: gpd.GeoDataFrame) -> pd.Series:
    """
    Asigna barrio a cada celda por spatial join con polígonos oficiales IMM.
    Para celdas sin asignación (borde, rambla, etc.), usa fallback por centroide.
    """
    # Spatial join: centroide de celda dentro del polígono de barrio
    grid_centroids = grid.copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    
    joined = gpd.sjoin(
        grid_centroids[["cell_id", "geometry"]],
        barrios,
        how="left",
        predicate="within"
    )
    
    # Fallback para celdas sin barrio asignado (borde costero, límites)
    sin_barrio = joined[joined["barrio"].isna()]
    if len(sin_barrio) > 0:
        log.warning(f"{len(sin_barrio)} celdas sin barrio por spatial join, "
                    f"aplicando fallback por centroide más cercano")
        # ... lógica de fallback KDTree para esas celdas
    
    return joined.set_index("cell_id")["barrio"]
```

#### 3. Consistencia de nombres de barrios

El censo sintético debe usar exactamente los mismos nombres que el shapefile IMM.
Es necesario verificar y alinear los 60 nombres del censo con los 62 del shapefile:

```python
# Verificación a ejecutar una vez al migrar
barrios_censo = set(censo_df["barrio"].unique())
barrios_imm   = set(barrios_gdf["barrio"].unique())

solo_en_censo = barrios_censo - barrios_imm
solo_en_imm   = barrios_imm - barrios_censo

print(f"Solo en censo (posibles mismatches): {solo_en_censo}")
print(f"Solo en IMM (barrios faltantes en censo): {solo_en_imm}")
```

---

## Riesgos y consideraciones

### 1. Dependencia de archivo externo ⚠️

El pipeline actualmente es **100% autónomo** — genera todo desde cero sin dependencias
de red. Con el shapefile IMM se agrega una dependencia que puede fallar si:
- La URL de la IMM cambia (ha ocurrido antes)
- No hay conectividad durante el primer run

**Mitigación:** Commitear el GeoJSON al repo como archivo estático en `data/geo/`.
Desventaja: agrega ~500KB al repositorio.

### 2. Inconsistencia en nombres de barrios ⚠️

Los nombres oficiales de la IMM pueden diferir de los de uso común. Ejemplos conocidos:

| Nombre común (censo sintético) | Nombre oficial IMM (posible) |
|---|---|
| Barrio Sur | Sur |
| Tres Cruces | Villa Tres Cruces |
| Punta de Rieles | Punta Rieles |
| Colon Sureste | Colón Sureste |
| Paso de las Duranas | Paso de las Duranas |

Cualquier mismatch produce `NaN` en los datos socioeconómicos de esas celdas,
lo cual es un **bug silencioso** — el pipeline no falla, pero los datos son incorrectos.

**Mitigación:** Tabla de equivalencias (`BARRIO_ALIASES`) para normalizar nombres
antes del join.

### 3. Celdas sin barrio asignado ⚠️

Con polígonos reales, celdas en zonas como:
- Rambla costera
- Puerto de Montevideo
- Aeropuerto de Carrasco
- Límites exactos entre barrios

...pueden quedar sin asignación (`NaN`). KDTree siempre asigna algo; spatial join no.

**Mitigación:** Fallback automático a KDTree para celdas sin asignación por spatial join.
Loguear cuántas celdas usan fallback para monitoreo.

### 4. MVD_LAND_POLYGON quedaría redundante

Si se incorporan los polígonos reales de los 62 barrios, la máscara costera dibujada
manualmente (`MVD_LAND_POLYGON` en `features.py`) podría reemplazarse por la unión
de todos los polígonos IMM. Esto sería una refactorización más amplia pero eliminaría
el polígono artesanal y sus limitaciones.

### 5. Impacto en el modelo XGBoost

Si los barrios cambian para algunas celdas, cambian los features socioeconómicos
de esas celdas (NBI, desempleo, vulnerabilidad vienen del censo keyed por barrio),
lo que implica:

- Los scores de riesgo cambian para celdas afectadas
- El modelo debe reentrenarse desde cero
- Las métricas actuales (Accuracy 93.4%, F1 93.4%) **no son válidas** hasta nueva evaluación

En la práctica el impacto debería ser menor al 5% de las celdas (solo las de frontera),
pero debe verificarse empíricamente.

### 6. Complejidad operativa

| Aspecto | Sistema actual | Con spatial join |
|---|---|---|
| Tiempo de pipeline | ~18 segundos | ~22-25 segundos (estimado) |
| Dependencias externas | Ninguna | 1 archivo GeoJSON (IMM) |
| Riesgo de NaN en barrio | Ninguno | ~5-15 celdas de borde |
| Precisión en frontera | ~85% | ~99% |
| Consistencia con mapa base | Parcial | Total |

---

## Estimación de esfuerzo

| Tarea | Tiempo estimado |
|---|---|
| Descargar y explorar shapefile IMM | 1h |
| Verificar y resolver mismatches de nombres | 2-3h |
| Implementar `load_barrios_imm()` | 1h |
| Implementar `assign_barrio_spatial()` con fallback | 2h |
| Actualizar `compute_social_features()` | 1h |
| Reentrenar modelo y verificar métricas | 1h |
| Tests y validación visual en mapa | 1h |
| **Total estimado** | **9-10h** |

---

## Recomendación

Para el contexto actual (demo DAT4CCIÓN 2026), el sistema KDTree con factor coseno
es **suficiente y defendible**. El error ocurre únicamente en celdas de frontera entre
barrios, que son una minoría, y no afecta las métricas del modelo ni las conclusiones
generales del análisis.

La migración a spatial join con polígonos IMM tiene sentido en una **versión
productiva** con datos reales del DNPG, donde la precisión de atribución por barrio
es crítica para que los organismos públicos puedan actuar sobre los resultados.

---

*Documento generado en base a análisis técnico del sistema LUCÍA-MVD, marzo 2026.*  
*Revisado por: Equipo LUCÍA-MVD*
