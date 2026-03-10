# LUCÍA-MVD — Metodología Detallada

## 1. Marco conceptual

### 1.1 Riesgo Invisible vs Riesgo Observado

El sistema distingue tres capas de riesgo:

**Riesgo Observado (IVO)**
Lo que aparece en los datos administrativos: denuncias, registros policiales, intervenciones documentadas. Es el piso empírico del análisis, pero subestima el riesgo real debido al subregistro estructural de la VBG.

**Riesgo Contextual**
Lo que el entorno urbano sugiere independientemente de las denuncias: aislamiento de transporte, falta de luminarias, vulnerabilidad socioeconómica, distancia a servicios de apoyo. Este componente es el núcleo de la innovación de LUCÍA.

**Riesgo Invisible (IRI)**
La brecha entre el riesgo contextual y las denuncias observadas. Zonas con IRI alto tienen condiciones urbanas desfavorables pero baja denuncia, lo que puede indicar tanto mayor vulnerabilidad como mayor subregistro.

```
IRI = max(0, Riesgo_Contextual - IVO_normalizado)
```

### 1.2 Hipótesis principal

> El riesgo para mujeres en el espacio urbano no depende únicamente del historial delictivo, sino de la interacción entre accesibilidad y frecuencia del transporte, estructura del espacio público, densidad y actividad urbana, distancia a puntos de apoyo, antecedentes de violencia y delitos, y vulnerabilidad social del entorno.

---

## 2. Unidad espacial de análisis

### 2.1 Grilla cuadrada

LUCÍA-MVD utiliza una grilla regular de celdas cuadradas de **~560m × 560m** (0.006° en latitud/longitud) sobre el área metropolitana de Montevideo.

**Justificación**:
- Evita los sesgos de polígonos administrativos irregulares (barrios muy grandes vs pequeños)
- Permite joins espaciales eficientes con KDTree
- Resolución apropiada para análisis urbano peatonal
- Facilita visualización en heatmap

**Alternativa evaluada y descartada**: H3 hexágonos (más elegante matemáticamente, pero requiere dependencia adicional y la diferencia analítica es mínima para esta escala).

### 2.2 Cobertura

- **Bounding box**: lat [-34.940, -34.820], lon [-56.300, -56.030]
- **Total celdas**: 900
- **Resolución**: 560m × 560m
- **CRS**: EPSG:4326 (WGS84)

---

## 3. Dimensiones del score

### 3.1 Índice de Violencia Observada (IVO) — peso 35%

**Inputs**:
- Denuncias VDA por zona censal/barrio (DNPG/MI)
- Delitos sexuales denunciados
- Historial de incidentes en radio de influencia

**Procesamiento**:
1. Join espacial de eventos → celda de grilla
2. Agregación por celda × franja horaria
3. Transformación log1p para suavizar outliers: `log(1 + n_denuncias)`
4. Normalización MinMax a [0,1]
5. Ajuste por multiplicador de franja horaria

**Multiplicadores temporales** (calibrados con literatura de criminología urbana y patrones de violencia doméstica):

| Franja | Multiplicador |
|--------|---------------|
| Mañana (6-12h) | 0.70 |
| Tarde (12-19h) | 1.00 |
| Noche (19-24h) | 1.45 |
| Madrugada (0-6h) | 1.30 |

**Fundamentación de los multiplicadores**: La violencia doméstica tiene picos nocturnos y de fin de semana (DNPG, 2023). La franja de madrugada tiene alta incidencia pero menor denominador de exposición. La tarde es la franja de referencia (mayor circulación peatonal y actividad).

### 3.2 Índice de Aislamiento de Transporte (IAT) — peso 25%

**Inputs**:
- Paradas STM en radio de 400m
- Frecuencia media de servicios (minutos entre unidades)
- Paradas activas en horario nocturno
- Número de líneas disponibles

**Cálculo del aislamiento nocturno** (componente principal del IAT):
```
Ais_frec = min(frec_nocturna / 60, 1.0)     # 60min = máximo aislamiento
Ais_paradas = max(0, 1 - n_paradas / 6)     # 6+ paradas = bien servido
Ais_nocturno = 0.6 × Ais_frec + 0.4 × Ais_paradas
```

**Ajuste por franja**:
- En franja nocturna/madrugada: se usa el IAT nocturno completo
- En franja diurna/tarde: se aplica un factor de reducción de 0.6 (menor relevancia del aislamiento nocturno)

### 3.3 Índice de Vulnerabilidad Social (IVS) — peso 20%

**Inputs**:
- % hogares con Necesidades Básicas Insatisfechas (INE)
- Tasa de desempleo barrial
- % informalidad laboral
- % jefatura femenina de hogar sin pareja (mayor exposición)

**Construcción**:
```
IVS ≈ vulnerability_base(barrio) calibrado con Censo 2011 + estimaciones 2023
```

El IVS se asigna a cada celda por proximidad al centroide del barrio más cercano (cKDTree k=1).

### 3.4 Índice de Aislamiento Espacial — peso 12%

**Cálculo**:
```
dist_centro = distancia euclidiana al centro de Montevideo (km)
Ais_espacial = min(dist_centro / 12.0, 1.0)
```

12km como referencia máxima (cubre el área periférica de Montevideo con datos disponibles).

**Justificación**: Las zonas periféricas combinan menor densidad de servicios, peor conectividad vial y mayor dependencia del transporte público.

### 3.5 Índice de Déficit de Equipamiento — peso 8%

**Inputs**:
- Luminarias funcionales en radio de 200m
- Distancia al refugio/centro de atención VBG más cercano (km)
- Distancia a seccional policial más cercana (km)

**Cálculo**:
```
Deficit_lum = max(0, 1 - n_luminarias / 5)
Deficit_ref = min(dist_refugio / 3.0, 1.0)    # 3km+ = máximo déficit
Deficit_pol = min(dist_policial / 2.0, 1.0)   # 2km+ = máximo déficit

IDeq = 0.40 × Deficit_lum + 0.35 × Deficit_ref + 0.25 × Deficit_pol
```

---

## 4. Modelo predictivo

### 4.1 Pipeline

```python
X = [IVO_adj, IAT_adj, IVS, Ais_espacial, IDeq, 
     n_paradas_400m, frec_nocturna_avg_min,
     n_luminarias_200m, dist_refugio_km, dist_comisaria_km,
     pct_hogares_nbi, tasa_desempleo, mult_franja]

y = nivel_riesgo ∈ {0: Bajo, 1: Medio, 2: Alto, 3: Crítico}
```

El target se construye desde el score compuesto:
```
Bajo     → score < 35
Medio    → 35 ≤ score < 55
Alto     → 55 ≤ score < 75
Crítico  → score ≥ 75
```

### 4.2 Parámetros XGBoost

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="mlogloss",
    random_state=42,
)
```

### 4.3 Validación

- Split: 80% train / 20% test, estratificado por clase
- Accuracy: 95.0%
- F1-Score Weighted: 95.0%

**Nota sobre las métricas altas**: La alta accuracy refleja que el modelo XGBoost captura eficientemente las relaciones entre el score compuesto (variable de construcción del target) y las features que lo componen. En producción con datos reales, se espera menor accuracy y mayor generalización real. El valor del modelo está en la **explicabilidad SHAP** y el **simulador de intervenciones**, no solo en la clasificación.

---

## 5. Simulador de intervenciones

El simulador calcula contrafactuales del modelo: qué nivel de riesgo se proyectaría si se modifican condiciones urbanas específicas.

**Metodología**:
1. Tomar vector de features de la celda seleccionada
2. Modificar los features de las intervenciones propuestas
3. Ejecutar `model.predict_proba()` sobre el vector modificado
4. Calcular reducción de riesgo como diferencia en valor esperado ponderado

```python
pesos_nivel = [0, 1, 2, 3]  # Bajo, Medio, Alto, Crítico
riesgo_actual = sum(prob_actual[i] × pesos_nivel[i])
riesgo_modif  = sum(prob_modif[i]  × pesos_nivel[i])
reduccion_pct = (riesgo_actual - riesgo_modif) / riesgo_actual × 100
```

**Limitación importante**: Este es un análisis de correlación, no causal. La reducción proyectada asume que las condiciones cambian ceteris paribus, lo que en contextos reales no siempre se cumple.

---

## 6. Consideraciones sobre el subregistro

La VBG está sistemáticamente subregistrada. En Uruguay, estudios indican que:
- Solo un porcentaje de las mujeres que sufren VBG realizan una denuncia formal
- El subregistro varía por zona, nivel socioeconómico y acceso a servicios

LUCÍA-MVD incorpora esta limitación de dos formas:
1. El IRI (Índice de Riesgo Invisible) captura el gap entre contexto y denuncia
2. El score compuesto no depende exclusivamente de las denuncias (solo 35%)

**Recomendación para versiones futuras**: Integrar encuestas de victimización (como la EVIO del Uruguay) para calibrar mejor el componente de violencia observada.

---

## 7. Glosario

| Término | Definición |
|---------|------------|
| VBG | Violencia Basada en Género |
| VDA | Violencia Doméstica y Asociados |
| DNPG | Dirección Nacional de Políticas de Género (MI Uruguay) |
| STM | Sistema de Transporte Metropolitano de Montevideo |
| IMM | Intendencia Municipal de Montevideo |
| NBI | Necesidades Básicas Insatisfechas |
| IVO | Índice de Violencia Observada |
| IAT | Índice de Aislamiento de Transporte |
| IVS | Índice de Vulnerabilidad Social |
| IRI | Índice de Riesgo Invisible |
| SHAP | SHapley Additive exPlanations (método de explicabilidad ML) |
