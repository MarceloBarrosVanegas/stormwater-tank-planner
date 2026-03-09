# Stormwater Tank Planner

Framework de optimización multi-objetivo para la planificación de tanques de tormenta en redes de alcantarillado urbano, con acoplamiento de modelos hidrodinámicos 1D (SWMM) y 2D (Itzi), y evaluación probabilística de riesgo mediante CLIMADA/JRC.

---

## Resumen Ejecutivo

Este proyecto implementa una metodología de optimización anidada para determinar la ubicación, número y volumen óptimos de tanques de tormenta en el subsistema de alcantarillado El Colegio Occidental (Quito). El sistema acopla:

- **Modelo 1D**: SWMM 5.2 con ecuaciones de Saint-Venant en onda dinámica para la red de alcantarillado
- **Modelo 2D**: Itzi con Shallow Water Equations en aproximación de onda difusiva para inundación superficial
- **Optimización**: Esquema de tres niveles (NSGA-II + Greedy Secuencial + Evaluación iterativa)
- **Evaluación de riesgo**: Curvas de vulnerabilidad JRC/CLIMADA para cálculo de Daño Anual Esperado (EAD)

---

## Descripción del Área de Estudio

El subsistema El Colegio Occidental está ubicado en los sectores centro norte y norte de Quito, en las laderas occidentales del Pichincha. La cuenca comprende:

| Subcuenca | Área (Ha) | Descripción |
|-----------|-----------|-------------|
| Cuenca urbana principal | 1,683 | Área principal del sistema |
| Quebrada Rumihurco | 942 | Tributaria occidental |
| Quebrada San Lorenzo | 457 | Tributaria oriental |
| Quebrada San Antonio | 220 | Tributaria norte |
| Quebrada San Carlos | 209 | Tributaria sur |

La red principal alivia los colectores San Carlos, Atucucho, Flavio Alfaro, El Colegio, Sabanilla y San Lorenzo, descargando hacia el río Monjas con un caudal pico estimado de 69.41 m³/s para un período de retorno de 25 años.

---

## Arquitectura de Optimización

El sistema implementa un esquema anidado de tres niveles para evitar rankings estáticos: la idoneidad de un tanque depende del estado del sistema, que cambia tras cada intervención.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ARQUITECTURA DE OPTIMIZACIÓN                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  NIVEL 1: META-OPTIMIZACIÓN (NSGA-II)                                           │
│  ─────────────────────────────────────                                          │
│  Optimiza los pesos w que gobiernan la función de priorización:                 │
│                                                                                  │
│      w = [w_flow_over_capacity, w_flow_node_flooding, w_vol_node_flooding,      │
│            w_outfall_peak_flow, w_failure_probability]                          │
│                                                                                  │
│  Restricciones: Σw_k = 1, w_k ≥ 0                                               │
│                                                                                  │
│  Vector de objetivos F(w):                                                      │
│  • ΔV_flood (%) ↑        Reducción de volumen de inundación                     │
│  • ΔQ_flood (%) ↑        Reducción de caudal de desbordamiento                  │
│  • ΔQ_outfall (%) ↑      Reducción de caudal pico en outfalls                   │
│  • H_network ↓           Minimización de h/D en conductos                       │
│  • C_social (USD) ↓      Costo social neto                                      │
│                                                                                  │
│           ↓ Cada individuo (configuración de pesos)                             │
│                                                                                  │
│  NIVEL 2: SELECCIÓN SECUENCIAL (Greedy)                                         │
│  ──────────────────────────────────────                                         │
│  Construye solución incremental de tanques:                                     │
│                                                                                  │
│  while criterio de parada no alcanzado:                                         │
│      1. Calcular ranking de nodos según pesos w                                 │
│      2. Seleccionar nodo de mayor puntaje                                       │
│      3. Encontrar predio óptimo (radio de búsqueda)                             │
│      4. Diseñar tubería de conexión (Dijkstra sobre OSM)                        │
│      5. Dimensionar derivación (régimen permanente)                             │
│      6. Ejecutar simulación 1D-2D iterativa                                     │
│      7. Extraer métricas (costos, volúmenes, caudales)                          │
│      8. Actualizar ranking con métricas recalculadas                            │
│                                                                                  │
│           ↓ Retorna objetivos F(w) al Nivel 1                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Componentes del Pipeline de Evaluación (Nivel 2)

Para cada tanque candidato, el sistema ejecuta:

1. **PathFinder** (`rut_00_path_finder.py`): Algoritmo de Dijkstra sobre grafo de OpenStreetMap ponderado por:
   - Longitud geométrica
   - Penalización por desnivel adverso (flujo gravitacional)
   - Factor de ajuste según tipo de vía

2. **SewerPipeline** (`rut_03_run_sewer_design.py`): Dimensionamiento hidráulico en régimen permanente considerando:
   - Velocidad mínima de auto-limpieza
   - Velocidad máxima para protección del conducto
   - Capacidad máxima h/D especificada

3. **SWMM Modifier** (`rut_14_swmm_modifier.py`): Actualización del modelo con:
   - Nuevo nodo de tanque
   - Conexión de derivación
   - Volumen de almacenamiento
   - Cota de vertedero

4. **Simulación 1D-2D** (`rut_18_itzi_flood_model.py`): Ejecución acoplada SWMM-Itzi

5. **Extracción de métricas** (`rut_27_model_metrics.py`): Cálculo de indicadores de desempeño

---

## Modelación Hidrodinámica Acoplada 1D-2D

### Modelo 1D: Red de Alcantarillado (SWMM)

Resuelve las ecuaciones completas de Saint-Venant (onda dinámica) para flujo transitorio:

**Conservación de masa:**
```
∂A/∂t + ∂Q/∂x = q_ℓ
```

**Conservación de momento:**
```
∂Q/∂t + ∂(Q²/A)/∂x + gA·∂H/∂x + gA·S_f + gA·S_m = 0
```

Donde:
- Q = caudal (m³/s)
- A = área hidráulica (m²)
- H = z + y = carga hidráulica (m)
- S_f = n²Q|Q| / (A²R_h^(4/3)) = pendiente de fricción (Manning)
- S_m = pérdidas locales

Permite representar remanso, inversión de flujo y transiciones entre régimen libre y presurizado.

### Modelo 2D: Escorrentía Superficial (Itzi)

Resuelve las Shallow Water Equations en aproximación de onda difusiva (esquema de inercia parcial):

**Conservación de masa:**
```
∂h/∂t + ∂(uh)/∂x + ∂(vh)/∂y = R - I + q_exc
```

**Conservación de momento (balance gravitación-fricción):**
```
∂(z+h)/∂x + S_f,x = 0
∂(z+h)/∂y + S_f,y = 0
```

Donde:
- h = profundidad de agua (m)
- (u,v) = velocidades promedio en profundidad (m/s)
- R, I = tasas de precipitación e infiltración (m/s)
- q_exc = término de intercambio 1D-2D

### Acoplamiento Bidireccional

El intercambio en pozos de inspección se calcula mediante relación tipo orificio:

```
Q_exc,i = C_d · A_mh,i · √(2g|ΔH_i|) · sgn(ΔH_i)
```

Donde:
- ΔH_i = H_1D,i - H_2D,i = diferencia de carga
- C_d = coeficiente de descarga
- A_mh,i = área hidráulica efectiva del pozo

**Condiciones de intercambio:**
- Si H_1D > H_2D y H_1D > z_i: surgencia (red → superficie)
- Si H_2D > H_1D y H_2D > z_i: drenaje (superficie → red)

---

## Evaluación Probabilística del Riesgo

### Generación de Escenarios de Amenaza

Para cada período de retorno T_r ∈ {1, 2, 5, 10, 25, 50, 100} años:

1. **Ecuación IDF**: Intensidad-Duración-Frecuencia calibrada para Quito
2. **Método de Bloques Alternos**: Distribución temporal del hietograma
   - Duración: 60 minutos
   - Paso temporal: 5 minutos (12 bloques)
   - Bloque de mayor intensidad centrado

### Funciones de Vulnerabilidad (JRC/CLIMADA)

El daño a edificaciones se cuantifica mediante la Relación Media de Daño (MDR):

```
MDR_s(h): [0, ∞) → [0, 1]
```

Donde s indica el sector económico:

| Sector | Curva | Origen |
|--------|-------|--------|
| Residencial | JRC South America | Huizinga et al., 2017 |
| Comercial | JRC South America | Huizinga et al., 2017 |
| Industrial | JRC South America | Huizinga et al., 2017 |
| Infraestructura | Derivada | Metodología JRC/FEMA |
| Agricultura | Derivada | Metodología JRC/FAO |

### Cálculo del Daño por Evento

Para un período de retorno T_r:

```
D_edif(T_r) = Σ_i∈P(T_r) V_i · MDR_s_i(h_i)
```

Donde:
- V_i = valor de construcción del predio i (excluye terreno)
- h_i = profundidad de inundación extraída del ráster ITZI
- s_i = sector económico del predio

### Daño Anual Esperado (EAD)

Integración probabilística sobre todos los períodos de retorno:

```
EAD = ∫ D(T_r) · f(T_r) dT_r
```

Aproximado numéricamente mediante integración trapezoidal sobre los escenarios discretos.

---

## Estructura del Código

### Módulos de Optimización
- `rut_10_run_tanque_tormenta.py` - Orquestador principal
- `rut_15_optimizer.py` - Greedy y NSGA-II
- `rut_16_dynamic_evaluator.py` - Evaluador dinámico de soluciones
- `rut_23_nsga_optimizer.py` - NSGA-II para selección de tanques
- `rut_29_nsga_ranking_optimizer.py` - NSGA-II para pesos de ranking

### Módulos de Modelado Hidráulico
- `rut_00_path_finder.py` - Enrutamiento Dijkstra sobre OSM
- `rut_02_get_flodded_nodes.py` - Identificación de nodos críticos
- `rut_03_run_sewer_design.py` - Diseño hidráulico de tuberías
- `rut_06_pipe_sizing.py` - Cálculos de capacidad
- `rut_27_model_metrics.py` - Extracción de métricas SWMM

### Módulos de Daños y Riesgo
- `rut_18_itzi_flood_model.py` - Simulación 2D con Itzi
- `rut_19_flood_damage_climada.py` - Evaluación CLIMADA/JRC
- `rut_20_avoided_costs.py` - Costos de reposición diferida
- `rut_21_construction_cost.py` - Costos de construcción
- `rut_21_risk_analysis.py` - Análisis probabilístico EAD
- `rut_26_hydrological_impact.py` - Impacto en outfalls

### Módulos de Utilidad
- `rut_14_swmm_modifier.py` - Modificación de archivos SWMM
- `rut_15_dashboard.py` - Reportes Excel
- `rut_17_comparison_reporter.py` - Comparación de escenarios
- `rut_22_scenario_generator.py` - Generación de escenarios TR
- `rut_25_from_inp_to_vector.py` - Exportación a vectoriales
- `rut_28_water_quality.py` - Análisis de TSS y DBO

### Configuración
- `config.py` - Parámetros globales

---

## Requisitos

### Software
- Python 3.10+
- SWMM 5.2
- GRASS GIS 8.4 (para Itzi)

### Dependencias Principales
```
pyswmm>=2.0
swmmio>=0.7
pymoo>=0.6
geopandas>=0.14
rasterio>=1.3
osmnx>=1.6
climada>=4.0
```

### Datos de Entrada
- Modelo SWMM (.inp)
- DEM del terreno (.tif)
- Catastro de predios (.gpkg)
- Red vial OpenStreetMap

---

## Uso

### Ejemplo 1: Optimización Completa

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

# Configurar para análisis probabilístico
config.TR_LIST = [1, 2, 5, 10, 25, 50, 100]

runner = StormwaterOptimizationRunner(
    project_root=config.PROJECT_ROOT,
    eval_id="ejemplo_probabilistico"
)

resultado = runner.run_sequential_analysis(
    max_tanks=20,
    optimizer_mode='nsga',
    optimization_tr_list=[1, 2, 5, 10, 25],
    validation_tr_list=[1, 2, 5, 10, 25, 50, 100]
)
```

### Ejemplo 2: Análisis Rápido (Greedy)

```python
runner = StormwaterOptimizationRunner()

resultado = runner.run_sequential_analysis(
    max_tanks=10,
    optimizer_mode='greedy',
    stop_at_breakeven=True
)
```

### Ejemplo 3: Meta-optimización de Pesos

```python
from rut_29_nsga_ranking_optimizer import NSGARankingOptimizer

optimizer = NSGARankingOptimizer(elev_file=config.ELEV_FILE)

resultado = optimizer.optimize(
    n_generations=50,
    pop_size=24,
    max_tanks=15
)
```

---

## Configuración Clave

### Parámetros de Tanque
```python
TANK_DEPTH_M = 15.0              # Profundidad (m)
TANK_MIN_VOLUME_M3 = 1000.0      # Volumen mínimo (m³)
TANK_MAX_VOLUME_M3 = 100000.0    # Volumen máximo (m³)
MAX_TANKS = 30                   # Máximo número de tanques
```

### Pesos de Priorización
```python
FLOODING_RANKING_WEIGHTS = {
    'flow_over_capacity': 0.5,    # Exceso de capacidad
    'flow_node_flooding': 0.5,    # Caudal de inundación
    'vol_node_flooding': 0.0,     # Volumen de inundación
    'outfall_peak_flow': 0,       # Caudal pico outfall
    'failure_probability': 0,     # Probabilidad de falla
}
```

### NSGA-II
```python
NSGA_PARALLEL_WORKERS = 6        # Workers paralelos
SWMM_THREADS = 1                 # Threads por simulación
N_GENERATIONS = 50               # Generaciones
POP_SIZE = 24                    # Tamaño de población
```

---

## Resultados

El sistema genera en `optimization_results/`:

```
optimization_results/
├── nsga_evaluations/{id}/
│   ├── best_solution/           # Mejor solución
│   │   ├── modified.inp         # Modelo SWMM modificado
│   │   ├── tanks.gpkg           # Ubicación de tanques
│   │   └── report.xlsx          # Reporte detallado
│   ├── pareto_front.csv         # Frente de Pareto
│   └── evolution.png            # Convergencia
├── sequential_analysis/
│   ├── iteration_001/
│   ├── iteration_002/
│   └── ...
└── itzi_results/
    ├── max_water_depth.tif      # Profundidad máxima
    └── flood_damage_results.gpkg # Daños por predio
```

### Métricas de Salida

| Métrica | Descripción | Unidad |
|---------|-------------|--------|
| ΔV_flood | Reducción de volumen inundado | m³ |
| ΔV_flood_pct | Reducción porcentual | % |
| ΔQ_flood | Reducción de caudal de desbordamiento | m³/s |
| ΔQ_outfall | Reducción de caudal pico en outfall | m³/s |
| H_network | Health promedio de la red (h/D) | 0-1 |
| C_social | Costo social neto | USD |
| EAD | Daño Anual Esperado | USD/año |

---

## Referencias

1. EPA. (2022). *Storm Water Management Model (SWMM) Version 5.2*. Environmental Protection Agency.

2. Huizinga, J., Moel, H. de, & Szewczyk, W. (2017). *Global flood depth-damage functions: Methodology and the database with guidelines*. EU-JRC Technical Report.

3. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

4. Itzi. *Itzi - A 2D Flood Simulation Tool*. https://itzi.readthedocs.io/

5. Aznar, B., & Barnadas, M. (2016). *Itzi (version 17.1) [software]*.

6. OpenStreetMap Contributors. *OpenStreetMap Data*. https://www.openstreetmap.org/

---

## Licencia

MIT License
