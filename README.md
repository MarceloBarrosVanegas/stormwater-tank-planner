# Stormwater Tank Planner

Framework para planificación de tanques de tormenta mediante optimización multi-objetivo acoplada a modelos hidráulicos SWMM e Itzi.

---

## Descripción

Este proyecto implementa un sistema de optimización para la ubicación y dimensionamiento de tanques de tormenta en redes de alcantarillado urbano. El sistema acopla modelado hidráulico 1D (SWMM) y simulación de inundaciones 2D (Itzi) con algoritmos de optimización evolutiva (NSGA-II) para minimizar costos de construcción y maximizar la reducción de daños por inundación.

---

## Tabla de Contenidos

- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Módulos](#módulos)
- [Configuración](#configuración)
- [Workflow](#workflow)
- [Resultados](#resultados)

---

## Arquitectura

```
Entradas                 Procesamiento                Salidas
─────────────────────────────────────────────────────────────────────
SWMM .inp    ──┐                                     Reportes Excel
DEM Raster   ──┼──▶  PathFinder  ──┐                  Mapas GeoPackage
Predios      ──┤                   ├──▶  Optimizer    Dashboards
Network      ──┘   SewerPipeline ──┤   (NSGA-II)  ──▶ Gráficas
                    SWMM Modifier ──┘   o Greedy
                    Itzi 2D
```

### Estructura de Módulos

**Optimización:**
- `rut_10_run_tanque_tormenta.py` - Orquestador principal del workflow
- `rut_15_optimizer.py` - Implementación Greedy y NSGA-II
- `rut_16_dynamic_evaluator.py` - Evaluador dinámico de soluciones
- `rut_23_nsga_optimizer.py` - NSGA-II para selección de tanques
- `rut_29_nsga_ranking_optimizer.py` - NSGA-II para optimización de pesos de ranking

**Modelado Hidráulico:**
- `rut_00_path_finder.py` - Enrutamiento sobre OpenStreetMap
- `rut_01_swmm_handel.py` - Manejo de archivos SWMM
- `rut_02_get_flodded_nodes.py` - Identificación de nodos inundados
- `rut_03_run_sewer_design.py` - Diseño de redes de alcantarillado
- `rut_06_pipe_sizing.py` - Dimensionamiento de tuberías
- `rut_27_model_metrics.py` - Extracción de métricas hidráulicas

**Evaluación de Daños:**
- `rut_18_itzi_flood_model.py` - Simulación 2D con Itzi/GRASS GIS
- `rut_19_flood_damage_climada.py` - Evaluación de daños con CLIMADA/JRC
- `rut_20_avoided_costs.py` - Cálculo de costos evitados (inversión diferida)
- `rut_21_construction_cost.py` - Cálculo de costos de construcción
- `rut_21_risk_analysis.py` - Análisis de riesgo probabilístico (EAD)
- `rut_26_hydrological_impact.py` - Evaluación de impacto hidrológico

**Utilidades:**
- `rut_14_swmm_modifier.py` - Modificación de archivos SWMM
- `rut_15_dashboard.py` - Generación de reportes Excel
- `rut_17_comparison_reporter.py` - Comparación de escenarios
- `rut_22_scenario_generator.py` - Generación de escenarios por período de retorno
- `rut_25_from_inp_to_vector.py` - Exportación INP a vectoriales
- `rut_28_water_quality.py` - Análisis de calidad de agua

**Configuración:**
- `config.py` - Parámetros globales del proyecto

---

## Requisitos

### Software
- Python 3.10+
- SWMM 5.2+
- GRASS GIS 8.4+ (para simulaciones Itzi)

### Dependencias Python
```
pyswmm>=2.0
swmmio>=0.7
pymoo>=0.6
geopandas>=0.14
rasterio>=1.3
osmnx>=1.6
networkx>=3.0
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
climada>=4.0
```

### Datos de Entrada
- Modelo SWMM (.inp)
- Modelo Digital de Elevaciones (.tif)
- Polígonos de predios candidatos (.gpkg)
- Geometría de red de alcantarillado (.gpkg)

---

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/MarceloBarrosVanegas/stormwater-tank-planner.git
cd stormwater-tank-planner

# Crear entorno conda
conda env create -f environment.yml
conda activate stormwater

# Verificar instalación
python -c "import config; print('OK')"
```

---

## Uso

### Ejemplo 1: Análisis Secuencial (Greedy)

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

runner = StormwaterOptimizationRunner(
    project_root=config.PROJECT_ROOT,
    eval_id="ejemplo_01"
)

result = runner.run_sequential_analysis(
    max_tanks=10,
    max_iterations=50,
    min_tank_vol=1000.0,
    max_tank_vol=50000.0,
    optimizer_mode='greedy',
    stop_at_breakeven=True
)
```

### Ejemplo 2: Optimización NSGA-II

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

config.TR_LIST = [1, 2, 5, 10, 25, 50, 100]

runner = StormwaterOptimizationRunner()
result = runner.run_sequential_analysis(
    max_tanks=20,
    optimizer_mode='nsga',
    optimization_tr_list=[1, 2, 5, 10, 25],
    validation_tr_list=[1, 2, 5, 10, 25, 50, 100]
)
```

### Ejemplo 3: Optimización de Pesos de Ranking

```python
from rut_29_nsga_ranking_optimizer import NSGARankingOptimizer

optimizer = NSGARankingOptimizer(elev_file=config.ELEV_FILE)
pareto_results = optimizer.optimize(
    n_generations=50,
    pop_size=24,
    max_tanks=15
)
```

---

## Módulos

### rut_10_run_tanque_tormenta.py

Orquestador principal. Coordina la ejecución del workflow completo de optimización.

```python
class StormwaterOptimizationRunner:
    def run_sequential_analysis(self, max_tanks, optimizer_mode, ...)
        """
        Ejecuta análisis secuencial.
        
        Parameters:
        -----------
        optimizer_mode : str
            'greedy' o 'nsga'
        max_tanks : int
            Número máximo de tanques
        stop_at_breakeven : bool
            Detener cuando costo >= beneficios
        """
```

### rut_16_dynamic_evaluator.py

Evaluador dinámico que ejecuta el pipeline completo para cada solución candidata:
1. PathFinder: Enrutamiento óptimo sobre OSM
2. SewerPipeline: Diseño hidráulico de tuberías
3. SWMMModifier: Modificación del archivo INP
4. SWMM: Simulación hidráulica
5. Métricas: Extracción de resultados

```python
class DynamicSolutionEvaluator:
    def __init__(self, path_proy, elev_files_list, proj_to, ...)
    def evaluate_solution(self, active_pairs, eval_id=None)
```

### rut_27_model_metrics.py

Extractor de métricas hidráulicas del sistema SWMM.

```python
class MetricExtractor:
    def extract_metrics(self) -> SystemMetrics
    def generate_candidate_pairs(self) -> List[CandidatePair]
    def rank_candidates(self, pairs: List[CandidatePair]) -> List[CandidatePair]
```

### rut_19_flood_damage_climada.py

Evaluación de daños por inundación usando CLIMADA con curvas de daño-profundidad JRC.

Funcionalidad:
- Mapea uso de suelo a sectores CLIMADA (residential, commercial, industrial, infrastructure, agriculture)
- Aplica curvas de daño JRC para Sudamérica
- Calcula daño total y por predio en USD

### rut_23_nsga_optimizer.py

Implementación de NSGA-II para selección óptima de tanques.

```python
class TankOptimizationProblem(Problem):
    """
    Variables: Array binario [0,1] para cada candidato
    Objetivos:
        1. Minimizar costo de construcción
        2. Minimizar daño residual (o EAD si probabilístico)
        3. Minimizar costo de reparación de tuberías
    Restricciones:
        - Máximo número de tanques
        - Capacidad de predios
    """
```

### rut_29_nsga_ranking_optimizer.py

Optimización de los pesos de ranking (`FLOODING_RANKING_WEIGHTS` y `CAPACITY_MAX_HD`) mediante NSGA-II. Cada evaluación ejecuta el workflow completo de optimización Greedy.

---

## Configuración

Parámetros principales en `config.py`:

### Parámetros de Tanque
```python
TANK_DEPTH_M = 15.0              # Profundidad (m)
TANK_MIN_VOLUME_M3 = 1000.0      # Volumen mínimo (m³)
TANK_MAX_VOLUME_M3 = 100000.0    # Volumen máximo (m³)
MAX_TANKS = 30                   # Máximo número de tanques
TANK_VOLUME_SAFETY_FACTOR = 1.05 # Factor de seguridad
```

### Pesos de Ranking
```python
FLOODING_RANKING_WEIGHTS = {
    'flow_over_capacity': 0.5,    # Caudal sobre capacidad
    'flow_node_flooding': 0.5,    # Caudal de inundación
    'vol_node_flooding': 0.0,     # Volumen de inundación
    'outfall_peak_flow': 0,       # Caudal pico outfall
    'failure_probability': 0,     # Probabilidad de falla
}
```

### NSGA-II
```python
NSGA_PARALLEL_WORKERS = 6        # Workers paralelos
SWMM_THREADS = 1                 # Threads por simulación SWMM
N_GENERATIONS = 50               # Generaciones NSGA-II
POP_SIZE = 24                    # Tamaño de población
```

### Componentes de Costo
```python
COST_COMPONENTS = {
    'deferred_investment': True,   # Inversión diferida (reposición tuberías)
    'flood_damage': False,          # Daño por inundación (CLIMADA)
    'river_damage': False,         # Impacto downstream
}
LAND_COST_PER_M2 = 50.0          # Costo de terreno ($/m²)
```

---

## Workflow

```
FASE 1: PREPARACIÓN
───────────────────
Cargar modelo SWMM base
Cargar predios y red
Calcular elevaciones y pendientes
Identificar nodos inundados (baseline)

FASE 2: RANKING DE CANDIDATOS
──────────────────────────────
Generar pares (nodo → predio)
Calcular distancias y desniveles
Aplicar pesos de ranking
Ordenar candidatos por score

FASE 3: OPTIMIZACIÓN
─────────────────────
Modo Greedy:
    1. Seleccionar mejor candidato
    2. Diseñar tubería (PathFinder)
    3. Dimensionar tubería
    4. Agregar tanque a SWMM
    5. Correr SWMM
    6. Evaluar resultados
    7. Si útil, agregar a solución
    8. Repetir

Modo NSGA-II:
    1. Generar población inicial
    2. Evaluar fitness (SWMM)
    3. Cruzamiento SBX
    4. Mutación polinomial
    5. Selección no dominada
    6. Elitismo
    7. Repetir hasta N generaciones

FASE 4: EVALUACIÓN
───────────────────
Comparar vs baseline
Calcular métricas:
    - Reducción de volumen inundado
    - Reducción de caudal pico
    - Health de la red
    - Costo total
Generar reportes y mapas

FASE 5: ANÁLISIS DE DAÑOS (opcional)
─────────────────────────────────────
Ejecutar Itzi 2D
Calcular daños con CLIMADA
Calcular EAD (Daño Anual Esperado)
Generar curvas de daño
```

---

## Resultados

Estructura de salida en `optimization_results/`:

```
optimization_results/
├── nsga_evaluations/
│   └── {eval_id}/
│       ├── best_solution/
│       ├── pareto_front.csv
│       └── evolution.png
├── sequential_analysis/
│   ├── iteration_001/
│   ├── iteration_002/
│   └── ...
├── comparison_reports/
│   └── comparison_baseline_vs_solution.xlsx
└── itzi_results/
    ├── max_water_depth.tif
    └── flood_damage_results.gpkg
```

### Métricas de Salida

| Métrica | Descripción | Unidad |
|---------|-------------|--------|
| flooding_vol_reduction | Reducción de volumen inundado | m³ |
| flooding_vol_reduction_pct | Reducción porcentual de volumen | % |
| flooding_peak_flow_reduction | Reducción de caudal pico de inundación | m³/s |
| outfall_peak_flow_reduction | Reducción de caudal en outfall | m³/s |
| network_health | Health promedio de la red | 0-1 |
| total_cost | Costo total de construcción | USD |
| bc_ratio | Relación Beneficio/Costo | - |

---

## Referencias

1. EPA. (2022). *Storm Water Management Model (SWMM) Version 5.2*. Environmental Protection Agency.
2. CLIMADA Project. *JRC Depth-Damage Curves for South America*.
3. Deb, K. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
4. Itzi. *Itzi - A 2D Flood Simulation Tool*. https://itzi.readthedocs.io/
5. OpenStreetMap Contributors. *OpenStreetMap Data*.

---

## Licencia

MIT License
