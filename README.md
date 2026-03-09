# Stormwater Tank Planner

Framework para planificación de tanques de tormenta mediante optimización multi-objetivo acoplada a modelos hidráulicos SWMM e Itzi.

---

## ¿Qué hace este proyecto?

Este sistema encuentra la ubicación y tamaño óptimos para tanques de tormenta en redes de alcantarillado. El proceso funciona así:

1. **Identifica problemas**: Localiza nodos de la red donde ocurre inundación
2. **Busca predios**: Encuentra terrenos disponibles cerca de los problemas
3. **Diseña tuberías**: Calcula la ruta y dimensionamiento de tuberías de conexión
4. **Simula**: Ejecuta modelos hidráulicos para verificar resultados
5. **Optimiza**: Usa algoritmos genéticos (NSGA-II) para balancear costos y beneficios

El sistema puede operar en dos modalidades:
- **Determinística**: Optimiza para un único evento de lluvia (ej: período de retorno 25 años)
- **Probabilística**: Optimiza considerando múltiples eventos y calcula el Daño Anual Esperado (EAD)

---

## Estructura del Proyecto

```
Entradas              Procesamiento              Salidas
────────────────────────────────────────────────────────────────
Modelo SWMM   ──┐                              Reportes Excel
DEM (elevación)─┼──▶  Enrutamiento  ──┐        Mapas
Predios        ──┤    Diseño tuberías  ├──▶  Optimización  ──▶ Resultados
Red existente  ──┘    Simulación SWMM  ──┘   (NSGA-II o    ──▶ Gráficas
                       Simulación 2D         Greedy)
```

### Módulos principales

#### Optimización
- `rut_10_run_tanque_tormenta.py` - Punto de entrada principal. Coordina todo el workflow.
- `rut_15_optimizer.py` - Implementa dos estrategias: Greedy (secuencial) y NSGA-II (evolutivo).
- `rut_16_dynamic_evaluator.py` - Ejecuta el pipeline completo para cada solución candidata.
- `rut_23_nsga_optimizer.py` - NSGA-II para selección de tanques.
- `rut_29_nsga_ranking_optimizer.py` - NSGA-II para calibrar pesos de ranking.

#### Modelado Hidráulico
- `rut_00_path_finder.py` - Encuentra rutas óptimas sobre calles (OpenStreetMap).
- `rut_02_get_flodded_nodes.py` - Identifica nodos inundados y genera candidatos.
- `rut_03_run_sewer_design.py` - Dimensiona tuberías según caudales y pendientes.
- `rut_06_pipe_sizing.py` - Cálculos hidráulicos de capacidad.
- `rut_27_model_metrics.py` - Extrae métricas de simulaciones SWMM.

#### Evaluación de Daños y Riesgo
- `rut_18_itzi_flood_model.py` - Simulación 2D de inundaciones superficiales.
- `rut_19_flood_damage_climada.py` - Calcula daños económicos usando curvas JRC/CLIMADA.
- `rut_20_avoided_costs.py` - Calcula costos evitados por no reponer tuberías.
- `rut_21_construction_cost.py` - Presupuesta construcción de tanques y tuberías.
- `rut_21_risk_analysis.py` - Análisis probabilístico con Bootstrap.
- `rut_26_hydrological_impact.py` - Evalúa impacto en outfalls.

#### Utilidades
- `rut_14_swmm_modifier.py` - Modifica archivos SWMM para agregar tanques.
- `rut_15_dashboard.py` - Genera reportes Excel con gráficas.
- `rut_17_comparison_reporter.py` - Compara escenarios base vs optimizado.
- `rut_22_scenario_generator.py` - Crea escenarios para diferentes períodos de retorno.
- `rut_25_from_inp_to_vector.py` - Exporta redes SWMM a GeoPackage.
- `rut_28_water_quality.py` - Análisis de calidad de agua (TSS, DBO).

#### Configuración
- `config.py` - Parámetros globales editables.

---

## Requisitos

### Software necesario
- Python 3.10 o superior
- SWMM 5.2 (EPA Storm Water Management Model)
- GRASS GIS 8.4 (solo si se usa Itzi para inundaciones 2D)

### Dependencias Python principales
```
pyswmm>=2.0        # Interfaz Python para SWMM
swmmio>=0.7        # Lectura/escritura de archivos SWMM
pymoo>=0.6         # Framework de optimización multi-objetivo
geopandas>=0.14    # Procesamiento de datos espaciales
rasterio>=1.3      # Manejo de rasters (DEM)
osmnx>=1.6         # Descarga de redes OpenStreetMap
networkx>=3.0      # Análisis de grafos
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
climada>=4.0       # Evaluación de daños por desastres
```

### Datos de entrada requeridos
- **Modelo SWMM** (.inp): Red de alcantarillado existente
- **DEM** (.tif): Modelo digital de elevaciones del terreno
- **Predios** (.gpkg): Polígonos de terrenos donde podrían ubicarse tanques
- **Red** (.gpkg): Geometría vectorial de la red de alcantarillado

---

## Instalación

Paso 1: Clonar el repositorio
```bash
git clone https://github.com/MarceloBarrosVanegas/stormwater-tank-planner.git
cd stormwater-tank-planner
```

Paso 2: Crear entorno conda
```bash
conda env create -f environment.yml
conda activate stormwater
```

Paso 3: Verificar que todo funcione
```bash
python -c "import config; print('Configuración cargada correctamente')"
```

Nota: El proyecto busca automáticamente el módulo PyPiper en rutas de OneDrive. Si está en otra ubicación, modificar `config.py`.

---

## Cómo usar

### Caso 1: Análisis rápido (modo Greedy)

Útil para obtener una primera aproximación. Agrega tanques uno por uno hasta alcanzar un criterio de parada.

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

# Inicializar
runner = StormwaterOptimizationRunner(
    project_root=config.PROJECT_ROOT,
    eval_id="analisis_rapido"
)

# Ejecutar
resultado = runner.run_sequential_analysis(
    max_tanks=10,              # Máximo 10 tanques
    optimizer_mode='greedy',   # Modo secuencial
    stop_at_breakeven=True     # Detenerse cuando costo = beneficio
)
```

### Caso 2: Optimización completa (modo NSGA-II)

Busca soluciones óptimas considerando múltiples objetivos simultáneamente. Más lento pero más exhaustivo.

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

# Configurar para análisis probabilístico (múltiples períodos de retorno)
config.TR_LIST = [1, 2, 5, 10, 25, 50, 100]

runner = StormwaterOptimizationRunner()

resultado = runner.run_sequential_analysis(
    max_tanks=20,
    optimizer_mode='nsga',
    optimization_tr_list=[1, 2, 5, 10, 25],      # TRs para optimizar
    validation_tr_list=[1, 2, 5, 10, 25, 50, 100] # TRs para validar
)
```

### Caso 3: Optimizar pesos de ranking

El sistema usa pesos para ranquear qué tanques son más importantes. Estos pesos pueden optimizarse automáticamente.

```python
from rut_29_nsga_ranking_optimizer import NSGARankingOptimizer

optimizer = NSGARankingOptimizer(elev_file=config.ELEV_FILE)

# Ejecutar optimización de pesos
resultado = optimizer.optimize(
    n_generations=50,
    pop_size=24,
    max_tanks=15
)
```

---

## Configuración importante

El archivo `config.py` contiene parámetros que controlan el comportamiento del sistema:

### Parámetros de tanque
```python
TANK_DEPTH_M = 15.0              # Profundidad de diseño (metros)
TANK_MIN_VOLUME_M3 = 1000.0      # Volumen mínimo permitido (m³)
TANK_MAX_VOLUME_M3 = 100000.0    # Volumen máximo permitido (m³)
MAX_TANKS = 30                   # Máximo número de tanques
TANK_VOLUME_SAFETY_FACTOR = 1.05 # Factor de seguridad (5%)
```

### Cómo se ranquean los candidatos
El sistema asigna puntajes a posibles ubicaciones de tanques usando estos pesos:

```python
FLOODING_RANKING_WEIGHTS = {
    'flow_over_capacity': 0.5,    # Caudal que excede capacidad de tubería
    'flow_node_flooding': 0.5,    # Caudal de inundación en el nodo
    'vol_node_flooding': 0.0,     # Volumen de inundación
    'outfall_peak_flow': 0,       # Contribución al pico del outfall
    'failure_probability': 0,     # Probabilidad de falla de tubería
}
# Suma debe ser 1.0. Valor 0 = no considera ese criterio.
```

### Configuración de NSGA-II
```python
NSGA_PARALLEL_WORKERS = 6        # Cuántas simulaciones en paralelo
SWMM_THREADS = 1                 # Threads por simulación SWMM
N_GENERATIONS = 50               # Generaciones del algoritmo
POP_SIZE = 24                    # Individuos por generación
```

### Qué costos considerar
```python
COST_COMPONENTS = {
    'deferred_investment': True,   # Costo de reponer tuberías más tarde
    'flood_damage': False,          # Daño a edificaciones (requiere CLIMADA)
    'river_damage': False,         # Impacto ambiental downstream
}

LAND_COST_PER_M2 = 50.0          # Costo de adquirir terrenos ($/m²)
```

---

## Proceso paso a paso

### Fase 1: Preparación
1. Cargar modelo SWMM base
2. Cargar predios disponibles y red de alcantarillado
3. Calcular elevaciones del terreno desde el DEM
4. Ejecutar simulación base para identificar problemas actuales

### Fase 2: Generar candidatos
1. Identificar nodos donde ocurre inundación
2. Encontrar predios cercanos a esos nodos
3. Calcular distancia y desnivel entre nodo y predio
4. Asignar puntaje según pesos configurados
5. Ordenar lista de candidatos

### Fase 3: Optimización

**Modo Greedy:**
- Toma el mejor candidato de la lista
- Diseña la tubería de conexión
- Agrega el tanque al modelo SWMM
- Corre la simulación
- Si mejora los resultados, mantiene el tanque
- Repite hasta cumplir criterio de parada

**Modo NSGA-II:**
- Genera población inicial de soluciones (combinaciones de tanques)
- Evalúa cada solución corriendo SWMM
- Selecciona las mejores soluciones (no dominadas)
- Cruza y muta para generar nueva población
- Repite por N generaciones

### Fase 4: Evaluación de resultados
- Compara escenario optimizado vs escenario base
- Calcula métricas de desempeño
- Genera mapas y reportes

### Fase 5: Análisis de daños (opcional)
- Ejecuta simulación 2D con Itzi para obtener profundidades de inundación
- Calcula daños económicos usando CLIMADA
- Calcula EAD para análisis probabilístico

---

## Resultados generados

El sistema crea los siguientes archivos en `optimization_results/`:

```
optimization_results/
├── nsga_evaluations/           # Resultados de optimización NSGA-II
│   └── {id_evaluacion}/
│       ├── best_solution/      # Mejor solución encontrada
│       │   ├── modified.inp    # Modelo SWMM modificado
│       │   ├── tanks.gpkg      # Geometría de tanques
│       │   └── report.xlsx     # Reporte detallado
│       ├── pareto_front.csv    # Conjunto de soluciones no dominadas
│       └── evolution.png       # Gráfico de convergencia
├── sequential_analysis/        # Resultados modo Greedy
│   ├── iteration_001/          # Iteración 1
│   ├── iteration_002/          # Iteración 2
│   └── ...
├── comparison_reports/         # Comparaciones base vs optimizado
│   ├── comparison.xlsx
│   └── maps/
└── itzi_results/               # Resultados de inundación 2D
    ├── max_water_depth.tif
    └── flood_damage_results.gpkg
```

### Métricas calculadas

| Métrica | Descripción | Unidad |
|---------|-------------|--------|
| flooding_vol_reduction | Cuánto se redujo el volumen inundado | m³ |
| flooding_vol_reduction_pct | Reducción porcentual | % |
| flooding_peak_flow_reduction | Reducción del caudal pico de inundación | m³/s |
| outfall_peak_flow_reduction | Reducción del caudal en puntos de descarga | m³/s |
| network_health | Indicador de salud de la red (0-1) | - |
| total_cost | Suma de costos de construcción | USD |
| bc_ratio | Relación beneficio/costo | - |

---

## Referencias

1. EPA. (2022). *Storm Water Management Model (SWMM) Version 5.2*. Environmental Protection Agency.
2. CLIMADA Project. *JRC Depth-Damage Curves for South America*. https://www.wcr.ethz.ch/research/climada.html
3. Deb, K. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
4. Itzi. *Itzi - A 2D Flood Simulation Tool*. https://itzi.readthedocs.io/
5. OpenStreetMap Contributors. *OpenStreetMap Data*. https://www.openstreetmap.org/

---

## Licencia

MIT License
