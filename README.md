# Stormwater Tank Planner 🌧️

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SWMM](https://img.shields.io/badge/SWMM-5.2+-green.svg)](https://www.epa.gov/water-research/storm-water-management-model-swmm)
[![GRASS GIS](https://img.shields.io/badge/GRASS%20GIS-8.4+-orange.svg)](https://grass.osgeo.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **Sistema avanzado de optimización multi-objetivo para la planificación de tanques de tormenta en redes de alcantarillado urbano**

Este proyecto implementa un framework completo para la planificación óptima de tanques de tormenta (stormwater detention tanks) utilizando modelado hidráulico SWMM, simulación 2D de inundaciones con Itzi, y algoritmos de optimización NSGA-II. El sistema permite minimizar costos de construcción mientras maximiza la reducción de daños por inundación.

---

## 📋 Tabla de Contenidos

- [Características Principales](#-características-principales)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Módulos Principales](#-módulos-principales)
- [Configuración](#-configuración)
- [Workflow de Optimización](#-workflow-de-optimización)
- [Resultados](#-resultados)
- [Contribución](#-contribución)

---

## ✨ Características Principales

### 🎯 Optimización Multi-Objetivo
- **NSGA-II Algorithm**: Optimización evolutiva con pymoo para balancear costos vs beneficios
- **Modo Greedy**: Algoritmo voraz secuencial para análisis rápido de retornos decrecientes
- **Optimización de Pesos**: Ajuste automático de pesos de ranking mediante NSGA-II (rut_29)

### 🌊 Modelado Hidráulico Integrado
- **SWMM 5.2+**: Modelado 1D de redes de alcantarillado
- **Itzi 2D**: Simulación de inundaciones superficiales con GRASS GIS
- **Análisis Probabilístico**: Cálculo de Daño Anual Esperado (EAD) con múltiples períodos de retorno

### 💰 Evaluación Económica Completa
- **Costos de Construcción**: Cálculo detallado de tuberías, tanques y terrenos (rut_21)
- **Daños por Inundación**: Integración con CLIMADA y curvas JRC de daño-profundidad (rut_19)
- **Análisis de Riesgo**: Evaluación probabilística con bootstrap y mapas de riesgo espacial (rut_21)
- **Costos Evitados**: Cálculo de inversión diferida en reposición de tuberías (rut_20)

### 🗺️ Análisis Espacial Avanzado
- **PathFinder**: Enrutamiento óptimo sobre OpenStreetMap considerando elevación y distancia (rut_00)
- **Diseño de Tuberías**: Dimensionamiento hidráulico automático con cálculo de capacidad (rut_03, rut_06)
- **Selección de Predios**: Evaluación de predios candidatos con análisis de pendiente y capacidad

### 📊 Visualización y Reportes
- **Dashboard Interactivo**: Generación de reportes Excel con gráficos comparativos (rut_15)
- **Mapas de Comparación**: Visualización espacial de escenarios base vs optimizados (rut_17)
- **Curvas de Daño**: Gráficos de vulnerabilidad y pérdida esperada (rut_20)

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STORMWATER TANK PLANNER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   INPUTS     │───▶│  PROCESSING  │───▶│   OUTPUTS    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ • SWMM .inp  │    │ • PathFinder │    │ • Reportes   │                   │
│  │ • DEM Raster │    │ • SewerDesign│    │ • Mapas      │                   │
│  │ • Predios    │    │ • SWMM Mod.  │    │ • Dashboards │                   │
│  │ • Network    │    │ • NSGA-II    │    │ • Gráficas   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         MÓDULOS PRINCIPALES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Núcleo de Optimización:                                                     │
│  ├── rut_10_run_tanque_tormenta.py   # Orquestador principal                 │
│  ├── rut_15_optimizer.py             # Optimizador Greedy + NSGA-II          │
│  ├── rut_16_dynamic_evaluator.py     # Evaluador dinámico de soluciones      │
│  ├── rut_23_nsga_optimizer.py        # NSGA-II para selección de tanques     │
│  └── rut_29_nsga_ranking_optimizer.py# NSGA-II para optimización de pesos    │
│                                                                              │
│  Análisis y Modelado:                                                        │
│  ├── rut_00_path_finder.py           # Enrutamiento sobre OSM                │
│  ├── rut_01_swmm_handel.py           # Manejo de archivos SWMM               │
│  ├── rut_02_get_flodded_nodes.py     # Identificación de nodos inundados     │
│  ├── rut_03_run_sewer_design.py      # Diseño de alcantarillado              │
│  ├── rut_06_pipe_sizing.py           # Dimensionamiento de tuberías          │
│  └── rut_27_model_metrics.py         # Extracción de métricas                │
│                                                                              │
│  Evaluación de Daños y Riesgo:                                               │
│  ├── rut_18_itzi_flood_model.py      # Simulación 2D con Itzi                │
│  ├── rut_19_flood_damage_climada.py  # Daños con CLIMADA/JRC                 │
│  ├── rut_20_avoided_costs.py         # Costos evitados (reposición)          │
│  ├── rut_21_construction_cost.py     # Costos de construcción                │
│  ├── rut_21_risk_analysis.py         # Análisis de riesgo probabilístico     │
│  └── rut_26_hydrological_impact.py   # Impacto hidrológico                   │
│                                                                              │
│  Utilidades y Reportes:                                                      │
│  ├── rut_14_swmm_modifier.py         # Modificación de archivos SWMM         │
│  ├── rut_15_dashboard.py             # Generación de dashboards              │
│  ├── rut_17_comparison_reporter.py   # Comparador de escenarios              │
│  ├── rut_22_scenario_generator.py    # Generador de escenarios TR            │
│  ├── rut_25_from_inp_to_vector.py    # Exportación a vectoriales             │
│  └── rut_28_water_quality.py         # Análisis de calidad de agua           │
│                                                                              │
│  Configuración:                                                              │
│  └── config.py                       # Parámetros globales del proyecto      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Requisitos

### Software Requerido
- **Python** 3.10 o superior
- **SWMM 5.2+** (Storm Water Management Model)
- **GRASS GIS 8.4+** (para simulaciones Itzi)
- **Git** (para control de versiones)

### Dependencias Python Principales
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
itzi (instalado en entorno GRASS)
```

### Datos de Entrada Requeridos
- **Modelo SWMM** (`.inp`): Red de alcantarillado base
- **DEM** (`.tif`): Modelo Digital de Elevaciones
- **Predios** (`.gpkg`): Polígonos de predios candidatos
- **Red** (`.gpkg`): Geometría de la red de alcantarillado

---

## 🚀 Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/MarceloBarrosVanegas/stormwater-tank-planner.git
cd stormwater-tank-planner
```

### 2. Crear Entorno Conda
```bash
conda env create -f environment.yml
conda activate stormwater
```

### 3. Instalar PyPiper (Dependencia Interna)
El proyecto requiere el módulo PyPiper ubicado en:
```
~/OneDrive/ALCANTARILLADO_PyQt5/00_MODULOS/pypiper
```

Verificar que la estructura exista o actualizar `config.py` con la ruta correcta.

### 4. Configurar GRASS GIS para Itzi
Asegurar que GRASS GIS 8.4 esté instalado en:
```
C:\Program Files\GRASS GIS 8.4\
```

### 5. Verificar Instalación
```bash
python -c "import config; print('Configuración OK')"
```

---

## 💻 Uso

### Ejemplo Básico - Análisis Secuencial

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

# Inicializar el runner
runner = StormwaterOptimizationRunner(
    project_root=config.PROJECT_ROOT,
    eval_id="ejemplo_01"
)

# Ejecutar análisis secuencial con optimizador greedy
result = runner.run_sequential_analysis(
    max_tanks=10,
    max_iterations=50,
    min_tank_vol=1000.0,
    max_tank_vol=50000.0,
    optimizer_mode='greedy',  # 'greedy' o 'nsga'
    stop_at_breakeven=True,
    breakeven_multiplier=1.0
)
```

### Ejemplo - Optimización NSGA-II

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
from rut_23_nsga_optimizer import run_nsga_optimization
import config

# Configurar para análisis probabilístico (EAD)
config.TR_LIST = [1, 2, 5, 10, 25, 50, 100]

# Ejecutar NSGA-II
result = runner.run_sequential_analysis(
    max_tanks=20,
    optimizer_mode='nsga',
    optimization_tr_list=[1, 2, 5, 10, 25],  # TRs para optimización
    validation_tr_list=[1, 2, 5, 10, 25, 50, 100]  # TRs para validación
)
```

### Ejemplo - Optimización de Pesos de Ranking

```python
from rut_29_nsga_ranking_optimizer import NSGARankingOptimizer

# Optimizar pesos de ranking automáticamente
optimizer = NSGARankingOptimizer(
    elev_file=config.ELEV_FILE
)

# Ejecutar optimización
pareto_results = optimizer.optimize(
    n_generations=50,
    pop_size=24,
    max_tanks=15
)
```

---

## 📁 Módulos Principales

### `rut_10_run_tanque_tormenta.py`
**Orquestador principal** del workflow de optimización.

```python
class StormwaterOptimizationRunner:
    """Ejecuta el análisis completo de optimización de tanques."""
    
    def run_sequential_analysis(...)
        """
        Análisis secuencial que puede usar:
        - 'greedy': Algoritmo voraz iterativo
        - 'nsga': NSGA-II multi-objetivo
        """
```

### `rut_16_dynamic_evaluator.py`
**Evaluador dinámico** que ejecuta el pipeline completo para cada solución:
1. PathFinder (enrutamiento)
2. SewerPipeline (diseño)
3. SWMMModifier (modificación INP)
4. SWMM (simulación)
5. Métricas de resultado

### `rut_27_model_metrics.py`
**Extractor de métricas** hidráulicas del sistema:
- Volúmenes de inundación
- Caudales pico
- Profundidades máximas
- Health de la red
- Ranking de candidatos

### `rut_19_flood_damage_climada.py`
**Evaluación de daños** usando CLIMADA con curvas JRC:
- Mapeo de uso de suelo a sectores CLIMADA
- Curvas de daño por sector (residential, commercial, industrial)
- Daño total y por predio

### `rut_21_construction_cost.py`
**Cálculo de costos** de construcción:
- Tuberías (excavación, relleno, materiales)
- Tanques (excavación, concreto, impermeabilización)
- Terrenos (costo por m²)

---

## ⚙️ Configuración

El archivo `config.py` contiene todos los parámetros configurables:

### Parámetros de Tanques
```python
TANK_DEPTH_M = 15.0              # Profundidad del tanque (m)
TANK_MIN_VOLUME_M3 = 1000.0      # Volumen mínimo (m³)
TANK_MAX_VOLUME_M3 = 100000.0    # Volumen máximo (m³)
TANK_VOLUME_SAFETY_FACTOR = 1.05 # Factor de seguridad
MAX_TANKS = 30                   # Máximo número de tanques
```

### Parámetros de Optimización
```python
# Pesos para ranking de nodos (0 = no se optimiza)
FLOODING_RANKING_WEIGHTS = {
    'flow_over_capacity': 0.5,
    'flow_node_flooding': 0.5,
    'vol_node_flooding': 0.0,
    'outfall_peak_flow': 0,
    'failure_probability': 0,
}

# NSGA-II
NSGA_PARALLEL_WORKERS = 6
SWMM_THREADS = 1
```

### Parámetros de Costo
```python
COST_COMPONENTS = {
    'deferred_investment': True,   # Costo de reposición de tuberías
    'flood_damage': False,          # Daño por inundación (CLIMADA)
    'river_damage': False,         # Impacto downstream
}

LAND_COST_PER_M2 = 50.0          # Costo del terreno ($/m²)
```

---

## 🔄 Workflow de Optimización

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW DE OPTIMIZACIÓN                                  │
└─────────────────────────────────────────────────────────────────────────────┘

FASE 1: PREPARACIÓN
───────────────────
  │
  ├──▶ Cargar modelo SWMM base
  ├──▶ Cargar predios y red
  ├──▶ Calcular elevaciones y pendientes
  └──▶ Identificar nodos inundados (baseline)
       │
       ▼
FASE 2: RANKING DE CANDIDATOS
──────────────────────────────
  │
  ├──▶ Generar pares (nodo → predio)
  ├──▶ Calcular distancias y desniveles
  ├──▶ Aplicar pesos de ranking (FLOODING_RANKING_WEIGHTS)
  └──▶ Ordenar candidatos por score
       │
       ▼
FASE 3: OPTIMIZACIÓN (Greedy o NSGA-II)
────────────────────────────────────────
  │
  ┌─▶ GREEDY MODE                          ┌─▶ NSGA-II MODE
  │   ─────────────                         │   ──────────────
  │   Para cada iteración:                  │   Para cada generación:
  │   1. Seleccionar mejor candidato        │   1. Generar población
  │   2. Diseñar tubería (PathFinder)       │   2. Evaluar fitness (SWMM)
  │   3. Dimensionar tubería                │   3. Cruzamiento y mutación
  │   4. Agregar tanque a SWMM              │   4. Selección no dominada
  │   5. Correr SWMM                        │   5. Elitismo
  │   6. Evaluar resultados                 │
  │   7. Si útil, agregar a solución        │
  │                                          │
  ▼                                          ▼
FASE 4: EVALUACIÓN DE RESULTADOS
─────────────────────────────────
  │
  ├──▶ Comparar vs baseline
  ├──▶ Calcular métricas:
  │    • Reducción de volumen inundado
  │    • Reducción de caudal pico
  │    • Health de la red
  │    • Costo total
  └──▶ Generar reportes y mapas
       │
       ▼
FASE 5: ANÁLISIS DE DAÑOS (Opcional)
─────────────────────────────────────
  │
  ├──▶ Ejecutar Itzi 2D
  ├──▶ Calcular daños con CLIMADA
  ├──▶ Calcular EAD (Expected Annual Damage)
  └──▶ Generar curvas de daño
```

---

## 📊 Resultados

Los resultados se guardan en `optimization_results/`:

```
optimization_results/
├── nsga_evaluations/              # Resultados de NSGA-II
│   └── {eval_id}/
│       ├── best_solution/         # Mejor solución encontrada
│       ├── pareto_front.csv       # Frente de Pareto
│       └── evolution.png          # Gráfico de evolución
├── sequential_analysis/           # Análisis secuencial (Greedy)
│   ├── iteration_001/
│   ├── iteration_002/
│   └── ...
├── comparison_reports/            # Reportes comparativos
│   ├── comparison_baseline_vs_solution.xlsx
│   └── maps/
└── itzi_results/                  # Resultados de inundación 2D
    ├── max_water_depth.tif
    └── flood_damage_results.gpkg
```

### Métricas de Salida

| Métrica | Descripción | Unidad |
|---------|-------------|--------|
| `flooding_vol_reduction` | Reducción de volumen inundado | m³ |
| `flooding_vol_reduction_pct` | Reducción porcentual de volumen | % |
| `flooding_peak_flow_reduction` | Reducción de caudal pico de inundación | m³/s |
| `outfall_peak_flow_reduction` | Reducción de caudal en outfall | m³/s |
| `network_health` | Health promedio de la red | 0-1 |
| `total_cost` | Costo total de construcción | USD |
| `bc_ratio` | Relación Beneficio/Costo | - |

---

## 🤝 Contribución

### Guías de Contribución
1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Reportar Issues
Para reportar bugs o solicitar funcionalidades, usar el [issue tracker](https://github.com/MarceloBarrosVanegas/stormwater-tank-planner/issues).

---

## 📄 Licencia

Este proyecto está licenciado bajo MIT License - ver [LICENSE](LICENSE) para detalles.

---

## 👥 Autores

- **Marcelo Barros Vanegas** - *Desarrollo principal* - [GitHub](https://github.com/MarceloBarrosVanegas)

### Agradecimientos
- EPMAPS (Empresa Pública Metropolitana de Agua Potable y Saneamiento de Quito)
- CLIMADA Project por las curvas de daño JRC
- EPA por SWMM
- Itzi Development Team

---

## 📚 Referencias

1. EPA. (2022). *Storm Water Management Model (SWMM) Version 5.2*. Environmental Protection Agency.
2. CLIMADA Project. (2024). *JRC Depth-Damage Curves for South America*.
3. Deb, K. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*.
4. Itzi. (2024). *Itzi - A 2D Flood Simulation Tool*. https://itzi.readthedocs.io/
5. OpenStreetMap Contributors. (2024). *OpenStreetMap Data*.

---

<p align="center">
  <strong>Stormwater Tank Planner</strong> - Optimizando la infraestructura de drenaje urbano 🌧️
</p>
