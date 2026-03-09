# Stormwater Tank Planner

Framework de optimización multi-objetivo para planificación de tanques de tormenta con modelación hidrodinámica 1D-2D acoplada.

---

## 1. Marco Teórico y Ecuaciones

### 1.1 Modelo 1D: Ecuaciones de Saint-Venant (SWMM)

El modelo unidimensional resuelve las ecuaciones completas de Saint-Venant en régimen transitorio (onda dinámica):

**Conservación de masa:**
$$
\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_\ell
$$

**Conservación de momento:**
$$
\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x}\left(\frac{Q^2}{A}\right) + gA\frac{\partial H}{\partial x} + gA S_f + gA S_m = 0
$$

Donde:
- $Q$ = caudal $(m^3/s)$
- $A$ = área hidráulica $(m^2)$  
- $H = z + y$ = carga hidráulica (elevación de fondo $z$ + tirante $y$) $(m)$
- $q_\ell$ = aportes distribuidos $(m^2/s)$
- $g$ = gravedad $(m/s^2)$

**Pendiente de fricción (Manning):**
$$
S_f = \frac{n^2 Q|Q|}{A^2 R_h^{4/3}}
$$

Con $n$ = coeficiente de Manning y $R_h$ = radio hidráulico.

---

### 1.2 Modelo 2D: Shallow Water Equations (Itzi)

El flujo superficial se formula con las ecuaciones de aguas someras en aproximación de onda difusiva:

**Conservación de masa:**
$$
\frac{\partial h}{\partial t} + \frac{\partial (uh)}{\partial x} + \frac{\partial (vh)}{\partial y} = R - I + q_{exc}
$$

**Conservación de momento (simplificación de onda difusiva):**
$$
\frac{\partial (z+h)}{\partial x} + S_{f,x} = 0, \quad \frac{\partial (z+h)}{\partial y} + S_{f,y} = 0
$$

Donde:
- $h$ = profundidad de agua $(m)$
- $(u,v)$ = velocidades promedio en profundidad $(m/s)$
- $R, I$ = tasas de precipitación e infiltración $(m/s)$
- $q_{exc}$ = término de intercambio 1D-2D

**Fricción de fondo (Manning):**
$$
S_{f,x} = \frac{n^2 u \sqrt{u^2+v^2}}{h^{4/3}}, \quad S_{f,y} = \frac{n^2 v \sqrt{u^2+v^2}}{h^{4/3}}
$$

---

### 1.3 Acoplamiento Bidireccional 1D-2D

El intercambio en pozos de inspección se modela mediante relación tipo orificio:

$$
Q_{exc,i} = C_d A_{mh,i} \sqrt{2g|\Delta H_i|} \cdot \text{sgn}(\Delta H_i)
$$

Donde:
- $\Delta H_i = H_{1D,i} - H_{2D,i}$ = diferencia de carga
- $H_{2D,i} = z_i + h_i$ = carga superficial
- $C_d$ = coeficiente de descarga
- $A_{mh,i}$ = área hidráulica efectiva del pozo

**Mecanismos de intercambio:**
- **Surgencia**: cuando $H_{1D,i} > z_i$ y $\Delta H_i > 0$ (red → superficie)
- **Drenaje**: cuando $H_{2D,i} > z_i$ y $\Delta H_i < 0$ (superficie → red)

---

## 2. Metodología de Optimización

### 2.1 Arquitectura de Tres Niveles

La metodología implementa un esquema de optimización anidada:

```
NIVEL 1: META-OPTIMIZACIÓN (NSGA-II)
─────────────────────────────────────
Optimiza el vector de pesos w que parametriza la función de priorización:

     w = [w_flow_over_capacity
          w_flow_node_flooding
          w_vol_node_flooding
          w_outfall_peak_flow
          w_failure_probability]

Restricciones: Σw_k = 1,  w_k ≥ 0

         ↓ Cada individuo w ejecuta completamente el Nivel 2

NIVEL 2: SELECCIÓN SECUENCIAL (Greedy)
──────────────────────────────────────
Construye solución incremental de tanques:

FOR cada iteración:
    1. Calcular ranking de nodos según w
    2. Seleccionar nodo de mayor puntaje
    3. Encontrar predio óptimo (radio de búsqueda)
    4. Diseñar tubería (algoritmo de Dijkstra sobre OSM)
    5. Dimensionar derivación (régimen permanente)
    6. Ejecutar simulación 1D-2D iterativa
    7. Extraer métricas (costos, volúmenes, caudales)
    8. Actualizar ranking con métricas recalculadas
END

         ↓ Retorna vector de objetivos F(w)

NIVEL 0: LÍNEA BASE
───────────────────
Simulación sin tanques para múltiples períodos de retorno
Cálculo del EAD base
```

---

### 2.2 Vector de Objetivos

Para cada configuración de pesos $\mathbf{w}$, el desempeño se cuantifica mediante:

$$
\mathbf{F}(\mathbf{w}) =
\begin{bmatrix}
\Delta V_{\text{flood}} \;(\%) \; \uparrow \\[0.3em]
\Delta Q_{\text{flood}} \;(\%) \; \uparrow \\[0.3em]
\Delta Q_{\text{outfall}} \;(\%) \; \uparrow \\[0.3em]
H_{\text{network}} \; \downarrow \\[0.3em]
C_{\text{social}} \;(\mathrm{USD}) \; \downarrow
\end{bmatrix}
$$

Donde:
- $\Delta V_{\text{flood}}$ = reducción porcentual de volumen de inundación
- $\Delta Q_{\text{flood}}$ = reducción porcentual de caudal de desbordamiento
- $\Delta Q_{\text{outfall}}$ = reducción porcentual de caudal pico en outfalls
- $H_{\text{network}}$ = indicador agregado del estado hidráulico de la red (h/D)
- $C_{\text{social}}$ = costo social neto = inversión - beneficios por daños evitados

Los símbolos $\uparrow$ y $\downarrow$ indican maximización y minimización respectivamente.

---

### 2.3 Pipeline de Evaluación por Tanque

Para cada tanque candidato, el sistema ejecuta secuencialmente:

1. **PathFinder** (`rut_00`): Algoritmo de Dijkstra sobre grafo de OpenStreetMap
   - Función de peso: $f(e) = \alpha \cdot L + \beta \cdot \Delta z + \gamma \cdot T_{via}$
   
2. **SewerPipeline** (`rut_03`): Dimensionamiento hidráulico
   - Velocidad de auto-limpieza: $V_{min}$
   - Velocidad máxima: $V_{max}$
   - Capacidad máxima: $(h/D)_{max}$

3. **SWMM Modifier** (`rut_14`): Actualización del modelo
   - Nodo de tanque con carga hidráulica $H_{tank}$
   - Conexión de derivación con caudal de diseño $Q_{deriv}$
   - Volumen de almacenamiento $V_{stored}$

4. **Simulación 1D-2D** (`rut_18`): Acoplamiento SWMM-Itzi

5. **Extracción de métricas** (`rut_27`): Indicadores de desempeño

---

## 3. Evaluación Probabilística del Riesgo

### 3.1 Generación de Escenarios

Para cada período de retorno $T_r \in \{1, 2, 5, 10, 25, 50, 100\}$ años:

**Ecuación IDF** (Intensidad-Duración-Frecuencia):
$$
I(t, T_r) = \frac{a \cdot \ln(T_r) + b}{(c + t)^d}
$$

**Método de Bloques Alternos** (*Alternating Block Method*):
- Duración total: 60 minutos
- Paso temporal: $\Delta t = 5$ minutos (12 bloques)
- Bloque de mayor intensidad centrado en el hietograma
- Bloques restantes distribuidos simétricamente alternados

### 3.2 Funciones de Vulnerabilidad

La cuantificación del daño utiliza curvas del Joint Research Centre (JRC) implementadas en CLIMADA. La **Relación Media de Daño (MDR)** se define como:

$$
\text{MDR}_s(h): \mathbb{R}_{\geq 0} \rightarrow [0, 1]
$$

Donde:
- $h$ = profundidad de inundación $(m)$
- $s$ = sector económico (residencial, comercial, industrial, infraestructura, agricultura)

**Sectores y curvas:**

| Sector | Curva | Origen |
|--------|-------|--------|
| Residencial | JRC South America | Huizinga et al., 2017 |
| Comercial | JRC South America | Huizinga et al., 2017 |
| Industrial | JRC South America | Huizinga et al., 2017 |
| Infraestructura | Derivada | Metodología JRC/FEMA |
| Agricultura | Derivada | Metodología JRC/FAO |

### 3.3 Cálculo del Daño por Evento

Para un período de retorno $T_r$:

$$
D_{\text{edif}}(T_r) = \sum_{i \in \mathcal{P}(T_r)} V_i \cdot \text{MDR}_{s_i}(h_i)
$$

Donde:
- $V_i$ = valor de construcción del predio $i$ (excluye terreno)
- $h_i$ = profundidad de inundación extraída del ráster ITZI
- $s_i$ = sector económico del predio
- $\mathcal{P}(T_r)$ = conjunto de predios expuestos

### 3.4 Daño Anual Esperado (EAD)

El riesgo acumulado se cuantifica mediante integración probabilística:

$$
\text{EAD} = \int_{T_{min}}^{T_{max}} D(T_r) \cdot f(T_r) \, dT_r
$$

Aproximado numéricamente mediante integración trapezoidal sobre los escenarios discretos.

---

## 4. Implementación

### 4.1 Estructura del Código

**Optimización:**
- `rut_10_run_tanque_tormenta.py` - Orquestador principal
- `rut_15_optimizer.py` - Greedy y NSGA-II
- `rut_16_dynamic_evaluator.py` - Pipeline de evaluación
- `rut_23_nsga_optimizer.py` - NSGA-II para selección de tanques
- `rut_29_nsga_ranking_optimizer.py` - NSGA-II para pesos $\mathbf{w}$

**Modelación Hidráulica:**
- `rut_00_path_finder.py` - Enrutamiento Dijkstra sobre OSM
- `rut_02_get_flodded_nodes.py` - Identificación de nodos críticos
- `rut_03_run_sewer_design.py` - Diseño hidráulico
- `rut_06_pipe_sizing.py` - Cálculos de capacidad
- `rut_27_model_metrics.py` - Extracción de métricas

**Daños y Riesgo:**
- `rut_18_itzi_flood_model.py` - Simulación 2D
- `rut_19_flood_damage_climada.py` - CLIMADA/JRC
- `rut_20_avoided_costs.py` - Costos diferidos
- `rut_21_construction_cost.py` - Costos de construcción
- `rut_21_risk_analysis.py` - Análisis probabilístico EAD
- `rut_26_hydrological_impact.py` - Impacto en outfalls

**Utilidades:**
- `rut_14_swmm_modifier.py` - Modificación SWMM
- `rut_15_dashboard.py` - Reportes Excel
- `rut_17_comparison_reporter.py` - Comparación de escenarios
- `rut_22_scenario_generator.py` - Generación de escenarios TR
- `rut_25_from_inp_to_vector.py` - Exportación vectorial
- `rut_28_water_quality.py` - Análisis de TSS y DBO

### 4.2 Uso

**Optimización completa (modo NSGA-II):**

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

config.TR_LIST = [1, 2, 5, 10, 25, 50, 100]

runner = StormwaterOptimizationRunner(
    project_root=config.PROJECT_ROOT,
    eval_id="analisis_probabilistico"
)

resultado = runner.run_sequential_analysis(
    max_tanks=20,
    optimizer_mode='nsga',
    optimization_tr_list=[1, 2, 5, 10, 25],
    validation_tr_list=[1, 2, 5, 10, 25, 50, 100]
)
```

**Análisis rápido (modo Greedy):**

```python
runner = StormwaterOptimizationRunner()

resultado = runner.run_sequential_analysis(
    max_tanks=10,
    optimizer_mode='greedy',
    stop_at_breakeven=True
)
```

**Meta-optimización de pesos:**

```python
from rut_29_nsga_ranking_optimizer import NSGARankingOptimizer

optimizer = NSGARankingOptimizer(elev_file=config.ELEV_FILE)

resultado = optimizer.optimize(
    n_generations=50,
    pop_size=24,
    max_tanks=15
)
```

### 4.3 Configuración

**Parámetros de tanque (`config.py`):**
```python
TANK_DEPTH_M = 15.0              # Profundidad (m)
TANK_MIN_VOLUME_M3 = 1000.0      # Volumen mínimo (m³)
TANK_MAX_VOLUME_M3 = 100000.0    # Volumen máximo (m³)
MAX_TANKS = 30                   # Máximo número de tanques
TANK_VOLUME_SAFETY_FACTOR = 1.05 # Factor de seguridad
```

**Pesos de priorización:**
```python
FLOODING_RANKING_WEIGHTS = {
    'flow_over_capacity': 0.5,    # Exceso de capacidad
    'flow_node_flooding': 0.5,    # Caudal de inundación
    'vol_node_flooding': 0.0,     # Volumen de inundación
    'outfall_peak_flow': 0,       # Caudal pico outfall
    'failure_probability': 0,     # Probabilidad de falla
}
```

**NSGA-II:**
```python
NSGA_PARALLEL_WORKERS = 6        # Workers paralelos
SWMM_THREADS = 1                 # Threads por simulación
N_GENERATIONS = 50               # Generaciones
POP_SIZE = 24                    # Tamaño de población
```

---

## 5. Resultados

El sistema genera en `optimization_results/`:

```
optimization_results/
├── nsga_evaluations/{id}/
│   ├── best_solution/
│   │   ├── modified.inp         # Modelo SWMM modificado
│   │   ├── tanks.gpkg           # Ubicación y geometría de tanques
│   │   └── report.xlsx          # Reporte detallado
│   ├── pareto_front.csv         # Frente de Pareto
│   └── evolution.png            # Gráfico de convergencia
├── sequential_analysis/
│   ├── iteration_001/
│   ├── iteration_002/
│   └── ...
└── itzi_results/
    ├── max_water_depth.tif      # Profundidad máxima de inundación
    └── flood_damage_results.gpkg # Daños por predio
```

### Métricas de Salida

| Símbolo | Descripción | Unidad |
|---------|-------------|--------|
| $\Delta V_{\text{flood}}$ | Reducción de volumen inundado | m³ |
| $\Delta V_{\text{flood}}^{\%}$ | Reducción porcentual | % |
| $\Delta Q_{\text{flood}}$ | Reducción de caudal de desbordamiento | m³/s |
| $\Delta Q_{\text{outfall}}$ | Reducción de caudal pico en outfall | m³/s |
| $H_{\text{network}}$ | Health promedio de la red | 0-1 |
| $C_{\text{social}}$ | Costo social neto | USD |
| EAD | Daño Anual Esperado | USD/año |

---

## Referencias

1. EPA. (2022). *Storm Water Management Model (SWMM) Version 5.2*. Environmental Protection Agency.

2. Huizinga, J., Moel, H. de, & Szewczyk, W. (2017). *Global flood depth-damage functions: Methodology and the database with guidelines*. EU-JRC Technical Report.

3. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

4. Aznar, B., & Barnadas, M. (2016). *Itzi (version 17.1) [software]*.

5. OpenStreetMap Contributors. *OpenStreetMap Data*. https://www.openstreetmap.org/

---

## Licencia

MIT License
