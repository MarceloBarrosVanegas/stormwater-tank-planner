# Stormwater Tank Planner

**Framework de optimización multi-objetivo para planificación de tanques de tormenta con modelación hidrodinámica 1D-2D acoplada**

---

## Resumen

Se presenta una metodología para la localización y dimensionamiento de tanques de tormenta en el subsistema de alcantarillado El Colegio Occidental (Quito), con el objetivo de mitigar las inundaciones urbanas y reducir los daños económicos y ambientales asociados. El sistema se modela mediante el acoplamiento de un modelo hidrodinámico unidimensional en régimen transitorio (SWMM) con un modelo bidimensional de inundación superficial (ITZI), lo que permite representar tanto la dinámica interna de la red de drenaje como la propagación del flujo desbordado sobre el terreno urbano.

El riesgo de inundación se cuantifica económicamente a partir de curvas de vulnerabilidad del Joint Research Centre (JRC), integradas en el marco CLIMADA, que relacionan la profundidad de inundación con el daño esperado según el tipo de uso del suelo (residencial, comercial, industrial y equipamiento público).

La optimización se estructura como un esquema anidado de tres niveles: (i)~un meta-optimizador NSGA-II que explora el espacio de pesos del ranking, (ii)~un algoritmo Greedy de selección secuencial que construye soluciones completas de tanques para cada configuración de pesos, y (iii)~un pipeline de evaluación iterativa que, para cada tanque candidato, traza la ruta de derivación, dimensiona las secciones de derivación y ajusta iterativamente las dimensiones de los elementos alcanzar convergencia.

El esquema optimiza cinco objetivos simultáneos: reducción de volumen de inundación, reducción de caudal de desbordamiento, reducción de caudal pico en los outfalls, minimización de la relación h/D en conductos críticos, y minimización del costo social neto (inversión menos beneficios por daños evitados).

Complementariamente, se evalúa la solución resultante mediante un análisis de calidad de agua que modela la remoción de sólidos suspendidos totales (TSS) y demanda bioquímica de oxígeno (DBO) por sedimentación en los tanques, así como una comparación hidrológica entre escenarios de cuenca natural, urbanizada y urbanizada con tanques, que cuantifica el grado de restauración hacia la dinámica hidrológica natural alcanzado por la infraestructura propuesta.

---

## 1. Descripción del Área de Estudio

### 1.1 Delimitación espacial

La microcuenca hidrográfica y urbana del subsistema El Colegio Occidental se ubica en los sectores centro norte y norte de Quito, en las laderas occidentales del Pichincha. La cuenca comprende las siguientes subcuencas:

| Subcuenca | Área (Ha) | Color en mapa |
|-----------|-----------|---------------|
| Cuenca urbana principal | 1,683 | Amarillo |
| Quebrada Rumihurco | 942 | Verde |
| Quebrada San Lorenzo | 457 | Púrpura |
| Quebrada San Antonio | 220 | Turquesa |
| Quebrada San Carlos | 209 | Azul |

El subsistema está diseñado para captar los caudales de exceso provenientes de las laderas del Pichincha hacia los sectores bajos que colindan con la Av. De la Prensa, conduciendo los flujos hasta el río Monjas. La red principal aliviará los caudales en exceso de los colectores San Carlos, Atucucho, Flavio Alfaro, El Colegio, Sabanilla y San Lorenzo, con un caudal pico estimado de 69.41 m³/s en el punto de descarga hacia el cauce del río Monjas, ubicado aproximadamente 440 metros aguas abajo de la descarga actual del colector El Colegio.

### 1.2 Obras de retención existentes

El subsistema incorpora diversas obras de retención distribuidas estratégicamente. Un ejemplo representativo es la obra de retención en la quebrada Rumihurco, que cuenta con una capacidad de captación de aproximadamente 25,000 m³ y contempla un caudal máximo de 37.42 m³/s para un período de retorno de 25 años.

---

## 2. Marco Metodológico General

### 2.1 Arquitectura de optimización anidada

La metodología se estructura como una arquitectura de optimización anidada que integra (i) una simulación hidrodinámica acoplada en 1D-2D de referencia, (ii) una optimización multiobjetivo de los parámetros de priorización, y (iii) una selección secuencial con evaluación iterativa.

El propósito del esquema es evitar rankings estáticos: la idoneidad de un tanque depende del estado del sistema, el cual cambia tras cada intervención. En consecuencia, la selección se formula como un proceso iterativo acoplado a la simulación, donde las métricas se recalculan con el modelo actualizado para reflejar interacciones no lineales en la red.

El flujo de trabajo se organiza en tres niveles:

**Nivel 0**: Establece la línea base y define las métricas de referencia mediante simulación hidrodinámica sin tanques.

**Nivel 1**: Determina los pesos que gobiernan la función de priorización mediante NSGA-II, evaluando cada configuración en función del desempeño alcanzado por el procedimiento de selección del Nivel 2.

**Nivel 2**: Construye una solución mediante un algoritmo greedy secuencial que agrega tanques uno a uno; tras cada adición se evalúa de manera iterativa las dimensiones de la infraestructura propuesta, y se actualiza los valores de los parámetros de priorización.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DE OPTIMIZACIÓN ANIDADA                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  NIVEL 1: META-OPTIMIZACIÓN (NSGA-II)                                           │
│  ─────────────────────────────────────                                          │
│                                                                                  │
│  Variables de decisión:                                                         │
│      w = [w_flow_over_capacity                                                 │
│           w_flow_node_flooding                                                  │
│           w_vol_node_flooding                                                   │
│           w_outfall_peak_flow                                                   │
│           w_failure_probability]                                                │
│                                                                                  │
│  Restricciones: Σw_k = 1,  w_k ≥ 0                                              │
│  Parámetro adicional: h/D_max ∈ [0, 0.95]                                      │
│                                                                                  │
│         ↓ Cada individuo w ejecuta completamente el Nivel 2                     │
│                                                                                  │
│  NIVEL 2: SELECCIÓN SECUENCIAL (Greedy)                                         │
│  ───────────────────────────────────────                                        │
│                                                                                  │
│  WHILE criterio de parada no alcanzado:                                         │
│      1. Calcular ranking de nodos según pesos w                                 │
│      2. Seleccionar nodo de mayor puntaje                                       │
│      3. Encontrar predio óptimo (radio de búsqueda)                             │
│      4. Trazar ruta (Dijkstra sobre OSM)                                        │
│      5. Diseñar derivación (régimen permanente)                                 │
│      6. Ejecutar simulación 1D-2D iterativa                                     │
│      7. Extraer métricas (costos, volúmenes, caudales)                          │
│      8. Actualizar ranking con métricas recalculadas                            │
│  END                                                                              │
│                                                                                  │
│         ↓ Retorna vector de objetivos F(w)                                      │
│                                                                                  │
│  NIVEL 0: LÍNEA BASE                                                            │
│  ───────────────────                                                            │
│  Simulación sin tanques para múltiples períodos de retorno                      │
│  Cálculo del EAD base                                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Nivel 0: Preparación y línea base

El Nivel 0 consolida la información espacial y técnica necesaria para representar el sistema. Se integran los insumos cartográficos (catastro, red vial, predios, cobertura de suelo), el modelo de elevación del terreno y los modelos hidrodinámicos acoplados: SWMM (1D, onda dinámica) e ITZI (2D, esquema de inercia parcial).

Con el modelo acoplado se ejecutan simulaciones de referencia sin tanques para múltiples períodos de retorno (T_r ∈ {1, 2, 5, 10, 25, 50, 100} años). A partir de estas corridas se calcula el Daño Anual Esperado (EAD), integrando: (i) daños directos a edificaciones mediante curvas de vulnerabilidad JRC-CLIMADA aplicadas a los predios expuestos, y (ii) costos de reposición de infraestructura de alcantarillado para tuberías cuya capacidad es excedida.

El EAD constituye el valor base de costos residuales contra el cual se comparan las soluciones propuestas.

De estas simulaciones se extraen además métricas hidráulicas (volúmenes y caudales de inundación, profundidades en predios, estados hidráulicos de conductos) que permiten: (i) cuantificar mejoras de forma consistente entre escenarios, (ii) construir el conjunto de nodos candidatos y su priorización inicial, y (iii) establecer el punto de comparación para interpretar la contribución marginal de cada tanque en los niveles posteriores.

### 2.3 Nivel 1: Optimización de pesos mediante NSGA-II

El Nivel 1 determina los pesos w_k que parametrizan la función de priorización. Cada vector w asigna un valor distinto para modificar la importancia de cada variable durante la selección de pares nodo inundado - predio durante la ejecución del procedimiento del Nivel 2.

Se emplea NSGA-II para explorar el espacio de soluciones y aproximar un frente de Pareto entre objetivos de desempeño.

**Vector de pesos:**

$$
\mathbf{w} =
\begin{bmatrix}
w_{\text{flow\_over\_capacity}} \\
w_{\text{flow\_node\_flooding}}\\
w_{\text{vol\_node\_flooding}} \\
w_{\text{outfall\_peak\_flow}}\\
w_{\text{failure\_probability}}
\end{bmatrix},
\qquad
w_k \ge 0,
\qquad
\sum_{k=1}^{5} w_k = 1.
$$

La evaluación de cada individuo del algoritmo evolutivo (cada configuración de pesos w) se realiza ejecutando completamente el procedimiento Greedy Secuencial del Nivel 2. En consecuencia, la aptitud no se asigna en función de un ranking calculado sobre la línea base, sino a partir del desempeño inducido por los pesos propuestos una vez que el algoritmo de selección secuencial incorpora tanques, actualiza el modelo y recalcula métricas de forma iterativa.

**Vector de objetivos:**

$$
\mathbf{F}(\mathbf{w}) =
\begin{bmatrix}
\Delta V_{\text{flood}} \;(\\%) \; \uparrow \\[0.3em]
\Delta Q_{\text{flood}} \;(\\%) \; \uparrow \\[0.3em]
\Delta Q_{\text{outfall}} \;(\\%) \; \uparrow \\[0.3em]
H_{\text{network}} \; \downarrow \\[0.3em]
C_{\text{social}} \;(\\mathrm{USD}) \; \downarrow
\end{bmatrix}
$$

Donde:
- ΔV_flood = reducción porcentual de volumen de inundación (se maximiza)
- ΔQ_flood = reducción porcentual de caudal de desbordamiento (se maximiza)
- ΔQ_outfall = reducción porcentual de caudal en outfalls (se maximiza)
- H_network = indicador agregado del estado hidráulico de la red (se minimiza)
- C_social = costo social neto de la intervención (se minimiza)

### 2.4 Nivel 2: Selección secuencial con evaluación iterativa

El Nivel 2 construye una solución incremental mediante un procedimiento greedy que incorpora tanques secuencialmente. En cada iteración, el algoritmo selecciona el nodo de mayor puntaje según el ranking definido por los pesos w.

Para el nodo seleccionado, se identifican predios disponibles dentro del radio de búsqueda. Utilizando la red vial de OpenStreetMap como grafo ponderado, se calcula la ruta óptima desde el nodo inundado hasta cada predio mediante el algoritmo de Dijkstra. La función de peso de cada arista integra: (i) longitud geométrica, (ii) penalización por desnivel adverso que dificulta el flujo gravitacional, y (iii) factor de ajuste según tipo de vía.

El par nodo-predio óptimo se valida contra restricciones de elegibilidad: disponibilidad de área, diferencia de elevación favorable, y distancia mínima respecto a tanques existentes. Si se satisfacen las restricciones, se incorpora al modelo de SWMM.

Tras cada simulación con el modelo modificado, se recalculan las métricas de desempeño y se actualiza el ranking. Esta actualización captura que la efectividad marginal de un tanque depende del estado del sistema: un nodo prioritario puede perder relevancia si el almacenamiento aguas arriba reduce su afluencia, mientras emergen nuevos puntos críticos donde la derivación aporta mayor reducción de daños.

El proceso finaliza al alcanzar el número máximo de tanques, agotar candidatos elegibles, o cuando la reducción marginal de costos residuales respecto al incremento de inversión cae bajo un umbral predefinido.

---

## 3. Sistema Acoplado de Modelación Hidrodinámica (1D-2D)

### 3.1 Modelo 1D de alcantarillado (SWMM)

El dominio 1D se resuelve con el motor hidrodinámico de SWMM 5.2, que integra numéricamente las ecuaciones completas de Saint-Venant (onda dinámica) para flujo transitorio en conductos.

**Conservación de masa 1D:**

$$
\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_\ell
$$

**Conservación de momento 1D:**

$$
\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x}\!\left(\frac{Q^2}{A}\right) + gA\frac{\partial H}{\partial x} + gA\,S_f + gA\,S_m = 0
$$

Donde:
- Q = caudal (m³/s)
- A = área hidráulica (m²)
- H = z + y = carga hidráulica (m)
- q_ℓ = aportes distribuidos (m²/s)
- g = gravedad (m/s²)

**Pendiente de fricción (Manning):**

$$
S_f = \frac{n^2\,Q|Q|}{A^2\,R_h^{4/3}}
$$

Con n = coeficiente de Manning y R_h = radio hidráulico.

Esta formulación permite representar condiciones de remanso, inversión de flujo y transiciones entre régimen a superficie libre y presurizado.

### 3.2 Modelo 2D de escorrentía superficial (Itzi)

El flujo superficial se formula con las Shallow Water Equations promediadas en profundidad. ITZI adopta la aproximación de onda difusiva, en la que se descartan los términos inerciales en las ecuaciones de momento y se retiene el balance entre gradiente de superficie libre y fricción de fondo.

**Conservación de masa 2D:**

$$
\frac{\partial h}{\partial t} + \frac{\partial (uh)}{\partial x} + \frac{\partial (vh)}{\partial y} = R - I + q_{exc}
$$

**Conservación de momento (simplificación de onda difusiva):**

$$
\frac{\partial (z+h)}{\partial x} + S_{f,x} = 0, \quad \frac{\partial (z+h)}{\partial y} + S_{f,y} = 0
$$

Donde:
- h = profundidad de agua (m)
- (u,v) = velocidades promedio en profundidad (m/s)
- R, I = tasas de precipitación e infiltración (m/s)
- q_exc = término de intercambio 1D-2D

**Fricción de fondo (Manning):**

$$
S_{f,x} = \frac{n^2\,u\sqrt{u^2+v^2}}{h^{4/3}}, \quad S_{f,y} = \frac{n^2\,v\sqrt{u^2+v^2}}{h^{4/3}}
$$

### 3.3 Mecanismo de acoplamiento bidireccional 1D-2D

El intercambio en pozos de inspección se calcula mediante relación tipo orificio:

$$
Q_{exc,i} = C_d\,A_{mh,i}\,\sqrt{2g|\Delta H_i|} \cdot \text{sgn}(\Delta H_i)
$$

Donde:
- ΔH_i = H_1D,i - H_2D,i = diferencia de carga
- H_2D,i = z_i + h_i = carga superficial
- C_d = coeficiente de descarga
- A_mh,i = área hidráulica efectiva del pozo

**Mecanismos de intercambio:**
- **Surgencia**: cuando H_1D > z_i y ΔH_i > 0 (red → superficie)
- **Drenaje**: cuando H_2D > z_i y ΔH_i < 0 (superficie → red)

---

## 4. Evaluación Probabilística del Riesgo

### 4.1 Generación de escenarios de amenaza

Para cada período de retorno T_r ∈ {1, 2, 5, 10, 25, 50, 100} años:

**Ecuación IDF** (Intensidad-Duración-Frecuencia):

$$
I(t, T_r) = \frac{13.9378 \cdot \ln(T_r) + 40.7176}{(35.5037 + t)^{0.9997}}
$$

**Método de Bloques Alternos:**
- Duración total: 60 minutos
- Paso temporal: Δt = 5 minutos (12 bloques)
- Bloque de mayor intensidad centrado en el hietograma
- Bloques restantes distribuidos simétricamente alternados

### 4.2 Funciones de vulnerabilidad (JRC/CLIMADA)

El daño a edificaciones se cuantifica mediante la Relación Media de Daño (MDR):

$$
\text{MDR}_s(h): \mathbb{R}_{\geq 0} \rightarrow [0, 1]
$$

Donde h = profundidad de inundación (m) y s = sector económico.

**Sectores y curvas:**

| Sector | Curva | Origen |
|--------|-------|--------|
| Residencial | JRC South America | Huizinga et al., 2017 |
| Comercial | JRC South America | Huizinga et al., 2017 |
| Industrial | JRC South America | Huizinga et al., 2017 |
| Infraestructura | Derivada | Metodología JRC/FEMA |
| Agricultura | Derivada | Metodología JRC/FAO |

### 4.3 Cálculo del daño por evento

Para un período de retorno T_r:

$$
D_{\text{edif}}(T_r) = \sum_{i \in \mathcal{P}(T_r)} V_i \cdot \text{MDR}_{s_i}(h_i)
$$

Donde:
- V_i = valor de construcción del predio i (excluye terreno)
- h_i = profundidad de inundación extraída del ráster ITZI
- 𝒫(T_r) = conjunto de predios expuestos

### 4.4 Daño Anual Esperado (EAD)

El riesgo acumulado se cuantifica mediante integración probabilística:

$$
\text{EAD} = \int_{0}^{1} D(p) \, dp
$$

Aproximado numéricamente mediante integración trapezoidal:

$$
\text{EAD} \approx \sum_{j=1}^{N-1} \frac{D(p_j) + D(p_{j+1})}{2} \cdot (p_j - p_{j+1})
$$

Donde p_j = 1/T_r,j es la probabilidad anual de excedencia.

### 4.5 Propagación de incertidumbre mediante Bootstrap

Para caracterizar la incertidumbre, se implementa un procedimiento de bootstrap no paramétrico con B = 1,000 iteraciones:

**Remuestreo espacial de edificaciones:**

$$
\hat{D}_{\text{edif}}^{(b)}(T_r) = \sum_{i=1}^{N_{\text{predios}}} V_{c,i}^{(b)} \cdot \text{MDR}_{s_i}(h_i)
$$

**Perturbación de costos unitarios:**

$$
\tilde{c}_{\text{rep}}^{(b)}(D_j, Z_j) = c_{\text{rep}}(D_j, Z_j) \cdot (1 + \delta^{(b)}), \quad \delta^{(b)} \sim \mathcal{N}(0,\sigma^2)
$$

Con σ = 0.05 (5% de coeficiente de variación).

---

## 5. Formulación del Problema de Optimización

### 5.1 Variables de decisión

El espacio de soluciones es combinatorio y se emplea la arquitectura de optimización anidada descrita anteriormente. Las variables de decisión del meta-optimizador NSGA-II son los pesos que gobiernan la función de priorización de nodos candidatos (mostrados en la Ecuación del vector de pesos) más el parámetro de llenado máximo h/D_max ∈ [0, 0.95].

Antes de cada evaluación, los pesos se normalizan a Σ_k w_k = 1 para garantizar la consistencia del ranking.

### 5.2 Objetivos de optimización

Se formulan cinco objetivos simultáneos:

$$
\min_{\mathbf{w}} \; \mathbf{F}(\mathbf{w}) =
\begin{bmatrix}
-\Delta V_{\text{flood}} \;(\\%) \\
-\Delta Q_{\text{flood}} \;(\\%) \\
-\Delta Q_{\text{outfall}} \;(\\%) \\
-H_{\text{network}} \\
C_{\text{social}} \;(\\text{USD})
\end{bmatrix}
$$

Los componentes del vector son:
- **ΔV_flood**: reducción porcentual del volumen total de inundación respecto al baseline
- **ΔQ_flood**: reducción porcentual del caudal pico de desbordamiento
- **ΔQ_outfall**: reducción porcentual del caudal pico descargado a cuerpos receptores
- **H_network**: indicador agregado de salud hidráulica (utilización ponderada h/D)
- **C_social**: costo social neto de la intervención

### 5.3 Costo social neto (quinto objetivo)

El quinto objetivo integra el balance económico completo de cada solución candidata 𝒮:

$$
C_{\text{social}}(\mathcal{S}) = \underbrace{C_{\text{inv}}(\mathcal{S})}_{\text{Inversión}} - \underbrace{\left[\text{EAD}_{\text{base}} - \text{EAD}(\mathcal{S})\right]}_{\text{Daño evitado}} - \underbrace{\left[C_{\text{rep,base}} - C_{\text{rep}}(\mathcal{S})\right]}_{\text{Reparación evitada}}
$$

Un valor C_social < 0 indica que los beneficios económicos superan la inversión requerida.

### 5.4 Cálculo de costos de inversión

El costo de inversión se descompone en:

$$
C_{inv}(\mathcal{S}) = \sum_{i \in \mathcal{S}} \left[ C_{\text{tanque}}(V_i) + C_{\text{deriv}}(L_i, D_i) + C_{\text{terreno}}(A_i) \right]
$$

**Costo de tanques**: Volumen de concreto armado calculado por geometría de dos cámaras con parámetros constructivos definidos y operaciones de excavación.

**Costo de derivación**: Catálogo de precios unitarios según diámetro y profundidad de la tubería de conexión.

**Costo de terreno**: Valor catastral del predio según sector económico.

---

## 6. Meta-Optimización NSGA-II (Nivel 1)

### 6.1 Formulación del problema multi-objetivo

Las variables de decisión son los n pesos activos del ranking más el parámetro de llenado:

$$
\mathbf{x} = [w_1, w_2, \ldots, w_n, h/D_{max}], \qquad w_k \in [0, 1], \quad h/D_{max} \in [0, 0.95]
$$

### 6.2 Operadores genéticos y configuración

La configuración del algoritmo NSGA-II se determina dinámicamente:

- **Tamaño de población**: N_pop = max(4 · n_var, 12)
- **Número de generaciones**: N_gen = 100 (límite superior)
- **Selección**: Tournament binario (tamaño 2)
- **Cruce**: Simulated Binary Crossover (SBX) con p_c = 0.9 y η_c = 15
- **Mutación**: Polynomial Mutation con p_m = 1/n_var y η_m = 20

**Criterio de parada anticipada**: Se monitorea el indicador de hipervolumen I_H. Si la varianza en una ventana deslizante de w generaciones cae por debajo de umbral ε, se activa la terminación temprana.

### 6.3 Arquitectura de evaluación anidada

Cada individuo de la población NSGA-II ejecuta un ciclo completo del algoritmo Greedy, que a su vez invoca múltiples simulaciones SWMM + ITZI. Esta arquitectura de tres niveles funciona así:

```
NIVEL 1 (NSGA-II) ──→ NIVEL 2 (Greedy) ──→ NIVEL 3 (SWMM+ITZI)
       ↑                     ↑                    │
       └─────────────────────┴────────────────────┘
          (retroalimentación de métricas y objetivos)
```

---

## 7. Optimizador Greedy Secuencial (Nivel 2)

### 7.1 Pipeline de evaluación secuencial

El pipeline resuelve un problema de acoplamiento inherente: el caudal derivado al tanque depende del diámetro de la tubería, pero el diámetro debe diseñarse para el caudal que recibirá. Este acoplamiento se resuelve mediante un esquema iterativo.

#### Paso 1: Trazado de la ruta de derivación

Se determina la trayectoria desde el nodo de derivación hasta el predio candidato sobre el grafo de OpenStreetMap mediante el algoritmo de Dijkstra:

$$
c(u,v) = \alpha_L \cdot L_{uv} + \alpha_z \cdot \Delta z^{+}_{uv} + \alpha_r \cdot \phi(r_{uv})
$$

Parámetros de la función de costo:

| Símbolo | Definición | Valor | Descripción |
|---------|------------|-------|-------------|
| L_uv | Longitud geométrica | [0, ∞) m | Distancia euclidiana de la arista |
| Δz⁺_uv | Desnivel adverso | [0, ∞) m | max(0, z_v - z_u); penaliza subidas |
| φ(r_uv) | Factor de preferencia vial | Ver tabla | Penalización por tipo de vía |
| α_L | Peso longitud | 0.5 | Favorecer distancia mínima |
| α_z | Peso elevación | 0.2 | Penalización por pendientes negativas |
| α_r | Peso vial | 0.3 | Favorecer tipo de vía |

### 7.2 Ecuaciones de diseño hidráulico

#### Ecuación de continuidad

$$
q = v \cdot A
$$

#### Ecuación de Gauckler-Manning-Strickler

$$
v = \frac{1}{n} \cdot R_h^{\frac{2}{3}} \cdot S^{\frac{1}{2}}
$$

#### Ecuación de Chezy

$$
v = C \cdot \sqrt{R_h \cdot S}
$$

#### Equivalencia Manning-Chezy

$$
n = \frac{R_h^{\frac{1}{6}}}{C}
$$

### 7.3 Geometría de secciones circulares

**Relaciones para sección parcialmente llena:**

| Variable | Expresión |
|----------|-----------|
| Ángulo central | θ° = 2 arccos(1 - 2h/D) |
| Radio hidráulico | R_h = (D/4)(1 - 360 sin θ° / 2πθ°) |
| Velocidad | v = (0.397 D^(2/3)/n)(1 - 360 sin θ° / 2πθ°)^(2/3) · S^(1/2) |
| Caudal | q = [D^(8/3) / (7257.15·n·(2πθ°)^(2/3))] · (2πθ° - 360 sin θ°)^(5/3) · S^(1/2) |

**Relaciones para sección llena:**

| Variable | Expresión |
|----------|-----------|
| Área | A = πD²/4 |
| Perímetro | P = πD |
| Radio hidráulico | R_h = D/4 |
| Velocidad | V = (0.397/4) D^(2/3) S^(1/2) |
| Caudal | Q = (0.312/n) D^(8/3) S^(1/2) |

### 7.4 Números adimensionales

#### Número de Froude

$$
F = \frac{v}{\sqrt{g \cdot h}}
$$

- F < 1: flujo subcrítico (fuerzas viscosas < gravitacionales)
- F = 1: flujo crítico
- F > 1: flujo supercrítico (fuerzas viscosas > gravitacionales)

#### Número de Reynolds

$$
Re = \frac{v \cdot h}{\nu}
$$

- Re < 2000: flujo laminar
- 2000 < Re < 3500: flujo transicional
- Re > 3500: flujo turbulento

### 7.5 Condiciones hidráulicas de diseño

#### Tensión tractiva

$$
\tau = S \cdot R_h \cdot g \cdot \rho
$$

#### Velocidad mínima (auto-limpieza)

$$
v_{min} = \frac{1}{n} \cdot R_h^{\frac{2}{3}} \cdot \left(\frac{\tau_{min}}{\rho \cdot g \cdot R_h}\right)^{\frac{1}{2}}
$$

Con τ_min = 1.5 Pa para sistemas pluviales.

---

## 8. Vulnerabilidad de Infraestructura

### 8.1 Función de vulnerabilidad de capacidad

La vulnerabilidad de un tramo de alcantarillado j se define por su capacidad para operar sin fallar:

$$
\mathcal{F}_j = 
\begin{cases}
1 & \text{si } \phi_j^{\text{pres}} = 1 \text{ o } \phi_j^{\text{vel}} = 1 \\
0 & \text{en caso contrario}
\end{cases}
$$

**Criterio de presurización:**

$$
\phi_j^{\text{pres}} =
\begin{cases}
1 & \text{si } \neg P_j \text{ y } \frac{h_j}{D_j} > \left(\frac{h}{D}\right)_{\text{max},j} \\
1 & \text{si } P_j \text{ y } Z_j < 6\text{m} \text{ y } \frac{h_j}{D_j} > \left(\frac{h}{D}\right)_{\text{max},j} \\
0 & \text{en caso contrario}
\end{cases}
$$

**Calado máximo según régimen:**

$$
\left(\frac{h}{D}\right)_{\text{max},j} =
\begin{cases}
0.75 & \text{si } Fr_j = 1 \text{ o no definido} \\
0.80 & \text{si } Fr_j < 1 \text{ (subcrítico)} \\
0.70 & \text{si } Fr_j > 1 \text{ (supercrítico)}
\end{cases}
$$

### 8.2 Costo de reposición

$$
C_{infra}(T_r) = \sum_{j \in \mathcal{F}(T_r)} c_{rep}(D_j^{req}, Z_j) \cdot L_j
$$

---

## 9. Implementación Software

### 9.1 Estructura de módulos

| Módulo | Descripción |
|--------|-------------|
| `rut_10_run_tanque_tormenta.py` | Orquestador principal |
| `rut_15_optimizer.py` | Greedy y NSGA-II |
| `rut_16_dynamic_evaluator.py` | Evaluador dinámico |
| `rut_23_nsga_optimizer.py` | NSGA-II para tanques |
| `rut_29_nsga_ranking_optimizer.py` | NSGA-II para pesos |
| `rut_00_path_finder.py` | Dijkstra sobre OSM |
| `rut_03_run_sewer_design.py` | Diseño hidráulico |
| `rut_18_itzi_flood_model.py` | Simulación 2D |
| `rut_19_flood_damage_climada.py` | CLIMADA/JRC |
| `rut_21_risk_analysis.py` | Análisis EAD |

### 9.2 Uso

```python
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
import config

# Configurar análisis probabilístico
config.TR_LIST = [1, 2, 5, 10, 25, 50, 100]

runner = StormwaterOptimizationRunner(
    project_root=config.PROJECT_ROOT,
    eval_id="analisis_probabilistico"
)

# Ejecutar optimización NSGA-II
resultado = runner.run_sequential_analysis(
    max_tanks=20,
    optimizer_mode='nsga',
    optimization_tr_list=[1, 2, 5, 10, 25],
    validation_tr_list=[1, 2, 5, 10, 25, 50, 100]
)
```

---

## 10. Referencias

1. EPA. (2022). *Storm Water Management Model (SWMM) Version 5.2*. Environmental Protection Agency.

2. Huizinga, J., Moel, H. de, & Szewczyk, W. (2017). *Global flood depth-damage functions: Methodology and the database with guidelines*. EU-JRC Technical Report.

3. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

4. Aznar, B., & Barnadas, M. (2016). *Itzi (version 17.1) [software]*.

5. OpenStreetMap Contributors. *OpenStreetMap Data*. https://www.openstreetmap.org/

---

## Licencia

MIT License
