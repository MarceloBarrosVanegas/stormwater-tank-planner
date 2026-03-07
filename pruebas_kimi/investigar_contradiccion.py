"""
Investigar la contradiccion entre utilizacion media y distribucion.
"""

import pandas as pd
import numpy as np

# Simular datos que expliquen la contradiccion
print("=" * 70)
print("INVESTIGANDO LA CONTRADICCION")
print("=" * 70)
print()

# Escenario: la mayoria de links mejoran, pero algunos empeoran mucho
print("HIPOTESIS: Outliers que empeoran distorsionan el promedio")
print()

# Crear datos de ejemplo
np.random.seed(42)
n_links = 1174

# Baseline: distribucion con cola larga
baseline_util = np.random.gamma(2, 3, n_links)  # Media ~6 (600%)
baseline_util = np.clip(baseline_util, 0, 20)

# Solucion: la mayoria baja, pero algunos suben mucho
solution_util = baseline_util.copy()
# 80% de links mejoran (reducen h/D)
mask_improve = np.random.random(n_links) < 0.8
solution_util[mask_improve] = solution_util[mask_improve] * 0.6
# 20% de links empeoran (aumentan h/D) - outliers
solution_util[~mask_improve] = solution_util[~mask_improve] * 2.5

print(f"Baseline - Media: {baseline_util.mean()*100:.1f}%, Mediana: {np.median(baseline_util)*100:.1f}%")
print(f"Solution - Media: {solution_util.mean()*100:.1f}%, Mediana: {np.median(solution_util)*100:.1f}%")
print()

# Contar links por rangos
print("Distribucion por rangos de h/D:")
print(f"{'Rango':<15} {'Baseline':<12} {'Solution':<12} {'Cambio'}")
print("-" * 55)
for low, high in [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 20)]:
    count_base = ((baseline_util >= low) & (baseline_util < high)).sum()
    count_sol = ((solution_util >= low) & (solution_util < high)).sum()
    change = count_sol - count_base
    print(f"{low:.1f}-{high:.1f}:        {count_base:>5}       {count_sol:>5}      {change:>+5}")

print()
print("=" * 70)
print("CONCLUSION:")
print("Si algunos links empeoran mucho mientras la mayoria mejora,")
print("el promedio puede aumentar aunque la distribucion muestre mejora.")
print("=" * 70)
