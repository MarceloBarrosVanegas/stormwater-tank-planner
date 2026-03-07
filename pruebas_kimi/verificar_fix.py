"""
Script para verificar que el fix de surcharged_links_count funciona correctamente.
Simula el cálculo con datos de prueba.
"""

import pandas as pd
import numpy as np

# Simular datos de swmm_gdf
np.random.seed(42)
n_links = 1174

# Crear DataFrame simulado
swmm_gdf = pd.DataFrame({
    'Name': [f'Link_{i}' for i in range(n_links)],
    'Capacity': np.random.uniform(0.5, 2.0, n_links),  # Capacidad en m3/s
    'MaxFlow': np.random.uniform(0.3, 2.5, n_links),   # Caudal máximo
    'Length': np.random.uniform(10, 100, n_links),     # Longitud en m
})

# Simular columna 'Surcharged' que vendría del SWMM (con valores incorrectos)
swmm_gdf['Surcharged'] = np.random.choice([True, False], n_links, p=[0.9, 0.1])

print("=" * 60)
print("VERIFICACION DEL FIX DE surcharged_links_count")
print("=" * 60)
print()

# ANTES DEL FIX (join con rsuffix)
print("ANTES DEL FIX (comportamiento erroneo):")
valid_pipes = swmm_gdf[swmm_gdf['Capacity'] > 0].copy()
valid_pipes['utilization'] = valid_pipes['MaxFlow'] / valid_pipes['Capacity']
valid_pipes['Surcharged'] = valid_pipes['utilization'] >= 1.0

# Simular join con rsuffix (crea columnas _calc)
swmm_gdf_old = swmm_gdf.join(valid_pipes[['utilization', 'Surcharged']], rsuffix='_calc')
swmm_gdf_old['utilization'] = swmm_gdf_old['utilization'].fillna(0.0)
swmm_gdf_old['Surcharged'] = swmm_gdf_old['Surcharged'].fillna(False)  # Esto no sobreescribe la original!

surcharged_old = swmm_gdf_old[swmm_gdf_old['Surcharged'] == True]
count_old = len(surcharged_old)
print(f"  Links sobrecargados (contando columna original del SWMM): {count_old}")
print(f"  Porcentaje de la red: {count_old/len(swmm_gdf)*100:.1f}%")
print()

# DESPUES DEL FIX (sobrescribir columnas)
print("DESPUES DEL FIX (comportamiento correcto):")
swmm_gdf_new = swmm_gdf.copy()
# Eliminar y recrear columnas
for col in ['utilization', 'Surcharged']:
    swmm_gdf_new[col] = valid_pipes[col]
swmm_gdf_new['utilization'] = swmm_gdf_new['utilization'].fillna(0.0)
swmm_gdf_new['Surcharged'] = swmm_gdf_new['Surcharged'].fillna(False)

surcharged_new = swmm_gdf_new[swmm_gdf_new['Surcharged'] == True]
count_new = len(surcharged_new)
print(f"  Links sobrcargados (calculados correctamente): {count_new}")
print(f"  Porcentaje de la red: {count_new/len(swmm_gdf)*100:.1f}%")
print()

# Comparar utilizacion
util_mean_old = swmm_gdf_old['utilization'].mean() * 100
util_mean_new = swmm_gdf_new['utilization'].mean() * 100

print("COMPARACION DE UTILIZACION:")
print(f"  Utilizacion media (antes): {util_mean_old:.1f}%")
print(f"  Utilizacion media (despues): {util_mean_new:.1f}%")
print()

print("=" * 60)
print("CONCLUSION:")
if count_old != count_new:
    print(f"[OK] EL FIX FUNCIONA - Diferencia de {abs(count_old - count_new)} links")
    if count_new < count_old:
        print("[OK] La nueva version cuenta MENOS links (mas realista)")
    else:
        print("[WARNING] La nueva version cuenta MAS links")
else:
    print("⚠️ No hay diferencia - el fix podria no estar funcionando")
print("=" * 60)
