import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('optimization_results/sequence_tracking.csv')

print("=" * 80)
print("ANÁLISIS DE SEQUENCE_TRACKING.CSV")
print("=" * 80)
print(f"\nColumnas: {list(df.columns)}")
print(f"\nFilas: {len(df)}")

print("\n" + "=" * 80)
print("VERIFICACIÓN DE CÁLCULOS MARGINALES")
print("=" * 80)

# Calcular diferencias marginales
df['calc_marginal_reduction'] = df['flooding_reduction'].diff().fillna(df['flooding_reduction'].iloc[0])
df['calc_marginal_cost'] = df['cost_investment_total'].diff().fillna(df['cost_investment_total'].iloc[0])

# Comparar con los valores reportados
print("\n--- Comparación de Reducción Marginal ---")
print("Step | Reportado | Calculado | Diferencia | Coincide?")
print("-" * 60)
for i, row in df.iterrows():
    reportado = row['marginal_reduction']
    calculado = row['calc_marginal_reduction']
    diff = abs(reportado - calculado)
    coincide = "OK" if diff < 0.1 else "DIF"
    print(f"{int(row['step']):4d} | {reportado:9.2f} | {calculado:9.2f} | {diff:10.2f} | {coincide}")

print("\n--- Análisis de Costos ---")
print("Step | Costo Links  | Costo Tanks  | Costo Land   | Suma vs Total | Diferencia")
print("-" * 85)
for i, row in df.iterrows():
    suma = row['cost_links'] + row['cost_tanks'] + row['cost_land']
    total = row['cost_investment_total']
    diff = total - suma
    status = "OK" if abs(diff) < 1 else "REVISAR"
    print(f"{int(row['step']):4d} | ${row['cost_links']:11,.2f} | ${row['cost_tanks']:11,.2f} | ${row['cost_land']:11,.2f} | ${diff:12,.2f} | {status}")

print("\n--- Costo Marginal por Paso ($) ---")
print("Step | Costo Marginal | Reducción Marginal | $/m3 Marginal")
print("-" * 70)
for i, row in df.iterrows():
    if i == 0:
        cost_marg = row['cost_investment_total']
        red_marg = row['marginal_reduction']
    else:
        cost_marg = row['cost_investment_total'] - df.iloc[i-1]['cost_investment_total']
        red_marg = row['marginal_reduction']
    
    if red_marg > 0:
        cost_per_m3 = cost_marg / red_marg
    else:
        cost_per_m3 = float('inf')
    
    print(f"{int(row['step']):4d} | ${cost_marg:14,.2f} | {red_marg:18.2f} | ${cost_per_m3:12,.2f}")

print("\n" + "=" * 80)
print("ANÁLISIS DE STEPS CON VALORES EXTRAÑOS")
print("=" * 80)

# Buscar steps donde la reducción marginal es muy pequeña pero el costo es alto
print("\n--- Steps con baja eficiencia marginal ---")
for i, row in df.iterrows():
    if i == 0:
        cost_marg = row['cost_investment_total']
    else:
        cost_marg = row['cost_investment_total'] - df.iloc[i-1]['cost_investment_total']
    
    red_marg = row['marginal_reduction']
    
    if red_marg > 0:
        eff = cost_marg / red_marg
        if eff > 5000:  # Más de $5000 por m3
            print(f"Step {int(row['step'])}: ${eff:,.2f}/m3 (costo marg: ${cost_marg:,.2f}, reduc: {red_marg:.2f} m3)")

print("\n--- Steps donde n_tanks no aumenta (mismo número de tanques) ---")
for i in range(1, len(df)):
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    if curr['n_tanks'] == prev['n_tanks']:
        print(f"Step {int(curr['step'])}: n_tanks = {int(curr['n_tanks'])} (igual que step {int(prev['step'])})")
        print(f"  - Costo aumentó: ${curr['cost_investment_total'] - prev['cost_investment_total']:,.2f}")
        print(f"  - Reducción: {curr['marginal_reduction']:.2f} m3")
