"""
Busqueda por Objetivos - Encontrar que tanques logran cierto umbral de mejora
"""

import pandas as pd
import json
from pathlib import Path

# Cargar datos de tracking
csv_path = Path('C:/Users/Alienware/OneDrive/SANTA_ISABEL/00_tanque_tormenta/codigos/optimization_results_75_25/sequence_tracking.csv')
df = pd.read_csv(csv_path)

print("=" * 100)
print("BUSQUEDA POR OBJETIVOS")
print("=" * 100)

# Valores baseline
baseline_flooding = df.iloc[0]['flooding_volume']
baseline_outfall = df.iloc[0]['outfall_peak_flow']
baseline_nodes = df.iloc[0]['flooded_nodes_count']

print(f"\nBASELINE (Paso 0):")
print(f"  Flooding: {baseline_flooding:,.0f} m3")
print(f"  Outfall: {baseline_outfall:.1f} m3/s")
print(f"  Nodos inundados: {baseline_nodes:.0f}")

# Funcion para encontrar paso que alcanza objetivo
def find_step_for_objective(df, column, target_value, mode='less_than'):
    """
    Encuentra el primer paso donde se alcanza el objetivo
    mode: 'less_than' o 'greater_than'
    """
    if mode == 'less_than':
        mask = df[column] <= target_value
    else:
        mask = df[column] >= target_value
    
    valid_steps = df[mask]
    if len(valid_steps) > 0:
        return valid_steps.iloc[0]
    return None

# Ejemplo 1: Reducir flooding a cierto umbral
print("\n" + "=" * 100)
print("EJEMPLO 1: Reducir Flooding Volume")
print("=" * 100)

objetivos_flooding = [
    baseline_flooding * 0.75,  # Reducir 25%
    baseline_flooding * 0.50,  # Reducir 50%
    baseline_flooding * 0.25,  # Reducir 75%
]

for i, objetivo in enumerate(objetivos_flooding):
    result = find_step_for_objective(df, 'flooding_volume', objetivo)
    if result is not None:
        step = int(result['step'])
        n_tanks = int(result['n_tanks'])
        flooding = result['flooding_volume']
        reduction_pct = (1 - flooding/baseline_flooding) * 100
        
        target_pct = (1 - objetivo/baseline_flooding) * 100
        print(f"\nObjetivo {i+1}: Flooding <= {objetivo:,.0f} m3 (reduccion >= {target_pct:.0f}%)")
        print(f"  -> Alcanzado en Paso {step} con {n_tanks} tanques")
        print(f"  Flooding real: {flooding:,.0f} m3 ({reduction_pct:.1f}% reduccion)")
        print(f"  Inversion acumulada: ${result['cost_investment_total']:,.0f}")
        
        # Identificar que tanque se agrego en este paso
        if step > 0:
            added_node = result.get('added_node', 'N/A')
            added_predio = result.get('added_predio', 'N/A')
            print(f"  Tanque clave: {added_node} (Predio {added_predio})")
    else:
        print(f"\nObjetivo {i+1}: {objetivo:,.0f} m3 - NO ALCANZADO en esta secuencia")

# Ejemplo 2: Reducir Outfall Peak Flow
print("\n" + "=" * 100)
print("EJEMPLO 2: Reducir Outfall Peak Flow")
print("=" * 100)

objetivos_outfall = [
    baseline_outfall * 0.90,  # Reducir 10%
    baseline_outfall * 0.80,  # Reducir 20%
    baseline_outfall * 0.70,  # Reducir 30%
]

for objetivo in objetivos_outfall:
    result = find_step_for_objective(df, 'outfall_peak_flow', objetivo)
    if result is not None:
        step = int(result['step'])
        n_tanks = int(result['n_tanks'])
        outfall = result['outfall_peak_flow']
        reduction_pct = (1 - outfall/baseline_outfall) * 100
        
        target_pct = (1 - objetivo/baseline_outfall) * 100
        print(f"\nObjetivo: Outfall <= {objetivo:.1f} m3/s (reduccion >= {target_pct:.0f}%)")
        print(f"  -> Alcanzado en Paso {step} con {n_tanks} tanques")
        print(f"  Outfall real: {outfall:.1f} m3/s ({reduction_pct:.1f}% reduccion)")

# Ejemplo 3: Encontrar el "punto optimo" (mejor relacion costo-beneficio)
print("\n" + "=" * 100)
print("EJEMPLO 3: Punto Optimon de Inversion")
print("=" * 100)

# Calcular eficiencia marginal por paso
df['marginal_cost'] = df['cost_investment_total'].diff().fillna(0)
df['flooding_reduction_pct'] = (1 - df['flooding_volume']/baseline_flooding) * 100
df['efficiency'] = df['flooding_reduction_pct'] / df['cost_investment_total'].replace(0, float('inf'))

# Encontrar el paso con mejor eficiencia (excluyendo paso 0)
best_step = df.iloc[1:]['efficiency'].idxmax()
best_row = df.loc[best_step]

print(f"\nMejor eficiencia en Paso {int(best_row['step'])}:")
print(f"  Reduccion: {best_row['flooding_reduction_pct']:.1f}%")
print(f"  Inversion: ${best_row['cost_investment_total']:,.0f}")
print(f"  Eficiencia: {best_row['efficiency']*1e6:.2f}% por millon de dolares")
print(f"  Tanque agregado: {best_row.get('added_node', 'N/A')} (Predio {best_row.get('added_predio', 'N/A')})")

# Mostrar todos los pasos ordenados por eficiencia
print("\n" + "=" * 100)
print("RANKING DE TANQUES POR EFICIENCIA (% reduccion / $ invertido)")
print("=" * 100)

df_sorted = df.iloc[1:].sort_values('efficiency', ascending=False)
print("\nTop 10 tanques mas eficientes:")
print("-" * 80)
print(f"{'Rank':<6}{'Paso':<6}{'Tanque':<15}{'Predio':<10}{'Reduccion %':<12}{'Inversion':<15}{'Eficiencia':<12}")
print("-" * 80)

for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    tank = str(row.get('added_node', 'N/A'))[:12]
    predio = str(row.get('added_predio', 'N/A'))[:8]
    reduc = row['flooding_reduction_pct']
    cost = row['cost_investment_total']
    eff = row['efficiency'] * 1e6
    print(f"{i:<6}{int(row['step']):<6}{tank:<15}{predio:<10}{reduc:<12.1f}${cost:<14,.0f}{eff:<12.2f}")

print("\n" + "=" * 100)
