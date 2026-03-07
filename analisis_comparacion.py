import pandas as pd

# Cargar CSV principal
df = pd.read_csv('optimization_results/sequence_tracking.csv')

print("=" * 120)
print("COMPARACIÓN DE VALORES - CSV vs DASHBOARD")
print("=" * 120)

# Mostrar los datos de los steps 
print("\n--- Valores del CSV sequence_tracking.csv ---")
print("Step | n_tanks | added_predio | current_tank_volume | flooding_reduction | marginal_reduction | cost_investment_total")
print("-" * 120)

for i, row in df.iterrows():
    print(f"{int(row['step']):4d} | {int(row['n_tanks']):7d} | {row['added_predio']:12} | {row['current_tank_volume']:19.2f} | {row['flooding_reduction']:18.2f} | {row['marginal_reduction']:18.2f} | ${row['cost_investment_total']:15,.2f}")

print("\n" + "=" * 120)
print("PROBLEMAS DETECTADOS:")
print("=" * 120)

# 1. Steps donde current_tank_volume es 0 pero hay costo
print("\n1. Steps donde current_tank_volume = 0 pero hay costo:")
for i, row in df.iterrows():
    if row['current_tank_volume'] == 0 and row['cost_investment_total'] > 0:
        print(f"   Step {int(row['step'])}: Vol=0, Costo=${row['cost_investment_total']:,.2f}, added_predio={row['added_predio']}")

# 2. Verificar consistencia de n_tanks
print("\n2. Conteo de incrementos en n_tanks:")
prev_tanks = 0
for i, row in df.iterrows():
    if row['n_tanks'] > prev_tanks:
        prev_tanks = int(row['n_tanks'])
        print(f"   Step {int(row['step'])}: n_tanks aumentó a {int(row['n_tanks'])}")

# 3. Suma de volúmenes individuales vs total
print("\n3. Suma de current_tank_volume vs total_tank_volume:")
sum_individual = df['current_tank_volume'].sum()
last_total = df.iloc[-1]['total_tank_volume']
print(f"   Suma de volúmenes individuales: {sum_individual:.2f} m³")
print(f"   Total reportado (último step):  {last_total:.2f} m³")
print(f"   Diferencia: {abs(sum_individual - last_total):.2f} m³")

# 4. Comparar valores del dashboard vs CSV
print("\n4. Comparación con valores del dashboard (de la imagen):")
print("   En la imagen 00_dashboard_map.png, la tabla muestra:")
print("   - Paso 1: Predio 43, Volumen=56,223 m³, Q=37.82 m³/s, L=1.00 km")
print("   - Paso 2: Predio 51, Volumen=68,781 m³, Q=16.96 m³/s, L=0.38 km")
print("   - Paso 3: Predio 24, Volumen=1,514 m³, Q=2.10 m³/s, L=0.79 km")
print()
print("   Valores en CSV:")
for i in [0, 1, 2]:  # Steps 1, 2, 3
    row = df.iloc[i]
    print(f"   - Step {int(row['step'])}: added_predio={row['added_predio']}, current_tank_volume={row['current_tank_volume']:.2f} m³")

print("\n   ⚠️  DIFERENCIAS DETECTADAS:")
print("   - CSV Step 1: current_tank_volume = 51,596.12 m³ vs Dashboard = 56,223 m³")
print("   - CSV Step 2: current_tank_volume = 70,413.87 m³ vs Dashboard = 68,781 m³")
print("   - CSV Step 3: current_tank_volume = 1,614.97 m³ vs Dashboard = 1,514 m³")
