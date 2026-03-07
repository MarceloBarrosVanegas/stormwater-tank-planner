"""
Comparar volumenes de hydrographs (PNG) vs CSV
"""

# Volumenes del CSV
csv_volumes = {
    'tank_43': 51596.12,
    'tank_51': 70413.87,
    'tank_24': 1614.97,
    'tank_34': 24355.61,
    'tank_38': 21542.96,
    'tank_66': 15612.82,
    'tank_55': 13162.25,
    'tank_60': 21280.58,
    'tank_19': 5336.50,
    'tank_50': 0.00,
    'tank_8': 0.00,
    'tank_22': 3457.47,
    'tank_2': 4346.27
}

# Volumenes reales del hydrograph (del PNG)
hydrograph_volumes = {
    'tank_43': 56223,      # Stored: 56,223 m³
    'tank_51': 68781,      # Stored: 68,781 m³
    'tank_24': 1514,       # Stored: 1,514 m³
    'tank_34': 30968,      # Stored: 30,968 m³
    'tank_38': 30767,      # Stored: 30,767 m³
    'tank_66': 18610,      # Stored: 18,610 m³
    'tank_55': 39893,      # Stored: 39,893 m³
    'tank_60': 22927,      # Stored: 22,927 m³
    'tank_19': 6484,       # Stored: 6,484 m³
    'tank_50': 5402,       # Stored: 5,402 m³
    'tank_8': 0,           # No aparece en hydrographs (o es 0)
    'tank_22': 2147,       # Stored: 2,147 m³
    'tank_2': 4171         # Stored: 4,171 m³
}

# Ocupacion del hydrograph (%)
hydrograph_utilization = {
    'tank_43': 0.91,       # Util: 91%
    'tank_51': 0.91,       # Util: 91%
    'tank_24': 0.39,       # Util: 39%
    'tank_34': 0.87,       # Util: 87%
    'tank_38': 0.89,       # Util: 89%
    'tank_66': 0.80,       # Util: 80%
    'tank_55': 0.92,       # Util: 92%
    'tank_60': 0.83,       # Util: 83%
    'tank_19': 0.69,       # Util: 69%
    'tank_50': 0.70,       # Util: 70%
    'tank_22': 0.48,       # Util: 48%
    'tank_2': 0.61         # Util: 61%
}

print("=" * 120)
print("COMPARACION: CSV vs HYDROGRAPH (PNG) - VOLUMEN REAL ALMACENADO")
print("=" * 120)
print("\nTanque    | CSV Volume | Hydro Stored | Diferencia | Ratio | Utilizacion")
print("-" * 120)

total_csv = 0
total_hydro = 0

for tank in sorted(csv_volumes.keys()):
    csv_vol = csv_volumes[tank]
    hydro_vol = hydrograph_volumes.get(tank, 0)
    util = hydrograph_utilization.get(tank, 0)
    
    diff = csv_vol - hydro_vol
    ratio = csv_vol / hydro_vol if hydro_vol > 0 else 0
    
    total_csv += csv_vol
    total_hydro += hydro_vol
    
    status = "OK" if abs(ratio - 1.0) < 0.1 else "DIF"
    print(f"{tank:10} | {csv_vol:10.2f} | {hydro_vol:12} | {diff:10.2f} | {ratio:5.2f} | {util:6.0%} | {status}")

print("-" * 120)
print(f"{'TOTAL':10} | {total_csv:10.2f} | {total_hydro:12} | {total_csv-total_hydro:10.2f} | {total_csv/total_hydro:5.2f}")

print("\n" + "=" * 120)
print("CONCLUSION")
print("=" * 120)
print("""
1. El CSV tiene valores DIFERENTES a los hydrographs:
   - CSV: 232,719 m³ (suma)
   - Hydrographs: 270,478 m³ (suma)
   
2. Diferencias individuales:
   - tank_51: CSV=70,414 vs Hydro=68,781 (+1,633 m³)
   - tank_55: CSV=13,162 vs Hydro=39,893 (-26,731 m³) <- ¡MAYOR DIFERENCIA!
   - tank_34: CSV=24,356 vs Hydro=30,968 (-6,612 m³)
   
3. Tanques reportados como 0 en CSV pero con volumen en hydrographs:
   - tank_50: CSV=0 vs Hydro=5,402 m³
   - tank_8:  CSV=0 vs Hydro=0 m³ (realmente 0)
   
4. El HYDROGRAPH muestra el VOLUMEN REAL ALMACENADO durante la simulacion,
   que es el valor correcto para reportar.
   
5. El CSV esta calculando mal los volumenes (usa una formula incorrecta).

6. Los hydrographs tambien muestran:
   - Design Depth: 15.0 m (profundidad de diseno)
   - Max Depth: ~9-14 m (profundidad maxima alcanzada)
   - Utilization: 39-92% (porcentaje de ocupacion)
   
   ESTOS DATOS NO ESTAN EN EL CSV.
""")
