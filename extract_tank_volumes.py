"""
Extraer volúmenes específicos de tanques del archivo SWMM .out
"""
import sys
sys.path.insert(0, 'venv_deadcode\\Lib\\site-packages')

from pyswmm import Output
import pandas as pd

out_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.out'

print("=" * 100)
print("ANÁLISIS DE TANQUES - DATOS REALES DEL ARCHIVO .out")
print("=" * 100)

with Output(out_file) as out:
    # Lista de tanques según el CSV
    tank_nodes_from_csv = [
        'tank_43', 'tank_51', 'tank_24', 'tank_34', 'tank_38', 
        'tank_66', 'tank_55', 'tank_60', 'tank_19', 'tank_50',
        'tank_8', 'tank_22', 'tank_2'
    ]
    
    print("\n--- Tanques encontrados en el modelo ---")
    all_nodes = list(out.nodes)
    tank_nodes = [n for n in all_nodes if 'tank_' in str(n)]
    print(f"Total tanques en el modelo: {len(tank_nodes)}")
    print(f"Tanques: {tank_nodes}")
    
    # Extraer datos de cada tanque
    print("\n--- DATOS DE TANQUES (del archivo .out) ---")
    print("Tanque    | Max Depth | Flooding Volume | Inflow Volume | Overflow")
    print("-" * 80)
    
    tank_data = []
    for tank in tank_nodes:
        try:
            # Profundidad máxima
            depth_series = out.node_series(tank, 'DEPTH')
            max_depth = max(depth_series.values()) if depth_series else 0
            
            # Volumen de flooding
            flood_series = out.node_series(tank, 'FLOODING_VOLUME')
            flood_vol = sum(flood_series.values()) if flood_series else 0
            
            # Volumen de inflow (entrada total)
            inflow_series = out.node_series(tank, 'TOTAL_INFLOW')
            inflow_vol = sum(inflow_series.values()) if inflow_series else 0
            
            # Overflow (exceso)
            overflow_series = out.node_series(tank, 'OVERFLOW')
            overflow_vol = sum(overflow_series.values()) if overflow_series else 0
            
            print(f"{tank:10} | {max_depth:9.2f} | {flood_vol:15.2f} | {inflow_vol:13.2f} | {overflow_vol:8.2f}")
            
            tank_data.append({
                'tank': tank,
                'max_depth_m': max_depth,
                'flooding_volume_m3': flood_vol,
                'inflow_volume_m3': inflow_vol,
                'overflow_volume_m3': overflow_vol
            })
        except Exception as e:
            print(f"{tank:10} | ERROR: {e}")
    
    df = pd.DataFrame(tank_data)
    
    print("\n--- COMPARACIÓN CON CSV ---")
    print("\nValores del CSV (current_tank_volume):")
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
        'tank_50': 0,  # No aparece con volumen > 0 en steps finales
        'tank_8': 0,
        'tank_22': 3457.47,
        'tank_2': 4346.27
    }
    
    print("\nTanque    | CSV Volume | OUT Inflow  | Ratio")
    print("-" * 60)
    for tank in tank_nodes:
        csv_vol = csv_volumes.get(tank, 0)
        out_data = df[df['tank'] == tank]
        if len(out_data) > 0:
            out_inflow = out_data.iloc[0]['inflow_volume_m3']
            ratio = csv_vol / out_inflow if out_inflow > 0 else 0
            print(f"{tank:10} | {csv_vol:10.2f} | {out_inflow:11.2f} | {ratio:5.2f}")
    
    # Calcular volumen total de flooding
    total_flooding = df['flooding_volume_m3'].sum()
    print(f"\n--- TOTALES ---")
    print(f"Total flooding volume (todos los tanques): {total_flooding:.2f} m³")
    
    # Extraer datos de los links de derivación específicos
    print("\n--- LINKS DE DERIVACIÓN ESPECÍFICOS ---")
    derivation_links = [l for l in out.links if 'tank_' in str(l) or l.startswith(('1.', '2.', '3.'))]
    print(f"Links de derivación encontrados: {len(derivation_links)}")
    print(f"Primeros 20: {derivation_links[:20]}")
    
    # Caudales máximos en links de derivación
    print("\n--- CAUDALES EN LINKS DE DERIVACIÓN ---")
    deriv_flows = []
    for link in derivation_links[:50]:  # Limitar a 50
        try:
            flow_series = out.link_series(link, 'FLOW')
            if flow_series:
                max_flow = max(flow_series.values())
                if max_flow > 0.1:  # Solo mostrar si hay flujo significativo
                    deriv_flows.append({
                        'link': link,
                        'max_flow_m3s': max_flow
                    })
        except:
            pass
    
    df_deriv = pd.DataFrame(deriv_flows)
    if len(df_deriv) > 0:
        df_deriv = df_deriv.sort_values('max_flow_m3s', ascending=False)
        print("\nTop 20 links de derivación con mayor caudal:")
        print(df_deriv.head(20).to_string(index=False))

print("\n" + "=" * 100)
print("CONCLUSIÓN")
print("=" * 100)
print(f"""
1. Los tanques en el archivo .out muestran:
   - Flooding volume (volumen que rebosa del tanque): {total_flooding:.2f} m³ total
   - Inflow volume (volumen que entra al tanque): Variable por tanque
   
2. El CSV reporta "current_tank_volume" que parece ser:
   - El volumen de almacenamiento máximo del tanque (no el flooding)
   
3. Los valores NO coinciden directamente:
   - CSV: current_tank_volume ≈ 51,596 m³ para tank_43
   - OUT: inflow_volume ≈ 3,000-4,000 m³ para tank_43
   
4. La diferencia sugiere que el CSV está usando un cálculo diferente,
   posiblemente basado en el volumen físico máximo del tanque, no el 
   volumen de agua que pasó por él.
""")
