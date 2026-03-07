"""
Extraer datos del archivo SWMM .out para verificar valores reales
"""
import sys
sys.path.insert(0, 'venv_deadcode\\Lib\\site-packages')

from swmmio import Model
from pyswmm import Output
import pandas as pd

out_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.out'
inp_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.inp'

print("=" * 100)
print("EXTRACCIÓN DE DATOS DEL ARCHIVO SWMM .out")
print("=" * 100)
print(f"Archivo: {out_file}")

# Usar pyswmm para extraer datos
try:
    with Output(out_file) as out:
        print("\n--- Nodos disponibles ---")
        nodes = list(out.nodes)
        print(f"Total nodos: {len(nodes)}")
        print(f"Primeros 10 nodos: {nodes[:10]}")
        
        # Buscar nodos de tanques (usualmente contienen "TANK" o similar)
        tank_nodes = [n for n in nodes if 'TANK' in str(n).upper() or 'T' in str(n).upper()]
        print(f"\nPosibles nodos de tanques: {tank_nodes[:20]}")
        
        # Extraer datos de volumen de flooding por nodo
        print("\n--- Flooding Volume por Nodo ---")
        flooding_data = []
        for node in nodes:
            try:
                flood_vol = out.node_series(node, 'FLOODING_VOLUME')
                if flood_vol and len(flood_vol) > 0:
                    total_flood = sum(flood_vol.values())
                    if total_flood > 0:
                        flooding_data.append({
                            'node': node,
                            'total_flooding_volume': total_flood
                        })
            except:
                pass
        
        df_flooding = pd.DataFrame(flooding_data)
        if len(df_flooding) > 0:
            df_flooding = df_flooding.sort_values('total_flooding_volume', ascending=False)
            print(f"\nTop 20 nodos con más flooding:")
            print(df_flooding.head(20).to_string())
            print(f"\nTotal flooding volume: {df_flooding['total_flooding_volume'].sum():.2f} m³")
        
        # Extraer datos de links
        print("\n--- Links disponibles ---")
        links = list(out.links)
        print(f"Total links: {len(links)}")
        print(f"Primeros 20 links: {links[:20]}")
        
        # Buscar links de derivación
        derivation_links = [l for l in links if 'DERIV' in str(l).upper() or 'DIV' in str(l).upper() or '.1' in str(l)]
        print(f"\nPosibles links de derivación: {derivation_links[:20]}")
        
        # Extraer caudales máximos por link
        print("\n--- Caudales Máximos por Link ---")
        flow_data = []
        for link in links:
            try:
                flow_series = out.link_series(link, 'FLOW')
                if flow_series and len(flow_series) > 0:
                    max_flow = max(flow_series.values())
                    if max_flow > 0:
                        flow_data.append({
                            'link': link,
                            'max_flow': max_flow
                        })
            except:
                pass
        
        df_flow = pd.DataFrame(flow_data)
        if len(df_flow) > 0:
            df_flow = df_flow.sort_values('max_flow', ascending=False)
            print(f"\nTop 30 links con mayor caudal:")
            print(df_flow.head(30).to_string())
        
        # Extraer datos de tanques de almacenamiento (storage)
        print("\n--- Storage Units (Unidades de Almacenamiento) ---")
        storage_nodes = []
        for node in nodes:
            try:
                # Intentar obtener volumen de almacenamiento
                depth_series = out.node_series(node, 'DEPTH')
                if depth_series and len(depth_series) > 0:
                    max_depth = max(depth_series.values())
                    if max_depth > 0:
                        storage_nodes.append({
                            'node': node,
                            'max_depth': max_depth
                        })
            except:
                pass
        
        df_storage = pd.DataFrame(storage_nodes)
        if len(df_storage) > 0:
            df_storage = df_storage.sort_values('max_depth', ascending=False)
            print(f"\nNodos con mayor profundidad (posibles tanques):")
            print(df_storage.head(20).to_string())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
