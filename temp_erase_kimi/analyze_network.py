import sqlite3
import json

# Conectar al GPKG
db_path = 'C:/Users/Alienware/OneDrive/SANTA_ISABEL/00_tanque_tormenta/codigos/optimization_results_75_25/Seq_Iter_01/Seq_Iter_01.gpkg'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Obtener todos los datos relevantes
cursor.execute("""
    SELECT 
        fid,
        Ramal,
        Tramo,
        Pozo,
        Conexion,
        Derivacion,
        Obs,
        L,
        Estado
    FROM Seq_Iter_01
    ORDER BY Ramal, Tramo
""")

rows = cursor.fetchall()

# Analizar la red
print("=" * 100)
print("ANÁLISIS DE LA RED DE DERIVACIONES")
print("=" * 100)

# Diccionario para agrupar por target_id (tanque)
grupos_por_tanque = {}

for row in rows:
    fid, ramal, tramo, pozo, conexion, derivacion, obs, length, estado = row
    
    # Parsear JSON del campo Obs
    try:
        if obs and '|' in obs:
            json_part = obs.split('|')[1]
            data = json.loads(json_part)
            target_id = data.get('target_id')
            target_type = data.get('target_type')
            node_id = data.get('node_id')
            node_ramal = data.get('node_ramal')
            target_ramal = data.get('target_ramal')
            
            if target_id not in grupos_por_tanque:
                grupos_por_tanque[target_id] = {
                    'ramales': set(),
                    'tramos': [],
                    'conexiones': set(),
                    'target_info': data
                }
            
            grupos_por_tanque[target_id]['ramales'].add(ramal)
            grupos_por_tanque[target_id]['tramos'].append({
                'fid': fid,
                'ramal': ramal,
                'tramo': tramo,
                'pozo': pozo,
                'conexion': conexion,
                'derivacion': derivacion,
                'length': length,
                'node_id': node_id
            })
            if conexion:
                grupos_por_tanque[target_id]['conexiones'].add(conexion)
                
    except Exception as e:
        pass

# Mostrar grupos
print(f"\nTotal de tanques/grupos encontrados: {len(grupos_por_tanque)}")
print()

for target_id, info in grupos_por_tanque.items():
    target_info = info['target_info']
    print(f"\n{'='*80}")
    print(f"GRUPO: Tanque en Predio {target_id}")
    print(f"{'='*80}")
    print(f"  Node ID: {target_info.get('node_id')}")
    print(f"  Volumen: {target_info.get('target_total_volume', 0):.2f} m³")
    print(f"  Elevación fondo: {target_info.get('target_invert_elevation', 0):.2f} m")
    print(f"  Ramales involucrados: {sorted(info['ramales'])}")
    print(f"  Conexiones entre ramales: {sorted(info['conexiones']) if info['conexiones'] else 'Ninguna'}")
    print(f"  Total tramos: {len(info['tramos'])}")
    
    # Mostrar estructura jerárquica
    print(f"\n  Estructura de la red:")
    for tr in info['tramos'][:5]:  # Solo primeros 5 para no saturar
        deriv_str = " [DERIV]" if tr['derivacion'] else ""
        conn_str = f" -> {tr['conexion']}" if tr['conexion'] else ""
        print(f"    Ramal {tr['ramal']:>3} | Tramo {tr['tramo']:>6} | Pozo {tr['pozo']:>6} | L={tr['length']:>6.1f}m{deriv_str}{conn_str}")
    if len(info['tramos']) > 5:
        print(f"    ... y {len(info['tramos']) - 5} tramos más")

conn.close()
print("\n" + "=" * 100)
