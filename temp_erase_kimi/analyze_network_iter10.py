import sqlite3
import json

# Conectar al GPKG de iteración 10 (debería tener más tanques)
db_path = 'C:/Users/Alienware/OneDrive/SANTA_ISABEL/00_tanque_tormenta/codigos/optimization_results_75_25/Seq_Iter_10/Seq_Iter_10.gpkg'
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
    FROM Seq_Iter_10
    ORDER BY Ramal, Tramo
""")

rows = cursor.fetchall()

# Analizar la red
print("=" * 100)
print("ANÁLISIS DE LA RED - ITERACIÓN 10 (10 Tanques)")
print("=" * 100)

# Diccionario para agrupar por target_id (tanque)
grupos_por_tanque = {}
ramales_info = {}

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
            
            if target_id not in grupos_por_tanque:
                grupos_por_tanque[target_id] = {
                    'ramales': set(),
                    'tramos': [],
                    'conexiones': set(),
                    'target_node': node_id,
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
            if conexion and conexion != 'None':
                grupos_por_tanque[target_id]['conexiones'].add(conexion)
                
    except Exception as e:
        pass

# Mostrar grupos
print(f"\nTotal de tanques/grupos encontrados: {len(grupos_por_tanque)}")
print()

for target_id, info in sorted(grupos_por_tanque.items(), key=lambda x: str(x[0])):
    target_info = info['target_info']
    total_length = sum(t['length'] for t in info['tramos'])
    
    print(f"\n{'='*80}")
    print(f"GRUPO: Tanque en Predio {target_id} (Node: {info['target_node']})")
    print(f"{'='*80}")
    print(f"  Volumen tanque: {target_info.get('target_total_volume', 0):.2f} m³")
    print(f"  Ramales: {sorted(info['ramales'])}")
    print(f"  Total tramos: {len(info['tramos'])}")
    print(f"  Longitud total tuberías: {total_length:.1f} m")
    print(f"  Conexiones entre ramales: {sorted(info['conexiones']) if info['conexiones'] else 'Ninguna'}")

conn.close()
print("\n" + "=" * 100)
