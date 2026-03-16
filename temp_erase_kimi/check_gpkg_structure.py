import sqlite3
import json

# Conectar al GPKG
db_path = 'C:/Users/Alienware/OneDrive/SANTA_ISABEL/00_tanque_tormenta/codigos/optimization_results_75_25/Seq_Iter_01/Seq_Iter_01.gpkg'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Ver tablas
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tablas:', [t[0] for t in tables])

# Ver estructura de cada tabla
for table in [t[0] for t in tables]:
    print(f'\n=== {table} ===')
    cursor.execute(f'PRAGMA table_info({table})')
    cols = cursor.fetchall()
    for col in cols:
        print(f'  {col[1]}: {col[2]}')
    
    # Ver algunos datos
    try:
        cursor.execute(f'SELECT * FROM {table} LIMIT 2')
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            print(f'  Row {i}: {row}')
    except:
        pass

conn.close()
