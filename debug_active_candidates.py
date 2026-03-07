import pandas as pd

# Leer el CSV del step 1
df = pd.read_csv('optimization_results/Seq_Iter_01/step_01_summary.csv')
print("Step 01 Summary:")
print(df[['step', 'added_node', 'added_predio', 'current_tank_volume', 'total_tank_volume']].to_string())

# Verificar si el archivo model_Seq_Iter_01.out existe
import os
out_file = 'optimization_results/Seq_Iter_01/model_Seq_Iter_01.out'
print(f"\nArchivo .out existe: {os.path.exists(out_file)}")

# Leer el inp file para ver qué tanques tiene
inp_file = 'optimization_results/Seq_Iter_01/model_Seq_Iter_01.inp'
if os.path.exists(inp_file):
    with open(inp_file, 'r') as f:
        content = f.read()
        if '[STORAGE]' in content:
            storage_section = content.split('[STORAGE]')[1].split('[')[0]
            print(f"\nTanques en archivo INP:")
            for line in storage_section.strip().split('\n')[:15]:
                if line.strip() and not line.startswith(';'):
                    print(f"  {line}")
