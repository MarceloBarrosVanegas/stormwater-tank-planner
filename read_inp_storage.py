import re

inp_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.inp'

with open(inp_file, 'r') as f:
    content = f.read()

print("=" * 100)
print("PROPIEDADES DE TANQUES EN EL ARCHIVO .inp")
print("=" * 100)

# Buscar sección [STORAGE]
storage_match = re.search(r'\[STORAGE\](.*?)(?=\[|\Z)', content, re.DOTALL)
if storage_match:
    storage_section = storage_match.group(1).strip()
    print("\n--- Sección [STORAGE] ---")
    lines = storage_section.split('\n')
    tank_count = 0
    for line in lines:
        if line.strip() and not line.startswith(';'):
            parts = line.split()
            if len(parts) >= 6 and 'tank_' in line:
                tank_count += 1
                tank_name = parts[0]
                elev = parts[1]
                max_depth = parts[2]
                init_depth = parts[3]
                shape = parts[4]
                params = ' '.join(parts[5:])
                print(f"{tank_name:12} | Elev: {elev:8} | MaxD: {max_depth:8} | Shape: {shape:8} | Params: {params[:40]}")

print(f"\nTotal tanques encontrados en [STORAGE]: {tank_count}")

print("\n" + "=" * 100)
print("CALCULO DE VOLUMENES TEORICOS")
print("=" * 100)

# Vamos a buscar las curvas de almacenamiento
print("\n--- Sección [CURVES] (primeras relacionadas con tanques) ---")
curves_match = re.search(r'\[CURVES\](.*?)(?=\[|\Z)', content, re.DOTALL)
if curves_match:
    curves_section = curves_match.group(1).strip()
    lines = curves_section.split('\n')
    for line in lines[:50]:
        if 'tank' in line.lower() or 'storage' in line.lower():
            print(line)
