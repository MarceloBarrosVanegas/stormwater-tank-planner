import sys
sys.path.insert(0, 'venv_deadcode\\Lib\\site-packages')

from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute
import pandas as pd

# Simular lo que hace el código
out_file = r'optimization_results\Seq_Iter_01\model_Seq_Iter_01.out'

with Output(out_file) as out:
    tank_id = 'tank_43'
    
    # Esto es lo que extrae rut_27_model_metrics.py
    depth_series = pd.Series(out.node_series(tank_id, NodeAttribute.INVERT_DEPTH))
    flooding_series = pd.Series(out.node_series(tank_id, NodeAttribute.FLOODING_LOSSES))
    flow_series = pd.Series(out.node_series(tank_id, NodeAttribute.TOTAL_INFLOW))
    volume_series = pd.Series(out.node_series(tank_id, NodeAttribute.PONDED_VOLUME))
    
    # Calcular valores
    total_volume = flow_series.sum()  # Integración del caudal
    max_stored_volume = volume_series.max()  # Volumen máximo almacenado
    max_depth = depth_series.max()
    
    print(f"Datos extraídos del SWMM para {tank_id}:")
    print(f"  total_volume (integrado): {total_volume:.2f} m³")
    print(f"  max_stored_volume (máx): {max_stored_volume:.2f} m³")
    print(f"  max_depth: {max_depth:.2f} m")
    
    # El FIX debería usar max_stored_volume
    print(f"\nCon FIX debería ser: {max_stored_volume:.2f} m³")
    print(f"Sin FIX (usando total_volume): {total_volume:.2f} m³")
    
    # Lo que está en el CSV
    csv_value = 51596.12
    print(f"\nValor en CSV: {csv_value:.2f} m³")
    
    # Comparar
    if abs(csv_value - max_stored_volume) < 100:
        print("  => El CSV tiene max_stored_volume (FIX funciona)")
    elif abs(csv_value - total_volume) < 100:
        print("  => El CSV tiene total_volume (FIX NO funciona)")
    else:
        print(f"  => El CSV tiene un valor diferente")
        print(f"     Diferencia con max_stored: {abs(csv_value - max_stored_volume):.2f}")
        print(f"     Diferencia con total: {abs(csv_value - total_volume):.2f}")
