import sys
sys.path.insert(0, 'venv_deadcode\\Lib\\site-packages')

import pandas as pd
from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute

out_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.out'

print("=" * 80)
print("DEBUG - Extracción de métricas")
print("=" * 80)

with Output(out_file) as out:
    tank_id = 'tank_43'
    
    print(f"\nExtrayendo datos para {tank_id}:")
    
    # Esto es lo que hace rut_27_model_metrics.py
    flow_series = pd.Series(out.node_series(tank_id, NodeAttribute.TOTAL_INFLOW))
    volume_series = pd.Series(out.node_series(tank_id, NodeAttribute.PONDED_VOLUME))
    
    print(f"  flow_series.max(): {flow_series.max():.2f}")
    print(f"  flow_series.sum(): {flow_series.sum():.2f}")
    print(f"  volume_series.max(): {volume_series.max():.2f}")
    
    print(f"\n  => max_stored_volume debería ser: {volume_series.max():.2f}")
    print(f"  => total_volume sería: {flow_series.sum():.2f}")
