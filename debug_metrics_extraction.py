import sys
sys.path.insert(0, 'venv_deadcode\\Lib\\site-packages')

import pandas as pd
from rut_27_model_metrics import MetricsExtractor, SystemMetrics
from swmm.toolkit.shared_enum import NodeAttribute
from pyswmm import Output
import config

print("=" * 80)
print("DEBUG - Extracción de métricas paso a paso")
print("=" * 80)

# Simular la extracción para un tanque específico
out_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.out'

with Output(out_file) as out:
    tank_id = 'tank_43'
    
    print(f"\nExtrayendo datos para {tank_id}:")
    
    # Esto es lo que hace rut_27_model_metrics.py
    depth_series = pd.Series(out.node_series(tank_id, NodeAttribute.INVERT_DEPTH))
    flooding_series = pd.Series(out.node_series(tank_id, NodeAttribute.FLOODING_LOSSES))
    flow_series = pd.Series(out.node_series(tank_id, NodeAttribute.TOTAL_INFLOW))
    volume_series = pd.Series(out.node_series(tank_id, NodeAttribute.PONDED_VOLUME))
    
    print(f"  depth_series.max(): {depth_series.max():.2f}")
    print(f"  flow_series.max(): {flow_series.max():.2f}")
    print(f"  volume_series.max(): {volume_series.max():.2f}")
    
    # Calcular total_volume (integrado)
    total_volume = flow_series.sum()
    print(f"  flow_series.sum() (total_volume): {total_volume:.2f}")
    
    print(f"\n  Comparación:")
    print(f"    max_stored_volume (volume_series.max()): {volume_series.max():.2f}")
    print(f"    total_volume (flow_series.sum()): {total_volume:.2f}")
    print(f"    Diferencia: {abs(volume_series.max() - total_volume):.2f}")
