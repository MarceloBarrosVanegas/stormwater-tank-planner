import sys
sys.path.insert(0, 'venv_deadcode\\Lib\\site-packages')

from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute

out_file = r'optimization_results\Seq_Iter_26\model_Seq_Iter_26.out'

print("=" * 80)
print("DEBUG - Datos reales del archivo SWMM .out")
print("=" * 80)

with Output(out_file) as out:
    # Revisar tank_43 específicamente
    tank_id = 'tank_43'
    
    print(f"\n{tank_id}:")
    
    # TOTAL_INFLOW (lo que estaba usando antes)
    inflow_series = out.node_series(tank_id, NodeAttribute.TOTAL_INFLOW)
    total_inflow = sum(inflow_series.values()) if inflow_series else 0
    print(f"  TOTAL_INFLOW (suma): {total_inflow:.2f} m³")
    
    # PONDED_VOLUME (lo que debería usar)
    ponded_series = out.node_series(tank_id, NodeAttribute.PONDED_VOLUME)
    max_ponded = max(ponded_series.values()) if ponded_series else 0
    print(f"  PONDED_VOLUME (max): {max_ponded:.2f} m³")
    
    # Profundidad
    depth_series = out.node_series(tank_id, NodeAttribute.INVERT_DEPTH)
    max_depth = max(depth_series.values()) if depth_series else 0
    print(f"  Max depth: {max_depth:.2f} m")
    
    print(f"\n  Diferencia: {abs(total_inflow - max_ponded):.2f} m³")
    print(f"  Ratio: {total_inflow/max_ponded if max_ponded > 0 else 0:.2f}")
    
    # Revisar todos los tanques
    print("\n" + "=" * 80)
    print("Todos los tanques:")
    print("=" * 80)
    
    tank_nodes = [n for n in out.nodes if 'tank_' in n]
    for tank in tank_nodes:
        inflow = out.node_series(tank, NodeAttribute.TOTAL_INFLOW)
        ponded = out.node_series(tank, NodeAttribute.PONDED_VOLUME)
        
        total_in = sum(inflow.values()) if inflow else 0
        max_pond = max(ponded.values()) if ponded else 0
        
        print(f"{tank:10} | Inflow: {total_in:10.2f} | Ponded: {max_pond:10.2f} | Diff: {abs(total_in-max_pond):10.2f}")
