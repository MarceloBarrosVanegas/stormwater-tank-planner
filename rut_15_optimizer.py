"""
SWMM Stormwater Tank Optimizer
==============================

Multi-objective optimizer for stormwater tank placement using NSGA-II.
Finds optimal configurations (node → predio assignments and tank volumes)
to minimize construction costs while maximizing flood reduction.

Dependencies
------------
- pymoo: NSGA-II implementation
- numpy, pandas, geopandas
- rut_00_path_finder: Route optimization
- rut_03_run_sewer_design: Pipeline design
- rut_13_cost_functions: Cost calculations
- rut_14_swmm_modifier: SWMM file modification
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import shutil
import geopandas as gpd
from pathlib import Path
from collections import defaultdict



from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional




import config
config.setup_sys_path()

from rut_02_elevation import ElevationGetter, ElevationSource
from rut_13_cost_functions import CostCalculator
from rut_15_dashboard import EvolutionDashboardGenerator
from rut_17_comparison_reporter import ScenarioComparator
from rut_26_hydrological_impact import HydrologicalImpactAssessment




DYNAMIC_EVALUATOR_AVAILABLE = True



def format_currency(value: float, decimals: int = 2) -> str:
    """Format float as currency with thousands separator."""
    return f"${value:,.{decimals}f}"


class GreedyTankOptimizer:
    """
    Deterministic optimizer that adds tanks one by one based on Step 1 ranking.
    Used for 'Sequence Analysis' to find diminishing returns.
    
    Features:
    - Dynamic tank sizing based on flooding volume
    - Iterative re-ranking after each SWMM simulation
    - Pruning of tanks that become irrelevant after upstream additions
    """
    def __init__(self,
                 dynamic_evaluator: Optional[object] = None,
                 stop_at_breakeven: bool = False,  # Stop when cost >= savings * multiplier
                 breakeven_multiplier: float = 1.0,  # Allow investment up to X times avoided damage
                 flooding_cost_per_m3: float = 1250.0):  # $/m³ flooding damage
        
        self.evaluator = dynamic_evaluator
        self.metrics_extractor = dynamic_evaluator.metrics_extractor
        self.comparator = ScenarioComparator(self.metrics_extractor.metrics, baseline_inp_path=str(config.SWMM_FILE))
        self.stop_at_breakeven = stop_at_breakeven
        self.breakeven_multiplier = breakeven_multiplier
        self.flooding_cost_per_m3 = flooding_cost_per_m3
        
    def compare_solutions(self, active_pairs):
            """
            Orchestrates the full comparison between current solution and baseline.
            Groups data preparation, printing, visual reporting, and hydrological impact assessment.
            """
            # --- 1. DATA PREPARATION (Topology & Geometries) ---
            # derivations_geom = [pair['derivation_link_geometry'] for pair in active_pairs]
            swmm_gdf = self.current_metrics.swmm_gdf
            detailed_links_dict = {}
            for i, pair in enumerate(active_pairs):
                nid = str(pair['node_id'])
                downstream_links = self.evaluator.node_topology[nid].get('downstream', [])
                upstream_links = self.evaluator.node_topology[nid].get('upstream', [])


                try:
                    node_start = swmm_gdf.loc[downstream_links]['InletNode'].item()
                except:
                    try:
                        node_start = swmm_gdf.loc[upstream_links]['OutletNode'].item()
                    except:
                        node_start = f'{i}.0'



                detailed_links_dict[nid] = {
                    'upstream': upstream_links,
                    'downstream': downstream_links,
                    'derivation': [f"{node_start}-{i}.1"]
                }
    
            # --- 2. NODE RESULTS TABLE & PAIR UPDATE ---
            print(f"\n  {'NodeID':<12} {'Flood BASE':<15} {'Flood SOL':<15} {'Tank Vol':<12} {'Tank Depth':<10}")
            print(f"  {'-'*65}")
    
            for i, pair in enumerate(active_pairs):
                nid = str(pair['node_id'])
                tank_id = f"tank_{pair['predio_id']}"
                
                # Fetch current metrics from the simulation results
                base_f = self.baseline_metrics.node_data.get(nid, {}).get('flooding_volume', 0.0)
                curr_f = self.current_metrics.node_data.get(nid, {}).get('flooding_volume', 0.0)
                
                stored_vol, stored_depth = 0.0, 0.0
                if tank_id in self.current_metrics.tank_data:
                    stored_vol = self.current_metrics.tank_data[tank_id]['max_stored_volume']
                    stored_depth = self.current_metrics.tank_data[tank_id]['max_depth']
    
                # Update solution metadata for downstream reporting
                pair['tank_volume_simulation'] = stored_vol
                pair['tank_max_depth'] = stored_depth
                
                print(f"  {nid:<12} {base_f:<15.1f} {curr_f:<15.1f} {stored_vol:<12.1f} {stored_depth:<10.2f}")
    
            # --- 3. GENERATE VISUAL REPORTS (Dashboard & Hydrographs) ---
            print(f"\n  [Report] Generating comparison plots for {self.solution_name}...")
            self.comparator.generate_comparison_plots(
                solution=self.current_metrics,
                solution_name=self.solution_name,
                save_dir=self.case_dir,
                nodes_gdf=self.metrics_extractor.nodes_gdf,
                derivations=self.evaluator.last_designed_gdf,
                detailed_links=detailed_links_dict,
                tank_details=active_pairs,
                show_predios=False
            )
    
            # --- 4. GLOBAL PERFORMANCE SUMMARY (Deltas) ---
            vol_b = self.baseline_metrics.total_flooding_volume
            vol_s = self.current_metrics.total_flooding_volume
            delta_v = vol_b - vol_s
            pct_v = (delta_v / vol_b) * 100
    
            out_v_b = self.baseline_metrics.total_outfall_volume
            out_v_s = self.current_metrics.total_outfall_volume
            out_q_b = self.baseline_metrics.total_max_outfall_flow
            out_q_s = self.current_metrics.total_max_outfall_flow
    
            print(f"\n  [Summary] System Performance: {self.solution_name}")
            print(f"    - Flooding Volume:   {vol_s:,.1f} m3 (Base: {vol_b:,.1f} m3 | Delta: {delta_v:,.0f} m3, {pct_v:.1f}% reduction)")
            print(f"    - Outfall Volume:    {out_v_s:,.1f} m3 (Base: {out_v_b:,.1f} m3 | Delta: {out_v_b - out_v_s:,.0f} m3)")
            print(f"    - Outfall Peak Flow: {out_q_s:,.3f} m3/s (Base: {out_q_b:,.3f} m3/s | Delta: {out_q_b - out_q_s:,.3f} m3/s)")
            print()
    
            # --- 5. HYDROLOGICAL IMPACT ASSESSMENT ---
            print(f"  [Impact] Running Hydrological Impact Assessment for {self.solution_name}...")
            impact_eval = HydrologicalImpactAssessment(str(self.case_dir))
            # impact_eval.output_dir = self.case_dir / "hydrological_impact"
            # impact_eval.output_dir.mkdir(parents=True, exist_ok=True)

            impact_eval.run_assessment(
                baseline_inp=str(config.SWMM_FILE),
                solution_inp=str(self.current_inp_file)
            )

    def _remove_case_directory(self):
        """Remove a case directory if it exists."""
        case_dir = self.evaluator.case_dir
        
        if case_dir and Path(case_dir).exists():
            shutil.rmtree(case_dir)
            print(f"  [Cleanup] Removed directory: {case_dir}")




    def max_ramal_por_tanque(self, cumulative_paths):
        """
        cumulative_paths: dict[node_id] -> GeoDataFrame/DataFrame de 1 fila
        Usa columnas: node_ramal, target_type, target_id, total_flow
    
        Retorna:
          - max_by_tank: {tank_id: ramal_max}
          - ramales_by_tank: {tank_id: [ramales...]}
        """
    
        def dest_ramal_from_target_id(tid):
            # target_id = 1.2 -> 1
            s = str(tid).strip()
            if s.endswith(".0"):
                s = s[:-2]
            return int(s.split(".", 1)[0])
    
        # 1) Resumir a 1 registro por ramal (si hay varios node_id por ramal, toma el de mayor total_flow)
        ramal_info = {}  # ramal -> dict con target_type/target_id/flow
    
        for node_id, gdf in cumulative_paths.items():
            if gdf is None or len(gdf) == 0:
                continue
            row = gdf.iloc[0]
    
            ramal = int(row["node_ramal"])
            flow = float(row.get("total_flow", 0) or 0)
    
            if (ramal not in ramal_info) or (flow > ramal_info[ramal]["flow"]):
                ramal_info[ramal] = {
                    "flow": flow,
                    "target_type": row.get("target_type", None),
                    "target_id": row.get("target_id", None),
                }
    
        # 2) Semilla: ramales que llegan directo a tank
        sink_by_ramal = {}  # ramal -> tank_id
        for ramal, info in ramal_info.items():
            if info["target_type"] == "tank":
                sink_by_ramal[ramal] = info["target_id"]
    
        # 3) Propagar aguas arriba: si un ramal descarga a otro ramal que ya llega a tank, hereda ese tank
        changed = True
        while changed:
            changed = False
            for ramal, info in ramal_info.items():
                if ramal in sink_by_ramal:
                    continue
                if info["target_type"] == "node" and info["target_id"] is not None:
                    try:
                        r_dest = dest_ramal_from_target_id(info["target_id"])
                    except Exception:
                        continue
                    if r_dest in sink_by_ramal:
                        sink_by_ramal[ramal] = sink_by_ramal[r_dest]
                        changed = True
    
        # 4) Agrupar y sacar el ramal más alto por tanque
        ramales_by_tank = defaultdict(list)
        for ramal, tank_id in sink_by_ramal.items():
            ramales_by_tank[tank_id].append(ramal)
    
        max_by_tank = {tank_id: max(ramales) for tank_id, ramales in ramales_by_tank.items()}
    
        return max_by_tank, dict(ramales_by_tank)

    def node_id_a_ramal_max_del_tanque(self, cumulative_paths, return_tank_id=False):
        """
        cumulative_paths: dict[node_id] -> GeoDataFrame/DataFrame (1 fila)
    
        Reglas:
        - Para cada ramal, se toma 1 fila "representativa" (por defecto: la de mayor total_flow)
          para saber a dónde descarga el ramal.
        - Si target_type == 'node' y target_id == '1.2' => descarga al ramal 1 (split('.')[0]).
        - Si target_type == 'tank' y target_id == 55 => ese ramal llega directo al tanque 55.
        - Se propaga aguas arriba (2->1->tank) para saber qué ramales terminan en cada tanque.
    
        Retorna:
          - dict_respuesta:
              si return_tank_id=False: {node_id: ramal_max_del_tanque (o None)}
              si return_tank_id=True:  {node_id: {"tank_id":..., "max_ramal":...}}
          - (extra) max_by_tank, sink_by_ramal por si quieres inspeccionar
        """
    
        def dest_ramal_from_target_id(tid):
            s = str(tid).strip()
            if s.endswith(".0"):
                s = s[:-2]
            return int(s.split(".", 1)[0])
    
        # 1) Construir info por ramal usando 1 fila representativa (mayor total_flow)
        ramal_info = {}  # ramal -> {"flow":..., "target_type":..., "target_id":...}
    
        for node_id, gdf in cumulative_paths.items():
            if gdf is None or len(gdf) == 0:
                continue
            row = gdf.iloc[0]
    
            ramal = int(row["node_ramal"])
            flow = float(row.get("total_flow", 0) or 0)
    
            if (ramal not in ramal_info) or (flow > ramal_info[ramal]["flow"]):
                ramal_info[ramal] = {
                    "flow": flow,
                    "target_type": row.get("target_type", None),
                    "target_id": row.get("target_id", None),
                }
    
        # 2) Semilla: ramales que llegan directo a tank
        sink_by_ramal = {}  # ramal -> tank_id
        for ramal, info in ramal_info.items():
            if info["target_type"] == "tank":
                sink_by_ramal[ramal] = info["target_id"]
    
        # 3) Propagar aguas arriba: si ramal descarga a otro ramal que ya llega a tank => hereda tank
        changed = True
        while changed:
            changed = False
            for ramal, info in ramal_info.items():
                if ramal in sink_by_ramal:
                    continue
                if info["target_type"] == "node" and info["target_id"] is not None:
                    try:
                        r_dest = dest_ramal_from_target_id(info["target_id"])
                    except Exception:
                        continue
                    if r_dest in sink_by_ramal:
                        sink_by_ramal[ramal] = sink_by_ramal[r_dest]
                        changed = True
    
        # 4) Sacar el ramal máximo por tanque
        ramales_by_tank = defaultdict(list)
        for ramal, tank_id in sink_by_ramal.items():
            ramales_by_tank[tank_id].append(ramal)
    
        max_by_tank = {tank_id: max(ramales) for tank_id, ramales in ramales_by_tank.items()}
    
        # 5) Construir respuesta por node_id
        respuesta = {}
        for node_id, gdf in cumulative_paths.items():
            if gdf is None or len(gdf) == 0:
                respuesta[node_id] = None if not return_tank_id else {"tank_id": None, "max_ramal": None}
                continue
    
            row = gdf.iloc[0]
            ramal = int(row["node_ramal"])
    
            tank_id = sink_by_ramal.get(ramal)          # tanque al que llega ese ramal (si existe)
            max_ramal = max_by_tank.get(tank_id) if tank_id is not None else None
    
            if return_tank_id:
                respuesta[node_id] = {"tank_id": tank_id, "max_ramal": max_ramal}
            else:
                respuesta[node_id] = max_ramal
    
        return respuesta, max_by_tank, sink_by_ramal
    

    def run_sequential(self) -> pd.DataFrame:
        """
        Runs the iterative Greedy Sequential selection process.
        
        Logic:
        1. Select the top candidate from the ranked list.
        2. Apply filtering (technical, spatial, and economic constraints).
        3. If valid, 'install' the tank by updating predio area and used nodes.
        4. Run a full SWMM simulation with the new configuration.
        5. Generate comparison reports and update residual flooding.
        """
        # --- 1. INITIALIZATION & STATE SETUP ---
        results = []
        active_candidates = []
        used_nodes = set()
        permanently_excluded = set()
        pruning_failures = {}
        
        # Local metrics references
        self.baseline_metrics = self.metrics_extractor.metrics
        self.predios_gdf = self.metrics_extractor.predios_gdf
        self.nodes_gdf = self.metrics_extractor.nodes_gdf
        candidates = self.metrics_extractor.ranked_candidates.copy()

        economic_history = []  # [(iteration, cost, savings, n_tanks)]
        last_tank_config = None  # Track tank config to skip redundant graphs
        
        print(f"\n{'='*60}")
        print(f"  ITERATIVE GREEDY SEQUENTIAL ANALYSIS")
        print(f"{'='*60}")
        print(f"  Tank Depth: {config.TANK_DEPTH_M}m | Volume: {config.TANK_MIN_VOLUME_M3}-{config.TANK_MAX_VOLUME_M3} m³")
        print(f"  Max Prune Retries: {config.MAX_PRUNE_RETRIES}")

        # --- 2. MAIN ITERATIVE SELECTION LOOP ---
        iteration = 0
        volume_condition = True
        
        while volume_condition:
            if not candidates:
                print("  [Warning] No more candidates available in ranked list.")
                break
                
            cand = candidates.pop(0)  # Take the current best

            # Skip if flooding volume is below threshold
            if cand['total_volume'] < config.TANK_MIN_VOLUME_M3:
                continue
                
            # Skip if node is already occupied
            if cand['node_id'] in used_nodes:
                continue
            
            # Skip if node has been permanently excluded
            if cand['node_id'] in permanently_excluded:
                continue
            
            # Skip if node belongs to a newly designed pipeline (not original node)
            if self.evaluator.last_designed_gdf is not None:
                if cand['node_id'] in self.evaluator.last_designed_gdf['Pozo'].to_list():
                    continue
                    
            # Skip if pruning retry limit reached
            if pruning_failures.get(cand['node_id'], 0) >= config.MAX_PRUNE_RETRIES:
                permanently_excluded.add(cand['node_id'])
                continue
            
            # Proximity check: Avoid clustering tanks too close to each other
            if config.DERIVATION_MIN_DISTANCE_M > 0 and active_candidates:
                too_close = False
                for active in active_candidates:
                    dist = np.sqrt((cand['node_x'] - active['node_x'])**2 + (cand['node_y'] - active['node_y'])**2)
                    if dist < config.DERIVATION_MIN_DISTANCE_M:
                        too_close = True
                        break
                if too_close:
                    print('Aviso Casey: Candidate too close to existing tank, skipping.')
                    continue

            
            iteration += 1
            # --- 2c. TANK INSTALLATION & REGISTRATION ---
            used_nodes.add(cand['node_id'])
            active_candidates.append(cand)
            
            # Step logging
            print(f"\n  STEP {iteration}: Adding Tank @ {cand['node_id']}")
            # print(f"  {'-'*45}")
            # print(f"  Target Predio: (Tank #{n_tanks_in_predio})")
            # print(f"  Flooding Volume: {cand['total_volume']:,.0f} m³")
            # print(f"  Area Required: {required_area:,.0f} m² | Remaining: {predio_capacity[predio_idx]['total_area'] - predio_capacity[predio_idx]['used_area']:,.0f} m²")

            # --- 2d. SIMULATION & EVALUATION ---
            print(f"  Running SWMM simulation for {len(active_candidates)} tanks...")
            self.solution_name = f"Seq_Iter_{iteration:02d}"
            
            cost_link,  current_inp_file, overflow, small_tanks = self.evaluator.evaluate_solution(
                active_candidates,
                solution_name=self.solution_name,
                current_metrics=self.metrics_extractor,
                iteration = iteration
            )
            
            if current_inp_file is None:
                print('No pair was evaluated successfully, continue the optimization.')
                continue
            
            # Update current state references
            self.current_inp_file = current_inp_file
            self.case_dir = self.evaluator.case_dir
            self.current_metrics = self.evaluator.current_metrics
       
            
            # ========================================================================
            # EVALUACIÓN CONSISTENTE:
            # - overflow: list (problemas) o False (OK)
            # - small_tanks: list (problemas) o False (OK)
            # ========================================================================
            
            if overflow or small_tanks:
                node2info, _, _ = self.node_id_a_ramal_max_del_tanque(self.evaluator.cumulative_paths, return_tank_id=True)
                
                if not hasattr(self.evaluator, 'forbidden_pairs'):
                    self.evaluator.forbidden_pairs = set()
                
                overflow_nodes = set()
                small_tank_nodes = set()
                
                # ====================================================================
                # PARTE A: PROCESAR OVERFLOWS (Agregar a forbidden_pairs)
                # ====================================================================
                if overflow:  # overflow es list (nunca True)
                    print(f"\n  [Overflow Summary] Found {len(overflow)} predio capacity overflow(s):")
                    
                    for idx, ovf in enumerate(overflow, 1):
                        node_id = ovf['node_id']
                        target_id = ovf['target_id']
                        
                        print(f"    {idx}. Tank {ovf['tank_id']} at predio {ovf['predio_id']} (node {node_id})")
                        print(f"       → Exceeded physical capacity")
                        
                        # CRÍTICO: Agregar a forbidden_pairs (inviabilidad física)
                        self.evaluator.forbidden_pairs.add((str(node_id), str(target_id)))
                        print(f"       → Added ({node_id}, {target_id}) to FORBIDDEN pairs")
                        
                        overflow_nodes.add(node_id)
                
                # ====================================================================
                # PARTE B: PROCESAR SMALL TANKS (NO agregar a forbidden_pairs)
                # ====================================================================
                if small_tanks:  # small_tanks es list (nunca True)
                    print(f"\n  [Small Tank Summary] Found {len(small_tanks)} undersized tank(s):")
                    
                    for idx, st in enumerate(small_tanks, 1):
                        node_id = st['node_id']
                        
                        print(f"    {idx}. Tank {st['tank_id']} at predio {st['predio_id']} (node {node_id})")
                        print(f"       → Volume: {st['stored_volume']:.2f} m³ < {config.TANK_MIN_VOLUME_M3} m³")
                        
                        # CRÍTICO: NO agregar a forbidden_pairs (podría funcionar con otro candidato)
                        print(f"       → Will retry (NOT adding to forbidden pairs)")
                        
                        small_tank_nodes.add(node_id)
                
                # ====================================================================
                # PARTE C: CONSOLIDAR Y LIMPIAR
                # ====================================================================
                all_nodes = overflow_nodes.union(small_tank_nodes)
                
                print(f"\n  [Cleanup Summary]")
                print(f"    - Overflows (forbidden): {len(overflow_nodes)} node(s)")
                print(f"    - Small tanks (not forbidden): {len(small_tank_nodes)} node(s)")
                print(f"    - Total to clean: {len(all_nodes)} node(s)")
                
                


                # Limpiar paths
                for node_id in all_nodes:
                    if node_id in self.evaluator.cumulative_paths:
                        data_to_erase = node2info[node_id]
                        ramal_to_erase = str(data_to_erase['max_ramal'])
                        node_id_to_erase = [_ for _, __ in self.evaluator.cumulative_paths.items() if ramal_to_erase == __.node_ramal.item()][0]
                        del self.evaluator.cumulative_paths[node_id_to_erase]
                
                # Limpiar candidatos activos
                active_candidates = [
                    p for p in active_candidates
                    if str(p.get('node_id')) not in {str(nid) for nid in [node_id_to_erase]}
                ]
                
                # Limpiar directorio
                self._remove_case_directory()
                
                # Preparar para retry
                iteration = iteration - 1
                candidates = self.metrics_extractor.ranked_candidates.copy()
                
                print(f"\n  [Retry] Continuing with {len(active_candidates)} remaining candidate(s)\n")
                
                # Continuar con siguiente candidato
                continue





            n_tanks = 0
            for pair in active_candidates:
                if pair['is_tank']:
                    n_tanks += 1
                    # Get tank volume from designed data
                    predio_id = pair['predio_id']
                    tank_id = f"tank_{predio_id}"
                    tank_volume = self.current_metrics.tank_data[tank_id]['total_volume']
                    tank_depth = self.current_metrics.tank_data[tank_id]['max_depth']
                    pair['tank_volume_simulation'] = tank_volume
                    pair['tank_max_depth'] = tank_depth
                    pair['cost_tank'] = CostCalculator.calculate_tank_cost(tank_volume)
                    pair['cost_land'] = (tank_volume / config.TANK_DEPTH_M ) * config.LAND_COST_PER_M2
                
            cost_tank_land = 0.0
            for pair in active_candidates:
                if pair['is_tank']:
                    cost_tank_land += pair['cost_tank'] + pair['cost_land']
                
            self.current_metrics.cost = cost_link + cost_tank_land
            
            # Detailed visual and tabular comparison
            self.compare_solutions(active_candidates)

            candidates = self.metrics_extractor.ranked_candidates.copy()

            # Sum component costs
            total_tank_cost = sum(p['cost_tank'] for p in active_candidates if p['is_tank'])
            total_land_cost = sum(p['cost_land'] for p in active_candidates if p['is_tank'])
            total_tank_volume = sum(p['tank_volume_simulation'] for p in active_candidates if p['is_tank'])
            
            # Risk Proxy (Residual Flood Damage)
            residual_damage = self.current_metrics.total_flooding_volume * self.flooding_cost_per_m3
            
            # Calculate MARGINAL values
            if results:
                prev = results[-1]
                marginal_cost = self.current_metrics.cost - prev['cost_total']
                marginal_reduction = prev['flooding_volume'] - self.current_metrics.total_flooding_volume
                marginal_tank_volume = total_tank_volume - prev['total_tank_volume']
            else:
                marginal_cost = self.current_metrics.cost
                marginal_reduction = self.baseline_metrics.total_flooding_volume - self.current_metrics.total_flooding_volume
                marginal_tank_volume = total_tank_volume
            
            # Get current tank's individual data
            current_tank_cost = cand.get('cost_tank', 0) if cand.get('is_tank') else 0
            current_tank_land_cost = cand.get('cost_land', 0) if cand.get('is_tank') else 0
            current_tank_volume = cand.get('tank_volume_simulation', 0) if cand.get('is_tank') else 0

            results.append({
                # --- Identification ---
                'step': iteration,
                'n_tanks': n_tanks,
                'added_node': cand['node_id'],
                'added_predio': cand.get('predio_id', ''),
                
                # --- Cost Breakdown (Cumulative) ---
                'cost_total': self.current_metrics.cost,
                'cost_links': cost_link,
                'cost_tanks': total_tank_cost,
                'cost_land': total_land_cost,
                'cost_replacements': cost_link * 0.20, 
                
                # --- Cost Breakdown (MARGINAL - This Tank Only) ---
                'marginal_cost': marginal_cost,
                'marginal_tank_cost': current_tank_cost,
                'marginal_land_cost': current_tank_land_cost,
                'marginal_tank_volume': marginal_tank_volume,
                
                # --- Hydraulic Performance (Cumulative) ---
                'flooding_volume': self.current_metrics.total_flooding_volume,
                'flooding_reduction': self.baseline_metrics.total_flooding_volume - self.current_metrics.total_flooding_volume,
                'outfall_peak_flow': self.current_metrics.total_max_outfall_flow,
                'flooded_nodes_count': self.current_metrics.flooded_nodes_count,
                
                # --- Hydraulic Performance (MARGINAL - This Tank Only) ---
                'marginal_reduction': marginal_reduction,
                
                # --- Efficiency Metrics ---
                'efficiency_m3_per_dollar': marginal_reduction / marginal_cost if marginal_cost > 0 else 0,
                'cost_per_m3_reduced': marginal_cost / marginal_reduction if marginal_reduction > 0 else 0,
                
                # --- Network Health ---
                'surcharged_links_count': self.current_metrics.surcharged_links_count,
                'overloaded_links_length': self.current_metrics.overloaded_links_length,
                'system_mean_utilization': self.current_metrics.system_mean_utilization,
                
                # --- Infrastructure Specs ---
                'total_tank_volume': total_tank_volume,
                'current_tank_volume': current_tank_volume,
                
                # --- Economic Risk (Proxy) ---
                'residual_damage_usd': residual_damage,
                'total_social_cost': self.current_metrics.cost + residual_damage,
                
                # --- Legacy/Display ---
                'cost_display': format_currency(self.current_metrics.cost),
                'cost': format_currency(self.current_metrics.cost),
                'flooding_remaining': self.current_metrics.total_flooding_volume 
            })



            # === SPATIAL EXPORT (Network Health Snapshot) ===
            # Export the network state (with utilization) for this iteration
            if hasattr(self.current_metrics, 'swmm_gdf') and not self.current_metrics.swmm_gdf.empty:
                self._export_spatial_snapshot(iteration, self.current_metrics.swmm_gdf)
            
            # === BREAK CONDITIONS ===
            # 1. Volume Condition: Stop if flooding is effectively zero
            if self.current_metrics.total_flooding_volume < config.TANK_MIN_VOLUME_M3:
                print(f"  [Stop] Flooding reduced to near zero ({self.current_metrics.total_flooding_volume:.1f} m³).")
                volume_condition = False
                break
                
            # 2. Hard Stop: Max iterations safety
            if iteration >= config.MAX_TANKS:
                print(f"  [Stop] Max sequential iterations ({config.MAX_TANKS}) reached.")
                volume_condition = False
                break

        # === GENERATE VISUAL RESULTS DASHBOARD (New Series 07) ===
        if results:
            df_results = pd.DataFrame(results)
            dash_gen = EvolutionDashboardGenerator(df_results, Path("optimization_results"))
            dash_gen.generate_all()
            
            # Save CSV tracking
            csv_path = Path("optimization_results") / "sequence_tracking.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"  [Tracking] Saved CSV: {csv_path}")


        # # === GENERATE SUMMARY TEXT FILE ===
        # if active_candidates and self.evaluator:
        #     self._generate_summary_report(active_candidates, economic_history, predio_capacity)
        #
        # # === EXPORT SELECTED PREDIOS AS GPKG ===
        # if active_candidates:
        #     self._export_selected_predios_gpkg(active_candidates, predio_capacity)
        
        return pd.DataFrame(results)

    def _export_spatial_snapshot(self, iteration: int, swmm_gdf: gpd.GeoDataFrame):
        """
        Exports the current network state to a layer in a GPKG.
        Includes health metrics calculated in rut_27 (MaxFlow, Capacity, utilization).
        """
        save_dir = Path("optimization_results") / "spatial_history"
        save_dir.mkdir(parents=True, exist_ok=True)
        gpkg_path = save_dir / "network_evolution.gpkg"
        
        # Select relevant columns for visualization
        cols = ['Name', 'geometry', 'MaxFlow', 'Capacity', 'flow_pipe_capacity', 
                'flow_over_pipe_capacity', 'utilization', 'Surcharged']
        
        # Ensure columns exist (utilization might not be in GDF if not calc properly)
        # Recalculate if missed? No, rut_27 should have done it.
        # Just check columns presence
        
        valid_cols = [c for c in cols if c in swmm_gdf.columns or c == 'geometry']
        gdf_export = swmm_gdf[valid_cols].copy()
        
        # Add iteration info
        gdf_export['iteration'] = iteration
        
        # Append to GPKG layer? Or separate layers?
        # Better: Single GPKG, multiple layers "iter_01", "iter_02"...
        layer_name = f"iter_{iteration:02d}"
        
        try:
            gdf_export.to_file(gpkg_path, layer=layer_name, driver="GPKG")
            # print(f"  [Spatial] Saved layer {layer_name} to {gpkg_path}")
        except Exception as e:
            print(f"  [Spatial] Error saving layer {layer_name}: {e}")
    
    def get_available_area_from_predios(self, predio_capacity: dict[Any, Any]):
        # Pre-calculate available area for each predio
        for idx, row in self.predios_gdf.iterrows():
            predio_idx = idx if isinstance(idx, int) else self.predios_gdf.index.get_loc(idx)
            area_m2 = row.geometry.area
            predio_capacity[str(predio_idx)] = {'total_area': area_m2, 'used_area': 0.0, 'n_tanks': 0}
        
        return predio_capacity




    def _generate_summary_report(self, active_candidates: list, economic_history: list, predio_capacity: dict):
        """Generate a text file with optimization summary results."""
        from pathlib import Path
        from datetime import datetime
        
        save_dir = Path("optimization_results")
        save_dir.mkdir(exist_ok=True)
        
        # Get final metrics
        if economic_history:
            final = economic_history[-1]
            total_cost = final['cost']
            total_savings = final['savings']
            n_tanks = final['n_tanks']
            bc_ratio = total_savings / total_cost if total_cost > 0 else 0
            roi = (total_savings - total_cost) / total_cost * 100 if total_cost > 0 else 0
            net_benefit = total_savings - total_cost
        else:
            total_cost = 0
            total_savings = 0
            n_tanks = len(active_candidates)
            bc_ratio = 0
            roi = 0
            net_benefit = 0
        
        # Get tank utilization data
        util_map = {}
        if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:
            util_map = self.evaluator.last_metrics.tank_utilization or {}
        
        # Write summary file
        summary_path = save_dir / "optimization_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("   RESUMEN DE OPTIMIZACIÓN DE TANQUES DE TORMENTA\n")
            f.write("=" * 60 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("RESULTADOS ECONÓMICOS\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Número de Tanques Activos:  {n_tanks}\n")
            f.write(f"  Costo de Construcción:      ${total_cost:,.0f}\n")
            f.write(f"  Ahorro por Flooding Evitado:${total_savings:,.0f}\n")
            f.write(f"  Beneficio Neto:             ${net_benefit:,.0f}\n")
            f.write(f"  Ratio B/C:                  {bc_ratio:.2f}\n")
            f.write(f"  ROI:                        {roi:.1f}%\n\n")
            
            # Baseline info
            if hasattr(self.evaluator, 'baseline_metrics'):
                baseline_flood = self.evaluator.baseline_metrics.total_flooding_volume
                # current_flood = self.evaluator.last_metrics.total_flooding_volume if self.evaluator.last_metrics else baseline_flood
                # Reemplaza la línea problemática por esto:
                last_metrics = getattr(self.evaluator, 'last_metrics', None)
                current_flood = last_metrics.total_flooding_volume if last_metrics else baseline_flood

                reduction = baseline_flood - current_flood
                pct_reduction = (reduction / baseline_flood * 100) if baseline_flood > 0 else 0
                f.write("-" * 60 + "\n")
                f.write("REDUCCIÓN DE INUNDACIÓN\n")
                f.write("-" * 60 + "\n")
                f.write(f"  Flooding Baseline:          {baseline_flood:,.0f} m³\n")
                f.write(f"  Flooding Final:             {current_flood:,.0f} m³\n")
                f.write(f"  Reducción Total:            {reduction:,.0f} m³ ({pct_reduction:.1f}%)\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("DETALLE DE TANQUES\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Tanque':<20} {'Predio':<8} {'Diseño':<12} {'Capturado':<12} {'Util %':<8}\n")
            f.write("-" * 60 + "\n")
            
            for ac in active_candidates:
                tk_name = f"TK_{ac['node_id']}_{ac['predio_id']}"
                predio = ac['predio_id']
                design_vol = ac['total_volume']
                
                if tk_name in util_map:
                    stored = util_map[tk_name].get('stored_volume', 0)
                    util_pct = (stored / design_vol * 100) if design_vol > 0 else 0
                else:
                    stored = 0
                    util_pct = 0
                
                f.write(f"{tk_name:<20} {predio:<8} {design_vol:>10,.0f} m³ {stored:>10,.0f} m³ {util_pct:>6.1f}%\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Fin del Reporte\n")
        
        print(f"  [Summary] Saved: {summary_path}")
    
    def _export_selected_predios_gpkg(self, active_candidates: list, predio_capacity: dict):
        """Export selected predios with tank data as GeoPackage."""


        
        save_dir = Path("optimization_results")
        save_dir.mkdir(exist_ok=True)
        
        # Get unique predios used
        used_predios = set()
        predio_tanks = {}  # {predio_id: [tank_data]}
        
        for ac in active_candidates:
            pid = ac['predio_id']
            used_predios.add(pid)
            if pid not in predio_tanks:
                predio_tanks[pid] = []
            predio_tanks[pid].append(ac)
        
        # Filter predios GDF to only selected ones
        if hasattr(self, 'predios_gdf') and self.predios_gdf is not None:
            selected = self.predios_gdf[self.predios_gdf.index.isin(used_predios)].copy()
        else:
            print("  [GPKG] No predios_gdf available for export")
            return
        
        # Add tank data columns
        selected['n_tanks'] = selected.index.map(lambda x: len(predio_tanks.get(x, [])))
        selected['total_design_vol'] = selected.index.map(
            lambda x: sum(t['total_volume'] for t in predio_tanks.get(x, []))
        )
        
        # Add utilization data if available
        util_map = {}
        if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:
            util_map = self.evaluator.last_metrics.tank_utilization or {}
        
        def get_stored_vol(pid):
            tanks = predio_tanks.get(pid, [])
            total = 0
            for t in tanks:
                tk_name = f"TK_{t['node_id']}_{t['predio_id']}"
                if tk_name in util_map:
                    total += util_map[tk_name].get('stored_volume', 0)
            return total
        
        selected['total_stored_vol'] = selected.index.map(get_stored_vol)
        selected['utilization_pct'] = (selected['total_stored_vol'] / selected['total_design_vol'] * 100).fillna(0)
        
        # Add tank names
        selected['tank_names'] = selected.index.map(
            lambda x: ', '.join([f"TK_{t['node_id']}_{t['predio_id']}" for t in predio_tanks.get(x, [])])
        )
        
        # Export
        gpkg_path = save_dir / "selected_predios.gpkg"
        selected.to_file(gpkg_path, driver="GPKG")
        print(f"  [GPKG] Saved: {gpkg_path} ({len(selected)} predios)")

    def _generate_results_dashboard(self, active_candidates: list, economic_history: list, predio_capacity: dict):
        """Generate a visual dashboard summarizing all optimization results."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        from pathlib import Path
        from datetime import datetime
        
        save_dir = Path("optimization_results")
        save_dir.mkdir(exist_ok=True)
        
        # Get final metrics
        final = economic_history[-1] if economic_history else {}
        total_cost = final.get('cost', 0)
        total_savings = final.get('savings', 0)
        n_tanks = final.get('n_tanks', len(active_candidates))
        flooding_reduced = final.get('flooding_reduced', 0)
        flooding_remaining = final.get('flooding_remaining', 0)
        baseline_flooding = final.get('baseline_flooding', 0)
        
        bc_ratio = total_savings / total_cost if total_cost > 0 else 0
        roi = (total_savings - total_cost) / total_cost * 100 if total_cost > 0 else 0
        net_benefit = total_savings - total_cost
        pct_controlled = (flooding_reduced / baseline_flooding * 100) if baseline_flooding > 0 else 0
        residual_cost = flooding_remaining * self.flooding_cost_per_m3
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#f5f5f5')
        
        # Title
        fig.suptitle('RESUMEN DE OPTIMIZACIÓN DE TANQUES DE TORMENTA', 
                     fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
        fig.text(0.5, 0.94, f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                 ha='center', fontsize=10, color='gray')
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3, 
                              left=0.05, right=0.95, top=0.90, bottom=0.05)
        
        # === BOX 1: ECONOMIC SUMMARY ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        ax1.set_title('💰 RESULTADOS ECONÓMICOS', fontsize=14, fontweight='bold', pad=10)
        economic_text = (
            f"Costo Construcción:    ${total_cost/1e6:,.2f}M\n\n"
            f"Ahorro por Flooding:   ${total_savings/1e6:,.2f}M\n\n"
            f"Beneficio Neto:        ${net_benefit/1e6:,.2f}M\n\n"
            f"Ratio B/C:             {bc_ratio:.2f}\n\n"
            f"ROI:                   {roi:.1f}%"
        )
        ax1.text(0.1, 0.8, economic_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#4caf50'))
        
        # === BOX 2: FLOODING CONTROL ===
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        ax2.set_title('🌊 CONTROL DE INUNDACIÓN', fontsize=14, fontweight='bold', pad=10)
        flood_text = (
            f"Flooding Baseline:     {baseline_flooding:,.0f} m³\n\n"
            f"Flooding Reducido:     {flooding_reduced:,.0f} m³\n\n"
            f"Flooding Restante:     {flooding_remaining:,.0f} m³\n\n"
            f"% Controlado:          {pct_controlled:.1f}%"
        )
        ax2.text(0.1, 0.8, flood_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e3f2fd', edgecolor='#2196f3'))
        
        # === BOX 3: RISK ASSESSMENT ===
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        ax3.set_title('⚠️ RIESGO RESIDUAL', fontsize=14, fontweight='bold', pad=10)
        risk_text = (
            f"Volumen Sin Control:   {flooding_remaining:,.0f} m³\n\n"
            f"Costo Potencial:       ${residual_cost/1e6:,.2f}M\n\n"
            f"% Sin Controlar:       {100-pct_controlled:.1f}%"
        )
        ax3.text(0.1, 0.8, risk_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee', edgecolor='#f44336'))
        
        # === BOX 4: INFRASTRUCTURE ===
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        ax4.set_title('🏗️ INFRAESTRUCTURA', fontsize=14, fontweight='bold', pad=10)
        total_design_vol = sum(ac['total_volume'] for ac in active_candidates)
        n_predios = len(set(ac['predio_id'] for ac in active_candidates))
        infra_text = (
            f"Tanques Instalados:    {n_tanks}\n\n"
            f"Predios Utilizados:    {n_predios}\n\n"
            f"Volumen Total Diseño:  {total_design_vol:,.0f} m³\n\n"
            f"Iteraciones:           {len(economic_history)}"
        )
        ax4.text(0.1, 0.8, infra_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#ff9800'))
        
        # === BOX 5: TANK TABLE ===
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        ax5.set_title('📋 DETALLE DE TANQUES', fontsize=14, fontweight='bold', pad=10)
        
        # Get utilization data
        util_map = {}
        if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:
            util_map = self.evaluator.last_metrics.tank_utilization or {}
        
        # Create table data
        table_data = [['Tanque', 'Predio', 'Vol Diseño', 'Vol Capturado', 'Util %']]
        for ac in active_candidates[:10]:  # Limit to 10 rows
            tk_name = f"TK_{ac['node_id']}_{ac['predio_id']}"
            design_vol = ac['total_volume']
            stored = util_map.get(tk_name, {}).get('stored_volume', 0)
            util_pct = (stored / design_vol * 100) if design_vol > 0 else 0
            table_data.append([
                tk_name[:18], 
                str(ac['predio_id']),
                f"{design_vol:,.0f}", 
                f"{stored:,.0f}",
                f"{util_pct:.0f}%"
            ])
        
        if len(active_candidates) > 10:
            table_data.append(['...', f'+{len(active_candidates)-10} más', '', '', ''])
        
        table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                          colWidths=[0.25, 0.1, 0.2, 0.2, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # === BOX 6: EFFICIENCY GAUGE ===
        ax6 = fig.add_subplot(gs[2, 0], projection='polar')
        ax6.set_title('📊 EFICIENCIA GENERAL', fontsize=14, fontweight='bold', y=1.1)
        
        # Simple gauge for percentage controlled
        theta = np.linspace(0, np.pi, 100)
        r = np.ones(100)
        ax6.plot(theta, r, 'lightgray', linewidth=20)
        
        controlled_theta = np.linspace(0, np.pi * pct_controlled / 100, 50)
        ax6.plot(controlled_theta, np.ones(50), '#4caf50', linewidth=20)
        
        ax6.set_ylim(0, 1.5)
        ax6.set_yticklabels([])
        ax6.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax6.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax6.text(np.pi/2, 0.5, f'{pct_controlled:.0f}%', ha='center', fontsize=16, fontweight='bold')
        
        # === BOX 7: KEY METRICS ===
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('off')
        ax7.set_title('MÉTRICAS CLAVE', fontsize=14, fontweight='bold', pad=10)
        
        # Big number displays
        metrics = [
            (f"${net_benefit/1e6:.1f}M", "Beneficio Neto", '#27ae60' if net_benefit > 0 else '#e74c3c'),
            (f"{bc_ratio:.1f}x", "Ratio B/C", '#3498db'),
            (f"{pct_controlled:.0f}%", "Controlado", '#9b59b6'),
        ]
        
        for i, (value, label, color) in enumerate(metrics):
            x = 0.15 + i * 0.3
            ax7.text(x, 0.6, value, transform=ax7.transAxes, fontsize=28, fontweight='bold',
                    ha='center', color=color)
            ax7.text(x, 0.3, label, transform=ax7.transAxes, fontsize=12,
                    ha='center', color='gray')
        
        # Save
        dashboard_path = save_dir / "optimization_dashboard.png"
        fig.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='#f5f5f5')
        plt.close(fig)
        print(f"  [Dashboard] Saved: {dashboard_path}")

