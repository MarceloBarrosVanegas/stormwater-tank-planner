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

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import geopandas as gpd
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Point


import config
config.setup_sys_path()

from rut_02_elevation import ElevationGetter, ElevationSource
from rut_02_get_flodded_nodes import WeightedCandidateSelector
from rut_13_cost_functions import CostCalculator
from rut_14_swmm_modifier import SWMMModifier
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

    def resize_tanks_based_on_exceedance(self, current_inp_file: str, active_pairs: list = None) -> dict:
        """
        Resizes tanks based on exceedance volume, updates the SWMM file, and re-extracts metrics.
        Iterates until no tank has flooding (exceedance_volume = 0).
        
        Returns:
            dict or None: If capacity exceeded, returns {'node_id': X, 'predio_id': Y, 'target_id': Z}
                          Otherwise returns None.
        """
        MAX_RESIZE_ITERATIONS = 10  # Safety limit to prevent infinite loops
        iteration = 0
        
        while iteration < MAX_RESIZE_ITERATIONS:
            iteration += 1
            run = False
            
            tanks = self.current_metrics.tank_data
            for tk_id, tk_data in tanks.items():
                exceedance_volume = tk_data['exceedance_volume']
                if exceedance_volume > 0:
                    new_vol = tk_data['max_stored_volume'] + exceedance_volume
                    new_area = (new_vol / config.TANK_DEPTH_M) * config.TANK_VOLUME_SAFETY_FACTOR + config.TANK_OCCUPATION_FACTOR
                    
                    # Validate predio capacity before resize
                    predio_id = int(tk_id.replace("tank_", ""))  # Extract predio_id from tank_X
                    if predio_id in self.evaluator.predio_tracking:
                        tracking = self.evaluator.predio_tracking[predio_id]
                        if new_area > tracking['area_total']:
                            # Capacity exceeded - find the node_id for this predio
                            node_id = None
                            target_id = predio_id
                            if active_pairs:
                                for pair in active_pairs:
                                    if pair.get('predio_id') == predio_id:
                                        node_id = pair.get('node_id')
                                        # Check if connected to node instead of direct predio
                                        if pair.get('target_type') == 'node':
                                            target_id = pair.get('predio_id')  # The node it connected to
                                        break
                            
                            print(f"  [Overflow] {tk_id} needs {new_area:.0f} m², "
                                  f"predio {predio_id} only has {tracking['area_total']:.0f} m²")
                            
                            return {
                                'node_id': node_id,
                                'predio_id': predio_id,
                                'target_id': target_id,
                                'tank_id': tk_id
                            }
                        # Update tracked volume
                        self.evaluator.predio_tracking[predio_id]['volumen_acumulado'] = new_vol
                    
                    modifier = SWMMModifier(current_inp_file)
                    modifier.modify_storage_area(tk_id, new_area)
                    modifier.save(current_inp_file)
                    print(f"  [Resize Iter {iteration}] {tk_id}: Increased volume by {exceedance_volume:.1f} m³ to {new_vol:.1f} m³")
                    run = True
                
                depth_target = config.TANK_DEPTH_M - (config.TANK_VOLUME_SAFETY_FACTOR - 1) * config.TANK_DEPTH_M
                variacion_permitida = depth_target * 0.05
                max_depth = tk_data['max_depth']
                
                if max_depth < depth_target - variacion_permitida:
                    # Reduce tank size
                    new_area = (tk_data['max_stored_volume'] / config.TANK_DEPTH_M) * config.TANK_VOLUME_SAFETY_FACTOR + config.TANK_OCCUPATION_FACTOR
                    modifier = SWMMModifier(current_inp_file)
                    modifier.modify_storage_area(tk_id, new_area)
                    modifier.save(current_inp_file)
                    print(f"  [Resize Iter {iteration}] {tk_id}: Reduced tank size to control max depth {max_depth:.2f} m")
                    run = True

            # If no changes were made, exit loop
            if not run:
                if iteration > 1:
                    print(f"  [Resize] Converged after {iteration - 1} iteration(s) - no more exceedance")
                break
            
            # Re-run SWMM simulation with updated tank sizes
            inp_dir = Path(current_inp_file).parent
            for out_file in inp_dir.glob('*.out'):
                try:
                    os.remove(out_file)
                except OSError:
                    pass

            self.metrics_extractor.run(current_inp_file)
            self.current_metrics = self.metrics_extractor.metrics
            
            # Check if all exceedance volumes are now zero
            total_exceedance = sum(tk['exceedance_volume'] for tk in self.current_metrics.tank_data.values())
            if total_exceedance <= 0:
                print(f"  [Resize] All tanks converged after {iteration} iteration(s) - no flooding")
                break
        
        if iteration >= MAX_RESIZE_ITERATIONS:
            print(f"  [Warning] Max resize iterations ({MAX_RESIZE_ITERATIONS}) reached - some tanks may still have exceedance")
        
        return None




    def _remove_case_directory(self):
        """Remove a case directory if it exists."""
        case_dir = self.evaluator.current_case_dir
        
        if case_dir and Path(case_dir).exists():
            shutil.rmtree(case_dir)
            print(f"  [Cleanup] Removed directory: {case_dir}")

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

            
            # --- 2a. TECHNICAL CONSTRAINTS & PRUNING ---
            
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
            if hasattr(self.evaluator, 'last_designed_gdf') and self.evaluator.last_designed_gdf is not None:
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
                    continue

            
            iteration += 1
            # --- 2c. TANK INSTALLATION & REGISTRATION ---
            used_nodes.add(cand['node_id'])
            # predio_idx = cand['predio_id']
            # predio_capacity[predio_idx]['used_area'] += required_area
            # predio_capacity[predio_idx]['n_tanks'] += 1
            # n_tanks_in_predio = predio_capacity[predio_idx]['n_tanks']
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
            
            cost_link, current_inp_file = self.evaluator.evaluate_solution(
                active_candidates,
                solution_name=self.solution_name,
                current_metrics=self.metrics_extractor,
            )
            
            if current_inp_file is None:
                print('No pair was evaluated successfully, continue the optimization.')
                continue
            
            # Update current state references
            self.current_inp_file = current_inp_file
            self.case_dir = self.evaluator.current_case_dir
            
            # --- 2e. POST-SIMULATION REPORTING & METRICS ---
            # Extraction and Comparison
            self.metrics_extractor.run(current_inp_file)
            self.current_metrics = self.metrics_extractor.metrics

            #resize tanks based on exceedance
            overflow = self.resize_tanks_based_on_exceedance(current_inp_file, active_candidates)
            
            if overflow:
                # Capacity exceeded - tank cannot fit in the assigned predio
                # Extract IDs for tracking and removal
                node_id = overflow['node_id']
                target_id = overflow['target_id']
                
                # Add to forbidden pairs to prevent reassignment in future iterations
                if not hasattr(self.evaluator, 'forbidden_pairs'):
                    self.evaluator.forbidden_pairs = set()
                self.evaluator.forbidden_pairs.add((str(node_id), str(target_id)))
                
                # Remove the failed path from cumulative tracking
                del self.evaluator.cumulative_paths[node_id]
                
                # Remove the node from active candidates to avoid reprocessing
                active_candidates = [
                    p for p in active_candidates
                    if str(p.get('node_id')) != str(node_id)
                ]
                
                print(f"  [Forbidden] Added ({node_id}, {target_id}) to forbidden pairs. Will retry.")
                
                # Clean up the failed case directory
                self._remove_case_directory()
                
                # Decrement iteration counter to retry with remaining candidates
                iteration = iteration - 1
                candidates = self.metrics_extractor.ranked_candidates.copy()
                
                # Skip graph generation and restart with next candidate
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

            results.append({  'cost': self.current_metrics.cost,
                'step': iteration,
                'n_tanks': n_tanks,
                'cost': format_currency(self.current_metrics.cost),
                'flooding_remaining': self.current_metrics.total_flooding_volume,
                'added_node': cand['node_id']
            })
            
            print(results[-1])
            
            #
            # # === BACKUP STATE FOR POTENTIAL ROLLBACK ===
            # if hasattr(self.evaluator, 'last_designed_gdf') and self.evaluator.last_designed_gdf is not None:
            #     self.evaluator.last_designed_gdf_backup = self.evaluator.last_designed_gdf.copy()
            #
            #
            # # === ECONOMIC TRACKING (for graphing) ===
            # # Get CLIMADA damage from evaluator
            # current_damage = getattr(self.evaluator, 'last_flood_damage_usd', 0)
            # baseline_damage = getattr(self.evaluator, 'baseline_flood_damage', 0)
            #
            # # Get Infrastructure Benefit (Deferred Investment savings)
            # infrastructure_benefit = getattr(self.evaluator, 'last_economic_result', {}).get('infrastructure_benefit', 0)
            #
            # # Total Avoided Cost = CLIMADA savings + Infrastructure Benefit
            # climada_savings = baseline_damage - current_damage
            # cost_saved = climada_savings + infrastructure_benefit
            #
            # # Get flooding volume from metrics for display
            # if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:
            #     flooding_vol_display = self.evaluator.last_metrics.total_flooding_volume
            # else:
            #     flooding_vol_display = flooding_vol
            # baseline_flood = self.evaluator.baseline_metrics.total_flooding_volume
            # flooding_reduced = baseline_flood - flooding_vol_display
    
            # print(f"  [Economics] Baseline CLIMADA Damage: ${baseline_damage:,.2f}")
            # print(f"  [Economics] Current CLIMADA Damage:  ${current_damage:,.2f}")
            # print(f"  [Economics] Avoided Cost:            ${cost_saved:,.2f}")
        #
        # # Track for plotting
        # economic_history.append({
        #     'iteration': iteration,
        #     'cost': cost,
        #     'savings': cost_saved,
        #     'n_tanks': len(active_candidates),
        #     'flooding_reduced': flooding_reduced,
        #     'flooding_remaining': flooding_vol_display,
        #     'baseline_flooding': baseline_flood
        # })
        #
        # # Calculate benefit-to-cost ratio
        # if cost > 0:
        #     benefit_ratio = cost_saved / cost
        # else:
        #     benefit_ratio = float('inf')
        #
        # print(f"  [Economic] Cost: ${cost:,.0f} | Savings: ${cost_saved:,.0f} | B/C Ratio: {benefit_ratio:.2f}")
        #
        # # === GENERATE ECONOMIC GRAPH PER ITERATION (only if config changed) ===
        # current_config = tuple(sorted([c['node_id'] for c in active_candidates]))
        # if economic_history and current_config != last_tank_config:
        #     self._plot_economic_curve(economic_history, iteration)
        #     last_tank_config = current_config
        # elif current_config == last_tank_config:
        #     print(f"  [Graph] Skipped (config unchanged)")
        #
        # # === BREAKEVEN CHECK ===
        # # Stop when cost >= savings * multiplier (e.g., multiplier=1.5 allows 50% more investment)
        # threshold = cost_saved * self.breakeven_multiplier
        # if self.stop_at_breakeven and cost >= threshold:
        #     print(f"\n  *** BREAKEVEN REACHED ***")
        #     print(f"  Construction cost (${cost:,.0f}) >= Threshold (${threshold:,.0f})")
        #     print(f"  [Threshold = Savings × {self.breakeven_multiplier:.2f}]")
        #     print(f"  Stopping optimization at {len(active_candidates)} tanks.")
        #     break
        #
        #
  
        #
        #
        #     # === PRUNING: REMOVE TANKS BASED ON UTILIZATION AND SYSTEMIC IMPACT ===
        #     # Only prune if: (1) stored_volume < MIN, AND (2) flood reduction is minimal
        #     tanks_to_remove = []
        #
        #
        #     if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics and self.evaluator.last_metrics.tank_utilization:
        #         util_map = self.evaluator.last_metrics.tank_utilization
        #
        #         for ac in active_candidates:
        #             # Tank name format: tank_{predio} (matches rut_16)
        #             tk_name = f"tank_{ac['predio_id']}"
        #
        #             if tk_name in util_map:
        #                 max_depth = util_map[tk_name]['max_depth']
        #                 # Calculate stored volume approx: (Vol / Depth) * UsedDepth
        #                 used_vol = (ac['total_volume'] / TANK_DEPTH) * max_depth
        #                 utilization_pct = (used_vol / ac['total_volume']) * 100 if ac['total_volume'] > 0 else 0
        #
        #                 # New logic: Only prune if utilization is very low (< 20%)
        #                 # Tanks with low stored_volume but decent utilization may have systemic impact
        #                 if used_vol < MIN_TANK_VOLUME and utilization_pct < config.TANK_MIN_UTILIZATION_PCT:
        #                     tanks_to_remove.append(ac)
        #                     # Increment failure count
        #                     node_id = ac['node_id']
        #                     pruning_failures[node_id] = pruning_failures.get(node_id, 0) + 1
        #                     retries_left = config.MAX_PRUNE_RETRIES - pruning_failures[node_id]
        #                     print(f"  [Prune] {tk_name}: Stored {used_vol:.0f} m³ ({utilization_pct:.1f}%) - LOW IMPACT (Retries left: {retries_left})")
        #                 elif used_vol < MIN_TANK_VOLUME:
        #                     # Low stored but good utilization % - keep it (systemic impact likely)
        #                     print(f"  [Keep] {tk_name}: Stored {used_vol:.0f} m³ but {utilization_pct:.1f}% utilization - possible systemic impact")
        #
        #
        #     for ac in tanks_to_remove:
        #         node_id = ac['node_id']
        #         predio_idx = ac['predio_id']
        #         # Restore area to predio
        #         tank_area = (ac['total_volume'] / TANK_DEPTH) * OCCUPATION_FACTOR
        #         predio_capacity[predio_idx]['used_area'] -= tank_area
        #         predio_capacity[predio_idx]['n_tanks'] -= 1
        #         # Remove from used_nodes so it could be reconsidered later
        #         used_nodes.discard(node_id)
        #         # Remove from active
        #         active_candidates.remove(ac)








        # # === GENERATE ECONOMIC GRAPH ===
        # if economic_history:
        #     self._plot_economic_curve(economic_history)
        #
        # # === GENERATE SUMMARY TEXT FILE ===
        # if active_candidates and self.evaluator:
        #     self._generate_summary_report(active_candidates, economic_history, predio_capacity)
        #
        # # === EXPORT SELECTED PREDIOS AS GPKG ===
        # if active_candidates:
        #     self._export_selected_predios_gpkg(active_candidates, predio_capacity)
        #
        # # === GENERATE VISUAL RESULTS DASHBOARD ===
        # if active_candidates and economic_history:
        #     self._generate_results_dashboard(active_candidates, economic_history, predio_capacity)
        #
        return pd.DataFrame(results)
    
    def get_available_area_from_predios(self, predio_capacity: dict[Any, Any]):
        # Pre-calculate available area for each predio
        for idx, row in self.predios_gdf.iterrows():
            predio_idx = idx if isinstance(idx, int) else self.predios_gdf.index.get_loc(idx)
            area_m2 = row.geometry.area
            predio_capacity[str(predio_idx)] = {'total_area': area_m2, 'used_area': 0.0, 'n_tanks': 0}
        
        return predio_capacity
        
        
    
    def _plot_economic_curve(self, economic_history: list, current_iteration: int = None):
        """Generate and save the Cost vs Savings economic curve."""
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        iterations = [e['iteration'] for e in economic_history]
        costs = [e['cost'] / 1e6 for e in economic_history]  # Convert to millions
        savings = [e['savings'] / 1e6 for e in economic_history]
        n_tanks = [e['n_tanks'] for e in economic_history]
        
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot Cost (red)
        color_cost = '#e74c3c'
        ax1.set_xlabel('Iteración', fontsize=12)
        ax1.set_ylabel('Valor ($M)', fontsize=12)
        line1 = ax1.plot(iterations, costs, color=color_cost, marker='o', linewidth=2, 
                         label='Costo Construcción', markersize=8)
        
        # Plot Savings (green) on same axis
        color_savings = '#27ae60'
        line2 = ax1.plot(iterations, savings, color=color_savings, marker='s', linewidth=2,
                         label='Ahorro por Flooding Evitado', markersize=8)
        
        ax1.set_ylim(bottom=0)
        
        # Fill area between curves
        ax1.fill_between(iterations, costs, savings, 
                         where=[s > c for s, c in zip(savings, costs)],
                         alpha=0.2, color=color_savings, label='Beneficio Neto (+)')
        ax1.fill_between(iterations, costs, savings,
                         where=[s <= c for s, c in zip(savings, costs)],
                         alpha=0.2, color=color_cost, label='Pérdida Neta (-)')
        
        # Secondary axis for number of tanks
        ax2 = ax1.twinx()
        color_tanks = '#3498db'
        ax2.set_ylabel('Número de Tanques', color=color_tanks, fontsize=12)
        line3 = ax2.bar(iterations, n_tanks, alpha=0.3, color=color_tanks, label='Tanques', width=0.4)
        ax2.tick_params(axis='y', labelcolor=color_tanks)
        ax2.set_ylim(bottom=0)
        # Force integer ticks for tank count
        from matplotlib.ticker import MaxNLocator
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        
        # === STATISTICS BOX ===
        last_entry = economic_history[-1]
        last_cost = last_entry['cost']
        last_savings = last_entry['savings']
        last_n_tanks = last_entry['n_tanks']
        flooding_reduced = last_entry.get('flooding_reduced', 0)
        
        bc_ratio = last_savings / last_cost if last_cost > 0 else float('inf')
        net_benefit = last_savings - last_cost
        roi_pct = ((last_savings - last_cost) / last_cost * 100) if last_cost > 0 else 0
        
        # Residual risk calculation
        flooding_remaining = last_entry.get('flooding_remaining', 0)
        baseline_flooding = last_entry.get('baseline_flooding', 0)
        residual_cost = flooding_remaining * self.flooding_cost_per_m3 if hasattr(self, 'flooding_cost_per_m3') else flooding_remaining * 1250
        pct_controlled = ((baseline_flooding - flooding_remaining) / baseline_flooding * 100) if baseline_flooding > 0 else 0
        
        stats_text = (
            f"═══ ESTADÍSTICAS ITER {current_iteration or len(iterations)} ═══\n"
            f"Tanques Activos: {last_n_tanks}\n"
            f"─────────────────────\n"
            f"Costo Construcción: ${last_cost/1e6:.2f}M\n"
            f"Ahorro Flooding:    ${last_savings/1e6:.2f}M\n"
            f"─────────────────────\n"
            f"Beneficio Neto:     ${net_benefit/1e6:.2f}M\n"
            f"Ratio B/C:          {bc_ratio:.2f}\n"
            f"ROI:                {roi_pct:.1f}%\n"
            f"─────────────────────\n"
            f"Flooding Reducido:  {flooding_reduced:,.0f} m³\n"
            f"═══ RIESGO RESIDUAL ═══\n"
            f"Flooding Restante:  {flooding_remaining:,.0f} m³\n"
            f"% Controlado:       {pct_controlled:.1f}%"
        )
        
        # Position stats box on right side
        props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.95, edgecolor='orange')
        ax1.text(0.98, 0.55, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='right',
                bbox=props, fontfamily='monospace')
        
        # Title and grid
        ax1.set_title('Análisis Económico: Costo vs Ahorro por Iteración', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Find and mark breakeven point if exists
        breakeven_found = False
        for i, (c, s) in enumerate(zip(costs, savings)):
            if c >= s:
                ax1.axvline(x=iterations[i], color='purple', linestyle='--', linewidth=2, alpha=0.7)
                ax1.annotate(f'BREAKEVEN\nIter {iterations[i]}', 
                            xy=(iterations[i], c), xytext=(iterations[i]+0.3, c*1.05),
                            fontsize=10, color='purple', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='purple'))
                breakeven_found = True
                break
        
        plt.tight_layout()
        
        # Save to optimization_results folder with iteration number
        save_dir = Path("optimization_results")
        save_dir.mkdir(exist_ok=True)
        
        if current_iteration:
            save_path = save_dir / f"economic_analysis_iter_{current_iteration:02d}.png"
        else:
            save_path = save_dir / "economic_analysis.png"
            
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Graph] Saved: {save_path}")
        plt.close(fig)


    def plot_sequence_curve(self, df_results, save_path=None):
        import matplotlib.pyplot as plt
        if df_results.empty: return
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Número de Tanques (Agregados por Ranking)')
        ax1.set_ylabel('Costo Total ($)', color=color)
        ax1.plot(df_results['n_tanks'], df_results['cost'], color=color, marker='o', label='Costo')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Volumen Inundación (m3)', color=color)  
        ax2.plot(df_results['n_tanks'], df_results['flooding_remaining'], color=color, marker='s', linestyle='--', label='Inundación Restante')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Análisis Secuencial: Costo vs Beneficio Marginal')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Curve saved to {save_path}")


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
        from pathlib import Path
        import geopandas as gpd
        
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
        ax7.set_title('🎯 MÉTRICAS CLAVE', fontsize=14, fontweight='bold', pad=10)
        
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

