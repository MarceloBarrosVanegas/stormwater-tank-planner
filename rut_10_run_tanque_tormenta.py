"""
Stormwater Tank Optimization Runner
===================================

This module defines the runner class for the optimization process.
It encapsulates data loading, environment setup, and execution.
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Add local modules to path
import config
config.setup_sys_path()

from rut_02_elevation import ElevationGetter, ElevationSource
from rut_15_optimizer import TankOptimizer
from rut_16_dynamic_evaluator import DynamicSolutionEvaluator
from rut_15_optimizer import GreedyTankOptimizer

class StormwaterOptimizationRunner:
    """
    Orchestrates the stormwater tank optimization workflow.
    """
    
    def __init__(self, project_root: Path = None, proj_to: CRS = CRS("EPSG:32717")):
        """
        Initialize the runner. Auto-detects project root if not provided.
        """
        self.project_root = Path(project_root) if project_root else Path(os.getcwd())
        print(f"Project Root: {self.project_root}")
        
        
        
        # Default configuration
        self.proj_to = proj_to
        self.optimizer = None
        self.evaluator = None
        self.nodes_gdf = None
        self.predios_gdf = None

    @staticmethod
    def validate_and_update_raster_crs(raster_path: Path, target_crs: CRS) -> Path:
        """
        Verifica el CRS del raster usando comparación lógica de objetos (pyproj).
        Si no coincide con target_crs, lo reproyecta.
        """
        with rasterio.open(raster_path) as src:
            # Verificar si el raster tiene CRS
            if src.crs is None:
                print(f"⚠️ ADVERTENCIA: El raster '{raster_path.name}' no tiene CRS definido.")
                print("   -> No se puede verificar ni reproyectar automáticamente. Se usará el archivo original.")
                return raster_path

            try:
                # rasterio.crs.CRS tiene to_wkt(), por lo que pyproj puede leerlo
                current_crs = CRS.from_user_input(src.crs)

                # La igualdad (==) en pyproj verifica si representan la misma proyección física
                if current_crs == target_crs:
                    print(f"El CRS del raster es correcto: {raster_path.name}")
                    return raster_path
            except Exception as e:
                print(f"No se pudo comparar CRS automáticamente ({e}). Se procederá a reproyectar.")

            print(f"CRS incorrecto o formato distinto en {raster_path.name}. Reproyectando a referencia del proyecto...")

            # Calcular la nueva transformación y dimensiones
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Definir nombre del nuevo archivo
            new_filename = f"{raster_path.stem}_reprojected{raster_path.suffix}"
            new_path = raster_path.parent / new_filename

            # Realizar la reproyección
            with rasterio.open(new_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )

            print(f"Raster reproyectado y guardado en: {new_path}")
            return new_path

    @staticmethod
    def ensure_crs(gdf, target_crs, name="layer"):
        if gdf.crs is None:
            print(f"⚠️ {name} no tiene CRS definido. No se puede reproyectar con seguridad.")
            return gdf

        try:
            # Convertir a objeto pyproj.CRS para comparación robusta (igual que en el raster)
            current_crs = CRS.from_user_input(gdf.crs)
            if current_crs == target_crs:
                print(f"CRS de {name} correcto. Omitiendo reproyección.")
                return gdf
        except Exception:
            pass # Si falla la comparación, asumimos que son diferentes

        print(f"Reproyectando {name} al CRS del proyecto...")
        gdf.to_crs(target_crs, inplace=True)

        return gdf

    def load_data(self, elev_file, predios_path, flooding_nodes_path):
        """Load and prepare all necessary spatial data."""
        print("Loading data...")

        elev_file = self.validate_and_update_raster_crs(elev_file, self.proj_to)

        # 1. Elevation Setup
        elev_source = ElevationSource(str(self.project_root), self.proj_to)
        elev_tree = elev_source.get_elev_source(
            [str(elev_file)], 
            check_unique_values=False,
            ellipsoidal2orthometric=False,
            m_ramales=None, 
            elevation_shift=0
        )
        self.getter = ElevationGetter(tree=elev_tree, m_ramales=None, threshold_distance=0.7)
        
        # 2. Load Predios
        self.predios_gdf = gpd.read_file(predios_path, engine='pyogrio')
        self.predios_gdf = self.ensure_crs(gdf=self.predios_gdf, target_crs=self.proj_to)
        coords = self.predios_gdf.geometry.centroid.get_coordinates().to_numpy()
        self.predios_gdf['z'] = self.getter.get_elevation_from_tree_coords(coords)
        
        # 3. Load Flooding Nodes
        self.nodes_gdf = gpd.read_file(flooding_nodes_path, engine='pyogrio')
        self.nodes_gdf = self.ensure_crs(gdf=self.nodes_gdf, target_crs=self.proj_to)

        print(f"  Loaded {len(self.nodes_gdf)} nodes and {len(self.predios_gdf)} predios.")

    def step_1_check_elevation_constraints(self, w_flow=0.5, w_dist=0.3, w_gap=0.2):
        """
        Step 1: Check Elevation Constraints & Heuristic Ranking
        -----------------------------------------------------
        Filters and ranks Node->Predio candidates based on physical feasibility and heuristic scoring.

        1. Feasibility Check:
           - Verifies gravity flow: Predio.z <= Node.InvertElevation
           
        2. Heuristic Scoring (Normalized 0-1):
           - Calculates a weighted score to prioritize candidates that offer the best trade-off.
           - Formula: Score = (w_flow * Norm_Flow) + (w_gap * Norm_Gap) + (w_dist * (1 - Norm_Dist))
           
        Args:
            w_flow (float): Weight for Flooding Flow (Benefit). Prioritizes nodes with high flooding.
            w_dist (float): Weight for Distance (Cost). Prioritizes predios closer to the node.
            w_gap  (float): Weight for Elevation Gap (Hydraulic Benefit). Prioritizes larger vertical drops.
            
        Returns:
            list: Ranked list of valid candidate dictionaries.
        """
        print("\nElevation Feasibility & Weighted Ranking")
        
        if self.nodes_gdf is None or self.predios_gdf is None:
            print("Error: Data not loaded.")
            return

        n_nodes = len(self.nodes_gdf)
        n_predios = len(self.predios_gdf)
        
        valid_pairs = []
        
        print(f"Comparing {n_nodes} nodes against {n_predios} predios...")
        
        # 1. Collect all valid candidates
        for i, node in enumerate(self.nodes_gdf.itertuples()):
            node_invert = node.InvertElevation
            node_surface_z = node.InvertElevation + node.NodeDepth
            
            for j, predio in enumerate(self.predios_gdf.itertuples()):
                if predio.z <= node_invert:
                    dist = np.sqrt((node.geometry.x - predio.geometry.centroid.x)**2 + 
                                   (node.geometry.y - predio.geometry.centroid.y)**2)
                    gap = node_invert - predio.z
                    
                    valid_pairs.append({
                        'NodeID': node.NodeID,
                        'PredioID': j,
                        'NodeZ': node_surface_z,
                        'NodeInvert': node_invert,
                        'PredioZ': predio.z,
                        'Desnivel': gap,
                        'Distance': dist,
                        'FloodingFlow': node.FloodingFlow,
                        'PredioArea': predio.geometry.area  # For capacity constraint
                    })
        
        n_valid = len(valid_pairs)
        print(f"Total valid pairs found: {n_valid}")
        
        if n_valid == 0:
            return []

        # 2. Normalize values (Min-Max scaling)
        df_c = pd.DataFrame(valid_pairs)
        
        # Avoid division by zero if all values are same
        def normalize_series(series):
            if series.max() == series.min(): return np.ones(len(series))
            return (series - series.min()) / (series.max() - series.min())

        norm_flow = normalize_series(df_c['FloodingFlow'])
        norm_gap  = normalize_series(df_c['Desnivel'])
        norm_dist = normalize_series(df_c['Distance'])
        
        # 3. Calculate Weighted Score
        # Distance is "bad", so we take (1 - norm) to make it "closeness" (good)
        df_c['Score'] = (w_flow * norm_flow) + (w_gap * norm_gap) + (w_dist * (1.0 - norm_dist))
        
        # --- NEW: Filter to keep only the Best Predio per Node ---
        print("  Filtering candidates: Keeping only the BEST predio for each node...")
        # Sort by Score descending so the first one for each NodeID is the best
        df_c = df_c.sort_values('Score', ascending=False)
        
        # Drop duplicates keeping the first (best)
        df_c = df_c.drop_duplicates(subset='NodeID', keep='first')
        
        # Convert back to list of dicts
        ranked_pairs = df_c.to_dict('records')
        
        print(f"\n--- TOP 5 CANDIDATES (After Filtering) (Weights: Flow={w_flow}, Dist={w_dist}, Gap={w_gap}) ---")
        for k in range(min(10, len(ranked_pairs))):
            p = ranked_pairs[k]
            print(f"  #{k+1}: Node {p['NodeID']} -> Predio {p['PredioID']}")
            print(f"      Flow: {p['FloodingFlow']:.3f} m3/s | Dist: {p['Distance']:.1f} m | Desnivel: {p['Desnivel']:.2f} m")
            print(f"      Score: {p['Score']:.4f}")
            
        print("")
        
        self.valid_pairs = ranked_pairs
        return ranked_pairs

    def step_2_evaluate_case_scenarios(self, swmm_file: Path, elev_file: Path):
        """
        Step 2: Case Evaluation (Top 3 Candidates)
        ------------------------------------------
        Takes the Top 3 candidates from Step 1 and evaluates them as a single "Case".
        
        1.  **Aggregated Paths:** Calculates routes for all 3 tanks.
        2.  **Merged Design:** Merges routes into one GPKG and designs the network (rut_03).
        3.  **Simulation:** Simulates the system with all 3 tanks active.
        """
        print("\n--- STEP 2: Case Evaluation (Top 3 Candidates) ---")
        
        if not hasattr(self, 'valid_pairs') or not self.valid_pairs:
            print("Error: No candidates found. Run Step 1 first.")
            return

        # 1. Select Top 3 Candidates
        top_candidates = self.valid_pairs[:3]
        print(f"Selected Top {len(top_candidates)} Candidates for Case Evaluation:")
        for k, cand in enumerate(top_candidates):
            print(f"  #{k+1}: Node {cand['NodeID']} -> Predio {cand['PredioID']} (Score: {cand.get('Score',0):.4f})")
        
        # 2. Setup Evaluator
        if not hasattr(self, 'evaluator') or self.evaluator is None:
            print("  Initializing Dynamic Evaluator...")
            self.setup_optimization(use_dynamic=True, swmm_file=swmm_file, elev_file=elev_file)
            
        # 3. Prepare Assignment (Multiple Tanks)
        TEST_VOLUME = 1000.0 
        n_nodes = len(self.nodes_gdf)
        assignments = [(0, 0.0)] * n_nodes
        
        # Map Node IDs to indices
        node_id_to_idx = {row.NodeID: idx for idx, row in enumerate(self.nodes_gdf.itertuples())}
        
        for cand in top_candidates:
            n_id = cand['NodeID']
            if n_id in node_id_to_idx:
                idx = node_id_to_idx[n_id]
                assignments[idx] = (cand['PredioID'] + 1, TEST_VOLUME)
        
        print(f"  Running evaluation for Case (3 Tanks x {TEST_VOLUME}m3)...")
        print("  ... Aggregating paths, running SewerDesign, and Simulating SWMM ...")
        
        start_t = time.time()
        try:
            cost, remaining_flooding = self.evaluator.evaluate_solution(assignments)
            
            end_t = time.time()
            print(f"\n  >>> CASE EVALUATION COMPLETE in {end_t - start_t:.1f}s <<<")
            print(f"  REAL Cost: ${cost:,.2f}")
            print(f"  Remaining Flooding: {remaining_flooding:,.2f} m3")
            
            total_flood = self.nodes_gdf['FloodingVolume'].sum()
            reduction = total_flood - remaining_flooding
            print(f"  Flood Reduction: {reduction:,.2f} m3 ({(reduction/total_flood)*100:.1f}%)")
            
        except Exception as e:
            print(f"  ERROR during Case evaluation: {e}")
            import traceback
            traceback.print_exc()

    def step_3_run_sequential_analysis(self, max_tanks: int = 10, max_iterations: int = 50,
                                         min_tank_vol: float = 1000.0, max_tank_vol: float = 10000.0, 
                                         tank_depth: float = 5.0,
                                         stop_at_breakeven: bool = False, 
                                         breakeven_multiplier: float = 1.0,
                                         optimizer_mode: str = 'greedy',  # 'greedy' or 'nsga'
                                         optimization_tr_list: list = None,  # For NSGA: [25] or [1,2,5,10,25]
                                         validation_tr_list: list = None,    # For NSGA final validation
                                         n_generations: int = 50,            # NSGA generations
                                         pop_size: int = 30,                 # NSGA population
                                         swmm_file: Path = None, elev_file: Path = None):
        """
        Step 3: Sequential Tank Analysis
        ---------------------------------
        Runs optimization analysis using either:
        - 'greedy': Iterative greedy algorithm (fast, local optimum)
        - 'nsga': NSGA-II multi-objective optimization (slower, global optimum)
        
        NSGA-II Parameters:
        - optimization_tr_list: [25] for deterministic, [1,2,5,10,25] for probabilistic
        - validation_tr_list: Optional different TRs for final solution validation
        - n_generations: Number of NSGA-II generations
        - pop_size: Population size per generation
        
        Stopping Conditions (Greedy mode):
        - max_tanks: Maximum number of active tanks
        - max_iterations: Maximum number of attempts/iterations
        - stop_at_breakeven: When construction cost >= flooding savings
        """
        print(f"\n{'='*60}")
        print(f"STEP 3: SEQUENTIAL TANK ANALYSIS")
        print(f"{'='*60}")
        print(f"  Optimizer Mode: {optimizer_mode.upper()}")
        print(f"  Max Tanks: {max_tanks} | Volume Range: {min_tank_vol} - {max_tank_vol} m³")
        if optimizer_mode == 'nsga':
            optimization_tr_list = optimization_tr_list or [25]
            mode_type = 'probabilistic' if len(optimization_tr_list) > 1 else 'deterministic'
            print(f"  NSGA Mode: {mode_type} | TRs: {optimization_tr_list}")
            print(f"  Generations: {n_generations} | Population: {pop_size}")
        elif stop_at_breakeven:
            print(f"  Stopping Criterion: ECONOMIC BREAKEVEN (multiplier={breakeven_multiplier})")

        
        # 1. Setup Environment
        if self.nodes_gdf is None:
            self.step_1_check_elevation_constraints()
            
        if self.evaluator is None:
            print(f"\n  [Setup] Initializing SWMM Evaluator...")


            self.work_dir = Path(os.getcwd()) / "optimization_results"
            self.setup_optimization(use_dynamic=True,
                                    swmm_file=swmm_file,
                                    elev_file=elev_file,
                                    max_depth = tank_depth,
                                    min_tank_vol=min_tank_vol,
                                    max_tank_vol=max_tank_vol, )
        
        # 2. Verify candidates exist
        if not hasattr(self, 'valid_pairs') or not self.valid_pairs:
            print("  [Error] No valid candidates. Run step_1 first.")
            return
        
        print(f"\n  [Analysis] Candidates: {len(self.valid_pairs)} nodes with valid predios")

        # 3. Run optimization based on mode
        if optimizer_mode == 'nsga':
            # NSGA-II Multi-Objective Optimization
            self._run_nsga_optimization(
                max_tanks=max_tanks,
                min_tank_vol=min_tank_vol,
                max_tank_vol=max_tank_vol,
                optimization_tr_list=optimization_tr_list,
                validation_tr_list=validation_tr_list,
                n_generations=n_generations,
                pop_size=pop_size
            )
        else:
            # Greedy Sequential Optimization (default)
            greedy_opt = GreedyTankOptimizer(
                nodes_gdf=self.nodes_gdf,
                predios_gdf=self.predios_gdf,
                dynamic_evaluator=self.evaluator,
                max_tanks=max_tanks,
                max_iterations=max_iterations,
                min_tank_volume=min_tank_vol,
                max_tank_volume=max_tank_vol,
                tank_depth=tank_depth,
                stop_at_breakeven=stop_at_breakeven,
                breakeven_multiplier=breakeven_multiplier,
            )

            try:
                df_seq = greedy_opt.run_sequential(self.valid_pairs)
                
                print(f"\n{'='*60}")
                print("SEQUENTIAL ANALYSIS RESULTS:")
                print(f"{'='*60}")
                print(df_seq.to_string(index=False))
                
                # Save results
                csv_path = self.work_dir / "sequential_results.csv"
                df_seq.to_csv(csv_path, index=False)
                print(f"\n  Saved: {csv_path}")
                
                # Plot Curve
                curve_path = self.work_dir / "sequential_curve.png"
                greedy_opt.plot_sequence_curve(df_seq, save_path=str(curve_path))
                
                # Generate Damage Curves (rut_20) - ITZI is always used
                print(f"\n  Generating flood damage curves...")
                try:
                    from rut_20_plot_damage_curves import plot_all_curves_combined, plot_individual_curves, save_curves_as_csv
                    curves_dir = self.work_dir / "damage_curves"
                    curves_dir.mkdir(parents=True, exist_ok=True)
                    plot_all_curves_combined(output_dir=curves_dir)
                    plot_individual_curves(output_dir=curves_dir)
                    save_curves_as_csv(output_dir=curves_dir)
                    print(f"  Damage curves saved to: {curves_dir}")
                except Exception as e:
                    print(f"  Warning: Could not generate damage curves: {e}")
                
            except Exception as e:
                print(f"  [Error] Sequential analysis failed: {e}")
                import traceback
                traceback.print_exc()
        

    def _run_nsga_optimization(self, max_tanks: int, min_tank_vol: float, max_tank_vol: float,
                                optimization_tr_list: list, validation_tr_list: list,
                                n_generations: int, pop_size: int):
        """Run NSGA-II multi-objective optimization."""
        from rut_23_nsga_optimizer import NSGATankOptimizer
        import config
        
        print(f"\n  [NSGA-II] Initializing multi-objective optimizer...")
        
        # Convert valid_pairs to DataFrame if needed
        import pandas as pd
        if isinstance(self.valid_pairs, list):
            # Convert list of dicts to DataFrame
            valid_pairs_df = pd.DataFrame(self.valid_pairs)
        else:
            valid_pairs_df = self.valid_pairs
        
        # Filter to top N candidates based on ranking score (from Step 1)
        top_n_candidates = 50  # Use top 50 ranked candidates
        if len(valid_pairs_df) > top_n_candidates:
            # Sort by score (descending) and take top N
            if 'score' in valid_pairs_df.columns:
                valid_pairs_df = valid_pairs_df.nlargest(top_n_candidates, 'score')
                print(f"  [NSGA-II] Filtered to TOP {top_n_candidates} candidates (by ranking score)")
            else:
                valid_pairs_df = valid_pairs_df.head(top_n_candidates)
                print(f"  [NSGA-II] Filtered to first {top_n_candidates} candidates")
        
        # Create NSGA optimizer
        nsga_opt = NSGATankOptimizer(
            evaluator=self.evaluator,
            valid_pairs=valid_pairs_df,
            max_tanks=max_tanks,
            min_tank_vol=min_tank_vol,
            max_tank_vol=max_tank_vol,
            cost_components=config.COST_COMPONENTS
        )
        
        # Run optimization
        result = nsga_opt.run(
            optimization_tr_list=optimization_tr_list,
            validation_tr_list=validation_tr_list,
            n_generations=n_generations,
            pop_size=pop_size,
            output_dir=self.work_dir / "nsga_results"
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("NSGA-II OPTIMIZATION RESULTS:")
        print(f"{'='*60}")
        print(f"  Total Pareto Solutions: {len(result.solutions)}")
        print(f"  Mode: {result.mode.upper()}")
        print(f"\n  BEST B/C SOLUTION:")
        best = result.best_bc_solution
        print(f"    Tanks: {best.n_tanks}")
        print(f"    Construction Cost: ${best.construction_cost:,.2f}")
        print(f"    Damage Cost: ${best.damage_cost:,.2f}")
        print(f"    B/C Ratio: {best.bc_ratio:.2f}")
        print(f"    Tank Locations: {best.tank_indices}")
        
        return result


    def setup_optimization(self, use_dynamic: bool = True,
                           swmm_file: Path = None,
                           elev_file: Path = None,
                           V_min: float = 1000.0,
                           V_max: float = 100000.0,
                           max_tanks: int = 20,
                           max_depth: float = 5.0,
                           min_tank_vol: float = 1000.0,
                           max_tank_vol: float = 100000.0):
        """Initialize the evaluator and optimizer components.
        
        Args:
            use_dynamic: Use dynamic SWMM simulation (True) or simple estimation (False)
            swmm_file: Path to SWMM .inp file
            elev_file: Path to elevation raster
        """
        
        
        if self.work_dir is None:
             self.work_dir = self.project_root / "codigos" / "optimization_results"
        
        self.swmm_file_original = swmm_file
             
        # Update Config with provided parameters
        import config
        config.TANK_DEPTH_M = max_depth
        config.TANK_MIN_VOLUME_M3 = min_tank_vol
        config.TANK_MAX_VOLUME_M3 = max_tank_vol

        self.evaluator = DynamicSolutionEvaluator(
            work_dir=self.work_dir,
            path_proy=self.project_root,
            inp_file=swmm_file,
            nodes_gdf=self.nodes_gdf,
            predios_gdf=self.predios_gdf,
            elev_files_list=[str(elev_file)],
            proj_to=self.proj_to,

        )
        
        # Initialize Optimizer
        self.optimizer = TankOptimizer(
            nodes_gdf=self.nodes_gdf,
            predios_gdf=self.predios_gdf,
            V_min=V_min,
            V_max=V_max,
            max_tanks=max_tanks,
            dynamic_evaluator=self.evaluator,
            use_dynamic_evaluation=use_dynamic
        )

    def run(self, n_gen: int = 50, pop_size: int = 20):
        """Execute the optimization loop."""
        if not self.optimizer:
            raise RuntimeError("Optimizer not initialized. Call setup_optimization() first.")
            
        print(f"\nStarting Optimization (Gen={n_gen}, Pop={pop_size})...")
        self.optimizer.run(n_gen=n_gen, pop_size=pop_size, seed=42)
        
        self.optimizer.print_pareto_front()
        self._save_results()

    def _save_results(self):
        """Generate and save result artifacts."""
        print("Saving results...")
        self.optimizer.plot_pareto_front(save_path=str(self.work_dir / "pareto_front.png"))
        self.optimizer.plot_dashboard(save_path=str(self.work_dir / "dashboard.png"))
        
        try:
            best_sol = self.optimizer.get_pareto_solutions().sort_values('cost').iloc[0]
            self.optimizer.plot_solution_map(
                solution_id=best_sol['solution_id'],
                inp_path=str(self.swmm_file_original) if hasattr(self, 'swmm_file') else None,
                save_path=str(self.work_dir / f"map_solution_{best_sol['solution_id']}.png")
            )
        except Exception as e:
            print(f"Could not plot best solution: {e}")

if __name__ == "__main__":

    # Usar CRS del proyecto desde config
    proj_to = config.PROJECT_CRS


    # Se asume ejecución desde carpeta codigos: root = parent dir
    project_root = config.PROJECT_ROOT

    elev_file = config.ELEV_FILE
    predios_path = config.PREDIOS_FILE
    flooding_nodes_path = config.FLOODING_NODES_FILE
    swmm_file = config.SWMM_FILE
    
    
    # Example Usage
    runner = StormwaterOptimizationRunner(project_root= project_root, proj_to=proj_to)
    
    # 1. Load Data
    runner.load_data(elev_file, predios_path, flooding_nodes_path)
  
    
    # Step 1: Check Elevation Constraints
    candidates = runner.step_1_check_elevation_constraints()


    # Step 3: Run Sequential Tank Analysis
    runner.step_3_run_sequential_analysis(
        max_tanks=10,              # Max active tanks (stopping condition 1)
        max_iterations=100,        # Max iterations (stopping condition 2)
        min_tank_vol=5000.0,       # Minimum tank size (m³)
        max_tank_vol=1000000.0,    # Very high - predio area will limit actual size
        tank_depth=7.0,            # Tank depth (m)
        stop_at_breakeven=True,    # Stop when cost >= threshold (condition 3)
        breakeven_multiplier=1000,  # Allow investment up to 1.5x avoided damage
        swmm_file=swmm_file,
        elev_file=elev_file,

        optimizer_mode = 'nsga',  # 'greedy' or 'nsga'
        optimization_tr_list = [25],  # For NSGA: [25] or [1,2,5,10,25]
        validation_tr_list= [1,2,5,10,25, 50 , 100],  # For NSGA final validation
        n_generations = 100,  # NSGA generations
        pop_size = 30,  # NSGA population
    )


    # para plastico verificar que no se salga de los pozos anteriores y poreseriotes
    # piedra, tuberia de homrigon simple no pueden presurizarse
    # controlar velocidades  entuberia segun material
    #criteriios de riesgo de coletores,






    

