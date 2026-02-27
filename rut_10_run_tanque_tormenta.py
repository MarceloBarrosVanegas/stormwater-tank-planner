"""
Stormwater Tank Optimization Runner
===================================

This module defines the runner class for the optimization process.
It encapsulates data loading, environment setup, and execution.
"""

import os
import geopandas as gpd
import pandas as pd
import time
from pathlib import Path
import numpy as np
from pyproj import CRS
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math



import osmnx as ox
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed  # Cambiar aquí
from threading import Lock  # Para thread-safety si es necesario
from shapely.geometry import Point, LineString
from pyproj import Transformer
from typing import List
from tqdm import tqdm

# Add local modules to path
import config
config.setup_sys_path()

from rut_16_dynamic_evaluator import DynamicSolutionEvaluator
from rut_15_optimizer import GreedyTankOptimizer
from rut_26_hydrological_impact import HydrologicalImpactAssessment



class StormwaterOptimizationRunner:
    """
    Orchestrates the stormwater tank optimization workflow.
    """
    
    def __init__(self, project_root: Path = None, proj_to: CRS = CRS("EPSG:32717"), eval_id: str = None):
        """
        Initialize the runner. Auto-detects project root if not provided.
        
        Args:
            project_root: Path to project root
            proj_to: Target CRS
            eval_id: Optional evaluation ID for NSGA runs (creates subfolder)
        """
        self.project_root = Path(project_root) if project_root else Path(os.getcwd())
        print(f"Project Root: {self.project_root}")
        
        # Default configuration
        self.proj_to = proj_to
        self.optimizer = None
        self.evaluator = None
        self.nodes_gdf = None
        self.predios_gdf = None
        self.eval_id = eval_id  # ID de evaluación para crear subcarpeta


    def run_sequential_analysis(self, 
                                         max_tanks: int = 10, 
                                         max_iterations: int = 50,
                                         min_tank_vol: float = 1000.0, 
                                         max_tank_vol: float = 10000.0,
                                         stop_at_breakeven: bool = False, 
                                         breakeven_multiplier: float = 1.0,
                                         optimizer_mode: str = 'greedy',  # 'greedy' or 'nsga'
                                         elev_file: Path = None,
                                         optimization_tr_list: list = None,  # For NSGA: [25] or [1,2,5,10,25]
                                         validation_tr_list: list = None,    # For NSGA final validation
                                         ranking_weights: dict = None,  # EXPLICITO: Pesos para ranking de nodos
                                         capacity_max_hd: float = None,  # EXPLICITO: h/D maximo para derivacion
                                         baseline_extractor_path: Path = None,  # EXPLICITO: Metricas baseline (no recorrer)
                                         ):
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
        
        EXPLICIT Parameters:
        - ranking_weights: Dict con pesos para ranking (ej: {'flow_node_flooding': 0.5, ...})
        - capacity_max_hd: Valor h/D maximo para calculo de capacidad de derivacion
        """
        print(f"\n{'='*100}")
        print(f"SEQUENTIAL TANK ANALYSIS")
        print(f"{'='*100}")
        print(f"  Optimizer Mode: {optimizer_mode.upper()}")
        print(f"  Max Tanks: {max_tanks} | Volume Range: {min_tank_vol} - {max_tank_vol} m³")
        print(f"  Stopping Criterion: Stop at Breakeven Enabled (Multiplier: {breakeven_multiplier})")
        
        # Aplicar parametros EXPLICITOS al config (antes de crear evaluador/optimizador)
        if ranking_weights is not None:
            print(f"\n  [Explicit Params] Aplicando ranking_weights:")
            for k, v in ranking_weights.items():
                if v > 0:
                    print(f"    {k}: {v:.4f}")
            config.FLOODING_RANKING_WEIGHTS = ranking_weights.copy()
        
        if capacity_max_hd is not None:
            print(f"  [Explicit Params] CAPACITY_MAX_HD: {capacity_max_hd:.4f}")
            config.CAPACITY_MAX_HD = capacity_max_hd
        
        # 1. Setup Environment
        if self.evaluator is None:
            print(f"\n  [Setup] Initializing SWMM Evaluator...")

            # Base work directory
            base_work_dir = Path(os.getcwd()) / "optimization_results"
            
            # Si hay eval_id, crear subcarpeta para esta evaluación
            if self.eval_id:
                self.work_dir = base_work_dir / "nsga_evaluations" / self.eval_id
                print(f"  [Eval ID] {self.eval_id} -> {self.work_dir}")
            else:
                self.work_dir = base_work_dir
                
            self.setup_optimization(
                elev_file=elev_file,
                ranking_weights=ranking_weights,
                capacity_max_hd=capacity_max_hd,
                baseline_extractor_path=baseline_extractor_path,
            )

            # 2. Run Optimization
            greedy_opt = GreedyTankOptimizer(
                dynamic_evaluator=self.evaluator,
                stop_at_breakeven=stop_at_breakeven,
                breakeven_multiplier=breakeven_multiplier,
            )

            try:
                result_object = greedy_opt.run_sequential()

                return result_object
                
                # print(f"\n{'='*60}")
                # print("SEQUENTIAL ANALYSIS RESULTS:")
                # print(f"{'='*60}")
                # print(df_seq.to_string(index=False))
                #
                # # Save results
                # csv_path = self.work_dir / "sequential_results.csv"
                # df_seq.to_csv(csv_path, index=False)
                # print(f"\n  Saved: {csv_path}")
                
                # # Generate Damage Curves (rut_20) - ITZI is always used
                # print(f"\n  Generating flood damage curves...")
                # try:
                #     from rut_20_plot_damage_curves import plot_all_curves_combined, plot_individual_curves, save_curves_as_csv
                #     curves_dir = self.work_dir / "damage_curves"
                #     curves_dir.mkdir(parents=True, exist_ok=True)
                #     plot_all_curves_combined(output_dir=curves_dir)
                #     plot_individual_curves(output_dir=curves_dir)
                #     save_curves_as_csv(output_dir=curves_dir)
                #     print(f"  Damage curves saved to: {curves_dir}")
                # except Exception as e:
                #     print(f"  Warning: Could not generate damage curves: {e}")
                
      
                
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
            min_tanks=config.MIN_TANKS,
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


    def setup_optimization(self,
                           elev_file: Path = None,
                           ranking_weights: dict = None,
                           capacity_max_hd: float = None,
                           baseline_extractor_path: Path = None
                           ):

        """Initialize the evaluator and optimizer components.
        
        Args:
            elev_file: Path to elevation raster
            ranking_weights: Dict con pesos explicitos para ranking
            capacity_max_hd: Valor h/D maximo explicito
            baseline_extractor_path: Ruta al pickle del extractor baseline (si None, se crea nuevo)
        """
        
        
        if self.work_dir is None:
             self.work_dir = self.project_root / "codigos" / "optimization_results"
        
        
        self.evaluator = DynamicSolutionEvaluator(
            work_dir=self.work_dir,
            path_proy=self.project_root,
            elev_files_list=[str(elev_file)],
            proj_to=self.proj_to,
            ranking_weights=ranking_weights,
            capacity_max_hd=capacity_max_hd,
            baseline_extractor_path=baseline_extractor_path,
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
    
    import multiprocessing
    multiprocessing.freeze_support()


    # set variables from config
    proj_to = config.PROJECT_CRS
    project_root = config.PROJECT_ROOT
    elev_file = config.ELEV_FILE
    predios_path = config.PREDIOS_FILE
    swmm_file = config.SWMM_FILE
    

    # 2. run Sequential Tank Analysis
    runner = StormwaterOptimizationRunner(project_root= project_root, proj_to=proj_to)
    runner.run_sequential_analysis(
        max_tanks=config.MAX_TANKS,              # Max active tanks (stopping condition 1)
        max_iterations=100,        # Max iterations (stopping condition 2)
        min_tank_vol=config.TANK_MIN_VOLUME_M3,       # Minimum tank size (m³)
        max_tank_vol=config.TANK_MAX_VOLUME_M3,    # Very high - predio area will limit actual size
        stop_at_breakeven=True,    # Stop when cost >= threshold (condition 3)
        breakeven_multiplier=50,  # Allow investment up to 1.5x avoided damage
        elev_file=elev_file,

        optimizer_mode = 'greedy',  # 'greedy' or 'nsga'
        optimization_tr_list = config.TR_LIST,  # For NSGA: [25] or [1,2,5,10,25]
        validation_tr_list= config.VALIDATION_TR_LIST,  # For NSGA final validation

    )


    # para plastico verificar que no se salga de los pozos anteriores y poreseriotes
    # piedra, tuberia de homrigon simple no pueden presurizarse
    # controlar velocidades  entuberia segun material
    #criteriios de riesgo de coletores,






    

