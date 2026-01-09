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
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pyproj import CRS


# Add paths for local modules
import config
config.setup_sys_path()



# Local imports
from rut_02_elevation import ElevationGetter, ElevationSource
from rut_13_cost_functions import CostCalculator


# Optional: DynamicSolutionEvaluator for real path-based evaluation
try:
    from rut_16_dynamic_evaluator import DynamicSolutionEvaluator
    DYNAMIC_EVALUATOR_AVAILABLE = True
except ImportError:
    DYNAMIC_EVALUATOR_AVAILABLE = False
    DynamicSolutionEvaluator = None

# Optional: pymoo for NSGA-II (installed separately)
try:
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.optimize import minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not installed. Run 'pip install pymoo' for optimization.")


if PYMOO_AVAILABLE:
    class SparseTankSampling(Sampling):
        """
        Custom sampling to initialize population with a sparse number of tanks.
        Respects 'max_tanks' constraint by only enabling ~target variables.
        """
        def __init__(self, target: int = 5, initial_solutions: np.ndarray = None):
            super().__init__()
            self.target = target
            self.initial_solutions = initial_solutions

        def _do(self, problem, n_samples, **kwargs):
            # 0=Assign (0..n_predios), 1=Volume (Vmin..Vmax)
            # Init everything to 0 for Assignment, Random for Volume
            X = np.full((n_samples, problem.n_var), 0, dtype=float)
            
            n_nodes = problem.n_var // 2
            
            for i in range(n_samples):
                # If we have initial solutions, use them for the first N individuals
                if self.initial_solutions is not None and i < self.initial_solutions.shape[0]:
                    X[i, :] = self.initial_solutions[i, :]
                    continue

                # Randomly pick 'target' nodes to have tanks
                # We cap target at n_nodes
                n_active = min(self.target, n_nodes)
                if n_active < 1: n_active = 1
                
                active_indices = np.random.choice(n_nodes, n_active, replace=False)
                
                for node_idx in active_indices:
                    # Get eligible predios for this node
                    valid_predios = problem.eligibility.get_eligible_predios(node_idx)
                    
                    if valid_predios:
                        # Pick random eligible predio (index into problem.predios_gdf)
                        # Assignment value is predio_idx + 1
                        predio_idx = np.random.choice(valid_predios)
                        X[i, node_idx * 2] = predio_idx + 1
                        
                        # Random volume [V_min, V_max]
                        X[i, node_idx * 2 + 1] = np.random.uniform(problem.V_min, problem.V_max)
                    else:
                        # If selected node has no eligible predios, it stays 0
                        pass
                        
            return X


@dataclass
class CandidatePair:
    """Represents a valid node-predio pairing."""
    node_id: str
    predio_id: int
    node_idx: int
    predio_idx: int
    flooding_flow: float      # m³/s
    flooding_volume: float    # m³
    predio_area: float        # m²
    predio_z: float           # m
    costo_suelo_m2: float     # USD/m²
    clasificacion_suelo: int  # 1-4


class EligibilityMatrix:
    """
    Calculates and stores valid node-predio pairings based on elevation constraint.
    
    A predio is eligible for a node if:
        predio.z <= node.z - node.NodeDepth (node invert elevation)
    """
    
    def __init__(self, nodes_gdf: gpd.GeoDataFrame, predios_gdf: gpd.GeoDataFrame):
        """
        Parameters
        ----------
        nodes_gdf : GeoDataFrame
            Flooding nodes with columns: NodeID, z, NodeDepth, FloodingFlow, FloodingVolume
        predios_gdf : GeoDataFrame
            Candidate land plots with columns: z, area (geometry.area), 
            costo_suelo_m2, clasificacion_suelo
        """
        self.nodes_gdf = nodes_gdf.copy()
        self.predios_gdf = predios_gdf.copy()
        
        # Ensure predio has required columns with defaults
        # borrar
        if 'costo_suelo_m2' not in self.predios_gdf.columns:
            self.predios_gdf['costo_suelo_m2'] = 50.0  # Default: $50/m²
        if 'clasificacion_suelo' not in self.predios_gdf.columns:
            self.predios_gdf['clasificacion_suelo'] = 2  # Default: medium
            
        self.matrix = self._build_matrix()
        self.valid_pairs = self._extract_valid_pairs()
        
    def _build_matrix(self) -> np.ndarray:
        """
        Build boolean eligibility matrix [n_nodes x n_predios].
        Constraint: 
        1. Elevation check: predio.z <= node_invert
        2. NEAREST check: Only the closest valid predio is marked True.
        """
        n_nodes = len(self.nodes_gdf)
        n_predios = len(self.predios_gdf)
        matrix = np.zeros((n_nodes, n_predios), dtype=bool)
        
        print("Building Eligibility Matrix (Nearest Predio Constraint)...")
        
        for i, node in enumerate(self.nodes_gdf.itertuples()):
            # Fixed: Use InvertElevation directly
            node_invert = getattr(node, 'InvertElevation', getattr(node, 'z', 0) - getattr(node, 'NodeDepth', 0))
            
            valid_indices = []
            distances = []
            
            for j, predio in enumerate(self.predios_gdf.itertuples()):
                # 1. Elevation Check
                if predio.z <= node_invert:
                    # Calculate distance
                    dist = node.geometry.distance(predio.geometry.centroid)
                    valid_indices.append(j)
                    distances.append(dist)
            
            if valid_indices:
                # 2. Find closest valid predio
                # We only allow connection to the NEAREST feasible one.
                min_dist_idx = np.argmin(distances)
                closest_predio_global_idx = valid_indices[min_dist_idx]
                
                # Set ONLY the closest one to True
                matrix[i, closest_predio_global_idx] = True
                    
        return matrix
    
    def _extract_valid_pairs(self) -> List[CandidatePair]:
        """Extract list of all valid node-predio pairs."""
        pairs = []
        for i, node in enumerate(self.nodes_gdf.itertuples()):
            for j, predio in enumerate(self.predios_gdf.itertuples()):
                if self.matrix[i, j]:
                    pairs.append(CandidatePair(
                        node_id=node.NodeID,
                        predio_id=predio.Index,
                        node_idx=i,
                        predio_idx=j,
                        flooding_flow=node.FloodingFlow,
                        flooding_volume=node.FloodingVolume,
                        predio_area=predio.geometry.area,
                        predio_z=predio.z,
                        costo_suelo_m2=predio.costo_suelo_m2,
                        clasificacion_suelo=predio.clasificacion_suelo
                    ))
        return pairs
    
    def get_eligible_predios(self, node_idx: int) -> List[int]:
        """Return list of eligible predio indices for a given node."""
        return list(np.where(self.matrix[node_idx, :])[0])
    
    def print_summary(self):
        """Print summary of eligibility matrix."""
        n_valid = self.matrix.sum()
        n_total = self.matrix.size
        print(f"Eligibility Matrix: {len(self.nodes_gdf)} nodes × {len(self.predios_gdf)} predios")
        print(f"Valid pairs: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)")
        print(f"Average eligible predios per node: {n_valid / len(self.nodes_gdf):.1f}")


class TankOptimizationProblem(Problem if PYMOO_AVAILABLE else object):
    """
    NSGA-II problem definition for stormwater tank placement.
    
    Decision Variables (per node):
    - predio_assignment[i] ∈ {0, 1, ..., n_predios}: 0 = no tank, else predio index + 1
    - volume[i] ∈ [V_min, V_max]: Tank volume (only used if predio_assignment > 0)
    
    Objectives:
    - F1: Minimize total cost (derivation + tank + land)
    - F2: Minimize total flooding volume (remaining after tanks)
    
    Constraints:
    - Each predio can only be used by one node (no duplicates except 0)
    - Soft penalty for exceeding max_tanks
    """
    
    def __init__(self, 
                 eligibility: EligibilityMatrix,
                 V_min: float = 100.0,
                 V_max: float = 5000.0,
                 max_tanks: int = 5,
                 penalty_factor: float = 100000.0):
        
        self.eligibility = eligibility
        self.n_nodes = len(eligibility.nodes_gdf)
        self.n_predios = len(eligibility.predios_gdf)
        self.V_min = V_min
        self.V_max = V_max
        self.max_tanks = max_tanks
        self.penalty_factor = penalty_factor
        
        # Decision variables: [predio_1, vol_1, predio_2, vol_2, ...]
        n_var = self.n_nodes * 2
        
        # Bounds: predio in [0, n_predios], volume in [V_min, V_max]
        xl = np.zeros(n_var)
        xu = np.zeros(n_var)
        for i in range(self.n_nodes):
            xl[i*2] = 0              # predio assignment lower bound
            xu[i*2] = self.n_predios # predio assignment upper bound
            xl[i*2 + 1] = V_min      # volume lower bound
            xu[i*2 + 1] = V_max      # volume upper bound
        
        if PYMOO_AVAILABLE:
            super().__init__(
                n_var=n_var,
                n_obj=2,
                n_constr=0,  # Constraints handled as penalties
                xl=xl,
                xu=xu
            )
        
        # Optional: Dynamic evaluator for real path-based costs
        self.dynamic_evaluator = None
        self.use_dynamic_evaluation = False
    
    def set_dynamic_evaluator(self, evaluator, enabled: bool = True):
        """Set the dynamic evaluator for real path-based cost calculation."""
        self.dynamic_evaluator = evaluator
        self.use_dynamic_evaluation = enabled
    
    def decode_individual(self, x: np.ndarray) -> List[Tuple[int, float]]:
        """
        Decode decision variable array into list of (predio_idx, volume) per node.
        predio_idx = 0 means no tank.
        """
        assignments = []
        for i in range(self.n_nodes):
            predio_raw = int(round(x[i*2]))
            volume = x[i*2 + 1]
            
            # predio_raw = 0 means no assignment
            # predio_raw > 0 means predio index (1-indexed)
            if predio_raw == 0:
                assignments.append((0, 0.0))
            else:
                predio_idx = predio_raw - 1  # Convert to 0-indexed
                # Check eligibility
                if predio_idx < self.n_predios and self.eligibility.matrix[i, predio_idx]:
                    assignments.append((predio_idx + 1, volume))  # Keep 1-indexed for clarity
                else:
                    assignments.append((0, 0.0))  # Invalid assignment
        return assignments
    
    def evaluate_simplified(self, assignments: List[Tuple[int, float]]) -> Tuple[float, float]:
        """
        Simplified evaluation using geometric distance.
        Returns base (cost, flooding) WITHOUT penalties.
        """
        total_cost = 0.0
        total_flooding_captured = 0.0
        
        # We process assignments to sum costs
        for node_idx, (predio_1idx, volume) in enumerate(assignments):
            if predio_1idx == 0:
                continue
                
            predio_idx = predio_1idx - 1
            node = self.eligibility.nodes_gdf.iloc[node_idx]
            predio = self.eligibility.predios_gdf.iloc[predio_idx]
            
            # Geometric distance
            distance = node.geometry.distance(predio.geometry.centroid)
            
            # Costs
            c_derivation = CostCalculator.calculate_derivation_cost(distance, node.FloodingFlow)
            c_tank = CostCalculator.calculate_tank_cost(volume)
            
            # Land
            TANK_DEPTH = 5.0
            area_required = volume / TANK_DEPTH * 1.2
            c_land = area_required * predio.costo_suelo_m2
            
            # Soil factor
            FACTOR_SUELO = {1: 1.0, 2: 1.15, 3: 1.35, 4: 1.60}
            factor = FACTOR_SUELO.get(predio.clasificacion_suelo, 1.0)
            c_tank *= factor
            
            total_cost += c_derivation + c_tank + c_land
            
            # Flooding
            flooding_captured = min(volume, node.FloodingVolume)
            total_flooding_captured += flooding_captured
            
        total_node_flooding = self.eligibility.nodes_gdf['FloodingVolume'].sum()
        remaining_flooding = total_node_flooding - total_flooding_captured
        
        return total_cost, remaining_flooding

    def _calculate_penalties(self, assignments: List[Tuple[int, float]]) -> float:
        """
        Calculate administrative penalties (duplicates, max tanks).
        """
        penalty_cost = 0.0
        used_predios = {}
        n_active_tanks = 0
        duplicate_count = 0
        
        for node_idx, (predio_1idx, volume) in enumerate(assignments):
            if predio_1idx == 0:
                continue
            
            predio_idx = predio_1idx - 1
            if predio_idx in used_predios:
                duplicate_count += 1
            else:
                used_predios[predio_idx] = node_idx
                n_active_tanks += 1
        
        # 1. Duplicates
        penalty_cost += duplicate_count * self.penalty_factor * 10.0
        
        # 2. Max tanks soft limit
        if n_active_tanks > self.max_tanks:
            excess = n_active_tanks - self.max_tanks
            penalty_cost += (excess ** 2) * self.penalty_factor
            
        return penalty_cost

    
    def _evaluate(self, x, out, *args, **kwargs):
        """Pymoo evaluation function."""
        n_individuals = x.shape[0]
        F = np.zeros((n_individuals, 2))
        
        for i in range(n_individuals):
            assignments = self.decode_individual(x[i])
            
            # Use dynamic evaluator (ALWAYS REAL EVALUATION requested)
            if self.dynamic_evaluator:
                cost, flooding = self.dynamic_evaluator.evaluate_solution(assignments)
            else:
                # Fallback only if strictly necessary, but user requested ALWAYS REAL
                print("WARNING: Dynamic Evaluator not provided! Cannot perform real evaluation.")
                raise ValueError("Dynamic Evaluator required for Real Evaluation.")
            
            # Add general penalties (duplicates, max tanks)
            penalties = self._calculate_penalties(assignments)
            cost += penalties

            
            F[i, 0] = cost
            F[i, 1] = flooding
        
        out["F"] = F


class TankOptimizer:
    """
    Main optimizer class for stormwater tank placement.
    
    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Flooding nodes with required columns.
    predios_gdf : GeoDataFrame
        Candidate land plots with required columns.
    V_min : float
        Minimum tank volume (m³).
    V_max : float
        Maximum tank volume (m³).
    max_tanks : int
        Soft limit on number of tanks.
    
    Examples
    --------
    >>> optimizer = TankOptimizer(nodes_gdf, predios_gdf)
    >>> results = optimizer.run(n_gen=50, pop_size=100)
    >>> optimizer.print_pareto_front()
    """
    
    def __init__(self,
                 nodes_gdf: gpd.GeoDataFrame,
                 predios_gdf: gpd.GeoDataFrame,
                 V_min: float = 100.0,
                 V_max: float = 5000.0,
                 max_tanks: int = 5,
                 dynamic_evaluator: 'DynamicSolutionEvaluator' = None,
                 use_dynamic_evaluation: bool = True): # Default to True
        
        self.nodes_gdf = nodes_gdf
        self.predios_gdf = predios_gdf
        self.V_min = V_min
        self.V_max = V_max
        self.max_tanks = max_tanks
        self.dynamic_evaluator = dynamic_evaluator
        self.use_dynamic_evaluation = use_dynamic_evaluation
        
        # Build eligibility matrix
        self.eligibility = EligibilityMatrix(nodes_gdf, predios_gdf)
        self.eligibility.print_summary()
        
        # Create optimization problem
        self.problem = TankOptimizationProblem(
            eligibility=self.eligibility,
            V_min=V_min,
            V_max=V_max,
            max_tanks=max_tanks
        )
        
        # Set dynamic evaluator on problem if provided
        if dynamic_evaluator: # Always try to set if provided
            self.problem.set_dynamic_evaluator(dynamic_evaluator, enabled=True)

        else:
            print("WARNING: Dynamic evaluation DISABLED (Evaluator not provided). Optimization will fail if Real Evaluation is expected.")
        
        self.result = None
        self.initial_candidates = [] # Store step 1 candidates here
        
    def set_candidates(self, candidates: List[dict]):
        """Set candidates from Step 1 for seeding."""
        self.initial_candidates = candidates
        
    def _create_seed_population(self, candidates: List[dict], pop_size: int) -> np.ndarray:
        """
        Create structured seed population from top candidates.
        Prioritizes MULTI-TANK solutions to maximize objective.
        """
        n_var = self.problem.n_var
        seeds = []
        
        # We want to fill the population with high-quality combinations.
        # Step 1 Candidates are sorted by Score (descending).
        
        # 1. The "Super Solution" (Top N combined, where N = max_tanks)
        # This is our bet on the absolute best individual locations working together.
        try:
            super_gene = np.zeros(n_var)
            count_super = 0
            for cand in candidates[:self.max_tanks]: 
                 node_rows = self.nodes_gdf[self.nodes_gdf['NodeID'] == cand['NodeID']]
                 if not node_rows.empty:
                    node_idx = self.nodes_gdf.index.get_loc(node_rows.index[0])
                    predio_idx = cand['PredioID'] 
                    super_gene[node_idx * 2] = predio_idx + 1
                    super_gene[node_idx * 2 + 1] = 4000.0 # Aggressive volume
                    count_super += 1
            if count_super > 0:
                seeds.append(super_gene)
        except Exception as e:
            print(f"[Seeding] Error creating super seed: {e}")
            
        # 2. Random Combinations of Top 50 Candidates
        # We want to create diverse solutions that use 50% to 100% of max_tanks
        # but ONLY choosing from the high-scoring candidates.
        top_candidates = candidates[:50] # Pool of best options
        
        # How many random seeds to generate? 
        # We leave some space for pure random (via Sampling class) if pop_size is large,
        # but if pop_size is small, we want mostly good seeds.
        n_random_seeds = pop_size - 1 
        if n_random_seeds < 1: n_random_seeds = 1
        
        for _ in range(n_random_seeds):
            gene = np.zeros(n_var)
            
            # How many tanks to place? 
            # Force DENSITY: Random [Half Capacity, Max Capacity]
            # User reported "only 1 or 2 nodes", so we force at least 50% usage
            min_tanks = max(2, int(self.max_tanks * 0.5))
            n_tanks = np.random.randint(min_tanks, self.max_tanks + 1)
            
            # Pick n_tanks from the top_candidates pool
            if len(top_candidates) >= n_tanks:
                chosen = np.random.choice(top_candidates, n_tanks, replace=False)
                
                for cand in chosen:
                    try:
                        node_rows = self.nodes_gdf[self.nodes_gdf['NodeID'] == cand['NodeID']]
                        if not node_rows.empty:
                            node_idx = self.nodes_gdf.index.get_loc(node_rows.index[0])
                            
                            predio_idx = cand['PredioID'] 
                            gene[node_idx * 2] = predio_idx + 1
                            
                            # Random Volume
                            gene[node_idx * 2 + 1] = np.random.uniform(1000.0, 5000.0)
                    except: pass
                seeds.append(gene)
                
        if not seeds: return None
        return np.array(seeds)
    
    def run(self, n_gen: int = 50, pop_size: int = 100, seed: int = 42) -> Optional[object]:
        """
        Run NSGA-II optimization.
        
        Parameters
        ----------
        n_gen : int
            Number of generations.
        pop_size : int
            Population size.
        seed : int
            Random seed for reproducibility.
        
        Returns
        -------
        pymoo.Result or None
            Optimization result, or None if pymoo not available.
        """
        if not PYMOO_AVAILABLE:
            print("Error: pymoo not installed. Cannot run optimization.")
            return None
        
        print(f"\nStarting NSGA-II optimization...")
        print(f"  Generations: {n_gen}")
        print(f"  Population: {pop_size}")
        print(f"  Decision variables: {self.problem.n_var}")
        print(f"  Target Max Tanks per solution: {self.max_tanks}")
        
        # Prepare Seeding if candidates provided
        initial_pop = None
        if hasattr(self, 'initial_candidates') and self.initial_candidates:
            print(f"  [Optimizer] Seeding population with {len(self.initial_candidates)} Top Candidates...")
            initial_pop = self._create_seed_population(self.initial_candidates, pop_size)
            
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=SparseTankSampling(target=self.max_tanks, initial_solutions=initial_pop), # Use sparse sampling with seeding
            crossover=SBX(prob=0.9, eta=15, vtype=float),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        self.result = minimize(
            self.problem,
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=True
        )
        
        print(f"\nOptimization complete!")
        print(f"Pareto front size: {len(self.result.F)}")
        
        return self.result
    
    def get_pareto_solutions(self) -> pd.DataFrame:
        """
        Get Pareto front solutions as a DataFrame.
        
        Returns
        -------
        DataFrame with columns: cost, flooding, assignments (decoded)
        """
        if self.result is None:
            print("No results. Run optimization first.")
            return pd.DataFrame()
        
        solutions = []
        for i, (x, f) in enumerate(zip(self.result.X, self.result.F)):
            assignments = self.problem.decode_individual(x)
            
            # Count unique predios (excluding duplicates, matching evaluation logic)
            used_predios = set()
            n_tanks = 0
            total_volume = 0.0
            for a, v in assignments:
                if a > 0:
                    predio_idx = a - 1
                    if predio_idx not in used_predios:
                        used_predios.add(predio_idx)
                        n_tanks += 1
                        total_volume += v
            
            solutions.append({
                'solution_id': i,
                'cost': f[0],
                'flooding_remaining': f[1],
                'n_tanks': n_tanks,
                'total_volume': total_volume,
                'assignments': assignments
            })
        
        return pd.DataFrame(solutions).sort_values('cost')
    
    def print_pareto_front(self):
        """Print summary of Pareto front solutions."""
        df = self.get_pareto_solutions()
        if df.empty:
            return
        
        print("\n" + "="*60)
        print("PARETO FRONT SOLUTIONS")
        print("="*60)
        
        for _, row in df.iterrows():
            print(f"\nSolution {row['solution_id']}:")
            print(f"  Cost: ${row['cost']:,.2f}")
            print(f"  Flooding Remaining: {row['flooding_remaining']:,.2f} m³")
            print(f"  N Tanks: {row['n_tanks']}")
            print(f"  Total Volume: {row['total_volume']:,.2f} m³")
    
    def plot_pareto_front(self, figsize=(10, 8), save_path=None):
        """
        Plot the Pareto front (cost vs flooding reduction).
        
        Parameters
        ----------
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the figure.
        """
        import matplotlib.pyplot as plt
        
        df = self.get_pareto_solutions()
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate flooding reduction (benefit)
        total_flooding = self.nodes_gdf['FloodingVolume'].sum()
        df['flooding_reduced'] = total_flooding - df['flooding_remaining']
        df['reduction_pct'] = (df['flooding_reduced'] / total_flooding) * 100
        
        # Scatter plot with solution IDs
        scatter = ax.scatter(
            df['cost'] / 1e6,  # Convert to millions
            df['reduction_pct'],
            c=df['n_tanks'],
            cmap='viridis',
            s=200,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.8
        )
        
        # Add labels for each point
        for _, row in df.iterrows():
            ax.annotate(
                f"Sol {row['solution_id']}\n{row['n_tanks']} tanks\n{row['total_volume']:,.0f}m³",
                (row['cost'] / 1e6, row['reduction_pct']),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=8,
                ha='left'
            )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Número de Tanques', fontsize=11)
        
        # Labels and title
        ax.set_xlabel('Costo Total (Millones USD)', fontsize=12)
        ax.set_ylabel('Reducción de Inundación (%)', fontsize=12)
        ax.set_title('Frente de Pareto: Costo vs Reducción de Inundación', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add stats box
        stats_text = (
            f"Total Flooding: {total_flooding:,.0f} m³\n"
            f"Nodos: {len(self.nodes_gdf)}\n"
            f"Predios: {len(self.predios_gdf)}\n"
            f"Max Tanks: {self.max_tanks}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_solution_map(self, solution_id: int, inp_path: str = None, figsize=(14, 10), save_path=None):
        """
        Plot a specific solution showing existing pipes, new derivations, nodes, and tanks.
        
        Parameters
        ----------
        solution_id : int
            ID of the solution to plot.
        inp_path : str, optional
            Path to the baseline .inp file to plot the existing network context.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        from shapely.geometry import LineString
        
        # Try importing swmmio for background network
        try:
            import swmmio
            SWMMIO_AVAILABLE = True
        except ImportError:
            SWMMIO_AVAILABLE = False
            print("Warning: swmmio not installed. Background pipes will not be shown.")
        
        df = self.get_pareto_solutions()
        if df.empty or solution_id not in df['solution_id'].values:
            print(f"Solution {solution_id} not found.")
            return None
        
        sol_row = df[df['solution_id'] == solution_id].iloc[0]
        assignments = sol_row['assignments']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # --- 1. Plot Existing Network (Background) ---
        if SWMMIO_AVAILABLE and inp_path and os.path.exists(inp_path):
            try:
                model = swmmio.Model(inp_path)
                conduits = model.conduits()
                
                # Convert coords to geometries
                # swmmio 'coords' col is list of [(x1,y1), (x2,y2)]
                lines = []
                for _, row in conduits.iterrows():
                    if 'coords' in row and isinstance(row['coords'], list):
                        if len(row['coords']) >= 2:
                            lines.append(LineString(row['coords']))
                
                if lines:
                    gpd.GeoSeries(lines).plot(ax=ax, color='lightgray', linewidth=1.0, alpha=0.5, zorder=1)
            except Exception as e:
                print(f"Could not load background network: {e}")

        # --- 2. Plot Predios ---
        # Plot all predios (faint)
        self.predios_gdf.plot(ax=ax, color='whitesmoke', edgecolor='lightgray', alpha=0.5, zorder=0)


        # --- 3. Plot Nodes (Context) ---
        # Plot all nodes (small dots)
        self.nodes_gdf.plot(ax=ax, color='gray', markersize=3, alpha=0.3, zorder=2)
        
        
        # --- 4. Plot Active Assignments (Tanks & Derivations) ---
        used_predios = {}  # predio_idx -> (node_idx, volume)
        for node_idx, (predio_1idx, volume) in enumerate(assignments):
            if predio_1idx > 0:
                predio_idx = predio_1idx - 1
                if predio_idx not in used_predios:
                    used_predios[predio_idx] = (node_idx, volume)
        
        # Colors for tanks
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(used_predios))))
        
        for i, (predio_idx, (node_idx, volume)) in enumerate(used_predios.items()):
            predio = self.predios_gdf.iloc[predio_idx]
            node = self.nodes_gdf.iloc[node_idx]
            color = colors[i % len(colors)]
            
            # A. Highlight Predio (Tank Location)
            gpd.GeoSeries([predio.geometry]).plot(
                ax=ax, color=color, edgecolor='black', linewidth=1, alpha=0.6, zorder=3
            )
            
            # B. Highlight Source Node
            ax.scatter(
                node.geometry.x, node.geometry.y,
                c=[color], s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=4
            )
            
            # C. Draw Derivation Line (Blue Thick)
            # Straight line for visualization (unless we have the actual routes GDF)
            # Use 'Blue' specifically as requested, or the matched color? 
            # User asked for "azul grueso width 3" for derivations.
            ax.plot(
                [node.geometry.x, predio.geometry.centroid.x],
                [node.geometry.y, predio.geometry.centroid.y],
                color='blue', linewidth=3.0, linestyle='-', alpha=0.8, zorder=3.5
            )
            
            # D. Label Tank
            ax.annotate(
                f"T{i+1}\n{volume:.0f}m³",
                (predio.geometry.centroid.x, predio.geometry.centroid.y),
                fontsize=9, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', alpha=0.9, edgecolor=color)
            )
        
        # --- 5. Aesthetics ---
        total_flooding = self.nodes_gdf['FloodingVolume'].sum()
        reduction_pct = ((total_flooding - sol_row['flooding_remaining']) / total_flooding) * 100
        
        title = (
            f"Mapa de Solución {solution_id}\n"
            f"Tanques: {sol_row['n_tanks']} | Costo: ${sol_row['cost']/1e6:.2f}M | "
            f"Reducción Inundación: {reduction_pct:.1f}%"
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Este (m)')
        ax.set_ylabel('Norte (m)')
        
        # Custom Legend
        legend_elements = [
            Line2D([0], [0], color='lightgray', lw=1, label='Red Existente'),
            Line2D([0], [0], color='blue', lw=3, label='Nueva Derivación'),
            Line2D([0], [0], marker='o', color='gray', label='Nodos', markerfacecolor='gray', markersize=5),
            Patch(facecolor='green', edgecolor='black', alpha=0.5, label='Tanque Propuesto')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved map to: {save_path}")
        
        return fig
    
    def plot_dashboard(self, figsize=(16, 12), save_path=None):
        """
        Create a comprehensive dashboard with Pareto front and best solutions.
        
        Parameters
        ----------
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        df = self.get_pareto_solutions()
        if df.empty:
            return None
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])
        
        # === Panel 1: Pareto Front ===
        ax1 = fig.add_subplot(gs[0, 0])
        
        total_flooding = self.nodes_gdf['FloodingVolume'].sum()
        df['flooding_reduced'] = total_flooding - df['flooding_remaining']
        df['reduction_pct'] = (df['flooding_reduced'] / total_flooding) * 100
        
        scatter = ax1.scatter(
            df['cost'] / 1e6,
            df['reduction_pct'],
            c=df['n_tanks'],
            cmap='viridis',
            s=150,
            edgecolors='black',
            linewidths=1.5
        )
        
        for _, row in df.iterrows():
            ax1.annotate(f"{row['solution_id']}", (row['cost']/1e6, row['reduction_pct']),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        ax1.set_xlabel('Costo (Millones USD)', fontsize=10)
        ax1.set_ylabel('Reducción Inundación (%)', fontsize=10)
        ax1.set_title('Frente de Pareto', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='N Tanques')
        
        # === Panel 2: Cost Breakdown ===
        ax2 = fig.add_subplot(gs[0, 1])
        
        x = np.arange(len(df))
        width = 0.35
        
        ax2.bar(x, df['cost']/1e6, width, label='Costo Total', color='steelblue')
        ax2.bar(x + width, df['flooding_reduced']/1000, width, label='Flooding Reducido (×1000 m³)', color='seagreen')
        
        ax2.set_xlabel('Solución ID', fontsize=10)
        ax2.set_ylabel('Valor', fontsize=10)
        ax2.set_title('Comparación de Soluciones', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(df['solution_id'].astype(str))
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # === Panel 3: Map of best solution (lowest cost) ===
        ax3 = fig.add_subplot(gs[1, 0])
        
        best_cost_sol = df.iloc[0]  # Already sorted by cost
        self._plot_solution_on_ax(ax3, best_cost_sol, "Mejor Costo")
        
        # === Panel 4: Map of best reduction ===
        ax4 = fig.add_subplot(gs[1, 1])
        
        best_reduction_sol = df.loc[df['reduction_pct'].idxmax()]
        self._plot_solution_on_ax(ax4, best_reduction_sol, "Mayor Reducción")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def _plot_solution_on_ax(self, ax, sol_row, subtitle):
        """Helper to plot a solution on a given axis."""
        assignments = sol_row['assignments']
        
        # Plot predios
        self.predios_gdf.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.5)
        
        # Plot nodes
        self.nodes_gdf.plot(ax=ax, color='blue', markersize=3, alpha=0.2)
        
        # Find used predios
        used_predios = {}
        for node_idx, (predio_1idx, volume) in enumerate(assignments):
            if predio_1idx > 0:
                predio_idx = predio_1idx - 1
                if predio_idx not in used_predios:
                    used_predios[predio_idx] = (node_idx, volume)
        
        # Plot used predios
        for predio_idx, (node_idx, volume) in used_predios.items():
            predio = self.predios_gdf.iloc[predio_idx]
            node = self.nodes_gdf.iloc[node_idx]
            
            gpd.GeoSeries([predio.geometry]).plot(ax=ax, color='green', edgecolor='black', linewidth=1.5, alpha=0.7)
            ax.scatter(node.geometry.x, node.geometry.y, c='red', s=50, zorder=5, edgecolors='black')
            ax.plot([node.geometry.x, predio.geometry.centroid.x],
                   [node.geometry.y, predio.geometry.centroid.y],
                   'k--', linewidth=1, alpha=0.5)
        
        total_flooding = self.nodes_gdf['FloodingVolume'].sum()
        reduction_pct = ((total_flooding - sol_row['flooding_remaining']) / total_flooding) * 100
        
        ax.set_title(f"{subtitle} - Sol {sol_row['solution_id']}\n"
                    f"${sol_row['cost']/1e6:.1f}M | {reduction_pct:.1f}% red. | {sol_row['n_tanks']} tanks",
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.grid(True, alpha=0.2)


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
                 nodes_gdf: gpd.GeoDataFrame, 
                 predios_gdf: gpd.GeoDataFrame,
                 dynamic_evaluator: Optional[object] = None,
                 max_tanks: int = 10,
                 max_iterations: int = 50,         # Max iterations (attempts)
                 min_tank_volume: float = 1000.0,  # Minimum tank size (m³)
                 max_tank_volume: float = 10000.0, # Maximum tank size (m³)
                 tank_depth: float = 5.0,          # Tank depth (m)
                 stop_at_breakeven: bool = False,  # Stop when cost >= savings * multiplier
                 breakeven_multiplier: float = 1.0,  # Allow investment up to X times avoided damage
                 flooding_cost_per_m3: float = 1250.0):  # $/m³ flooding damage
        self.nodes_gdf = nodes_gdf
        self.predios_gdf = predios_gdf
        self.evaluator = dynamic_evaluator
        self.max_tanks = max_tanks
        self.max_iterations = max_iterations
        self.min_tank_volume = min_tank_volume
        self.max_tank_volume = max_tank_volume
        self.tank_depth = tank_depth
        self.stop_at_breakeven = stop_at_breakeven
        self.breakeven_multiplier = breakeven_multiplier
        self.flooding_cost_per_m3 = flooding_cost_per_m3
        
    def run_sequential(self, candidates: List[dict]) -> pd.DataFrame:
        """
        Run ITERATIVE sequential evaluation:
        1. Select best candidate based on current flooding
        2. Run SWMM simulation
        3. Read residual flooding from SWMM output
        4. Re-rank remaining candidates based on NEW flooding
        5. Repeat until max_tanks reached
        
        This ensures each step selects the truly best candidate based on
        the ACTUAL flooding situation after previous interventions.
        """
        results = []
        active_candidates = []
        used_nodes = set()  # Track which nodes already have tanks
        
        # Track AREA usage per predio
        predio_capacity = {}
        
        # Constants for area calculation
        TANK_DEPTH = config.TANK_DEPTH_M  # Use configurable depth
        OCCUPATION_FACTOR = config.TANK_OCCUPATION_FACTOR  # Extra space for access, pumps, maneuvering
        # Use configurable volume limits from instance
        MIN_TANK_VOLUME = config.TANK_MIN_VOLUME_M3
        MAX_TANK_VOLUME = config.TANK_MAX_VOLUME_M3
        
        # Pre-calculate available area for each predio
        for idx, row in self.predios_gdf.iterrows():
            predio_idx = idx if isinstance(idx, int) else self.predios_gdf.index.get_loc(idx)
            area_m2 = row.geometry.area if hasattr(row.geometry, 'area') else 0.0
            predio_capacity[predio_idx] = {
                'total_area': area_m2,
                'used_area': 0.0,
                'n_tanks': 0
            }
        
        # Create mutable copy of candidates with flooding values
        # FloodingVolume is calculated from baseline metrics if available
        remaining_candidates = []
        for c in candidates:
            # Get flooding volume from baseline metrics if available
            flooding_vol = 0.0
            if hasattr(self.evaluator, 'baseline_metrics') and self.evaluator.baseline_metrics:
                flooding_vol = self.evaluator.baseline_metrics.flooding_volumes.get(c['NodeID'], 0.0)
            
            # Get node elevation from nodes_gdf for tie-breaking (higher = upstream)
            node_elev = 0.0
            node_row = self.nodes_gdf[self.nodes_gdf['NodeID'] == c['NodeID']]
            if not node_row.empty:
                node_elev = node_row.iloc[0].get('InvertElevation', node_row.iloc[0].get('z', 0))
            
            # Calculate dynamic tank volume based on flooding (clamped to min/max)
            tank_volume = max(MIN_TANK_VOLUME, min(flooding_vol, MAX_TANK_VOLUME))
            
            remaining_candidates.append({
                'NodeID': c['NodeID'],
                'PredioID': c['PredioID'],
                'FloodingFlow': c.get('FloodingFlow', 0.0),
                'FloodingVolume': flooding_vol,
                'TankVolume': tank_volume,
                'Score': c.get('Score', 0.0),
                'NodeElevation': node_elev  # For tie-breaking: prefer higher (upstream)
            })

        
        # Track tanks pruned - after MAX_PRUNE_RETRIES, permanently exclude
        pruning_failures = {}  # {NodeID: failure_count}
        permanently_excluded = set()  # Nodes that have exhausted all retries
        MAX_PRUNE_RETRIES = 2  # Reduced: if tank doesn't work in 2 tries, exclude it
        
        # Economic tracking for graph
        economic_history = []  # [(iteration, cost, savings, n_tanks)]
        last_tank_config = None  # Track tank config to skip redundant graphs
        
        stop_mode = "BREAKEVEN" if self.stop_at_breakeven else f"Max {self.max_tanks} tanks / {self.max_iterations} iters"
        print(f"\n=== ITERATIVE Greedy Sequential Analysis ({stop_mode}) ===")
        print(f"  Tank Depth: {TANK_DEPTH}m | Volume Range: {MIN_TANK_VOLUME}-{MAX_TANK_VOLUME} m³")
        print(f"  Max Prune Retries: {MAX_PRUNE_RETRIES}")


        
        for iteration in range(1, self.max_iterations + 1):
            # Check if max tanks reached
            if len(active_candidates) >= self.max_tanks:
                print(f"\n  *** MAX TANKS ({self.max_tanks}) REACHED - Stopping ***")
                break
            
            # Sort remaining candidates by:
            # 1. FloodingFlow (highest first) - main priority
            # 2. NodeElevation (highest first) - tie-breaker: prefer upstream nodes
            # This ensures when multiple nodes flood on same line, upstream is processed first
            remaining_candidates.sort(key=lambda x: (x['FloodingFlow'], x['NodeElevation']), reverse=True)

            
            # Find best valid candidate (not used, has predio space)
            selected = None
            selected_area = 0
            
            for cand in remaining_candidates:
                if cand['NodeID'] in used_nodes:
                    continue
                
                # Skip if node is permanently excluded (exhausted all retries)
                if cand['NodeID'] in permanently_excluded:
                    continue
                
                # Skip if node has exceeded max pruning retries
                if pruning_failures.get(cand['NodeID'], 0) >= MAX_PRUNE_RETRIES:
                    permanently_excluded.add(cand['NodeID'])
                    print(f"  [Exclude] {cand['NodeID']} permanently excluded after {MAX_PRUNE_RETRIES} failed retries")
                    continue

                
                predio_idx = cand['PredioID']
                if predio_idx not in predio_capacity:
                    continue
                    
                cap = predio_capacity[predio_idx]
                available_area = cap['total_area'] - cap['used_area']
                
                # Calculate max tank volume that fits in available area
                # Area = (Volume / Depth) * OccupationFactor
                # Volume = (Area * Depth) / OccupationFactor
                max_vol_from_area = (available_area / OCCUPATION_FACTOR) * TANK_DEPTH
                
                # Adjust tank volume to fit available area
                adjusted_volume = min(cand['TankVolume'], max_vol_from_area)
                
                # Round to nearest 1000 m³ multiple (ceiling to ensure capacity)
                adjusted_volume = int(np.ceil(adjusted_volume / 1000.0) * 1000)
                
                # Skip if adjusted volume is below minimum
                if adjusted_volume < MIN_TANK_VOLUME:
                    continue
                
                # Update candidate's volume to the adjusted value
                cand['TankVolume'] = adjusted_volume
                required_area = (adjusted_volume / TANK_DEPTH) * OCCUPATION_FACTOR
                
                selected = cand
                selected_area = required_area
                break
            
            if selected is None:
                print(f"\n  [Stop] No more valid candidates at iteration {iteration}")
                break
            
            # Mark as used
            used_nodes.add(selected['NodeID'])
            predio_idx = selected['PredioID']
            predio_capacity[predio_idx]['used_area'] += selected_area
            predio_capacity[predio_idx]['n_tanks'] += 1
            n_tanks_in_predio = predio_capacity[predio_idx]['n_tanks']
            
            active_candidates.append(selected)
            
            # Clear section header for this step
            print(f"\n{'═'*60}")
            print(f"  ITERATION {iteration}: Adding Tank @ {selected['NodeID']}")
            print(f"{'═'*60}")
            print(f"  Predio: {predio_idx} (Tank #{n_tanks_in_predio})")
            print(f"  Flooding Volume: {selected['FloodingVolume']:,.0f} m³")
            print(f"  Tank Volume: {selected['TankVolume']:,.0f} m³")
            print(f"  Area Used: {selected_area:,.0f} m² | Remaining: {predio_capacity[predio_idx]['total_area'] - predio_capacity[predio_idx]['used_area']:,.0f} m²")
            
            # Encode all active candidates for evaluator
            if self.evaluator:
                n_nodes = len(self.nodes_gdf)
                n_var = n_nodes * 2
                gene = np.zeros(n_var)
                
                for ac in active_candidates:
                    node_rows = self.nodes_gdf[self.nodes_gdf['NodeID'] == ac['NodeID']]
                    if not node_rows.empty:
                        node_idx = self.nodes_gdf.index.get_loc(node_rows.index[0])
                        gene[node_idx * 2] = ac['PredioID'] + 1
                        gene[node_idx * 2 + 1] = ac['TankVolume']  # Dynamic volume per tank
                
                # Decode for evaluator
                assignments = []
                for j in range(0, n_var, 2):
                    p_val = int(gene[j])
                    v_val = gene[j+1]
                    assignments.append((p_val, v_val))
                
                print(f"  Evaluating configuration with {len(active_candidates)} tanks...")
                cost, flooding_vol = self.evaluator.evaluate_solution(
                    assignments, 
                    solution_name=f"Seq_Iter_{iteration:02d}"
                )
                
                results.append({
                    'step': iteration,
                    'n_tanks': len(active_candidates),
                    'cost': cost,
                    'flooding_remaining': flooding_vol,
                    'added_node': selected['NodeID']
                })
                
                # === ECONOMIC TRACKING (for graphing) ===
                # Get CLIMADA damage from evaluator
                current_damage = getattr(self.evaluator, 'last_flood_damage_usd', 0)
                baseline_damage = getattr(self.evaluator, 'baseline_flood_damage', 0)
                
                # Get Infrastructure Benefit (Deferred Investment savings)
                infrastructure_benefit = getattr(self.evaluator, 'last_economic_result', {}).get('infrastructure_benefit', 0)
                
                # Total Avoided Cost = CLIMADA savings + Infrastructure Benefit
                climada_savings = baseline_damage - current_damage
                cost_saved = climada_savings + infrastructure_benefit
                
                # Get flooding volume from metrics for display
                if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:
                    flooding_vol_display = self.evaluator.last_metrics.total_flooding_volume
                else:
                    flooding_vol_display = flooding_vol
                baseline_flood = self.evaluator.baseline_metrics.total_flooding_volume
                flooding_reduced = baseline_flood - flooding_vol_display
                
                print(f"  [Economics] Baseline CLIMADA Damage: ${baseline_damage:,.2f}")
                print(f"  [Economics] Current CLIMADA Damage:  ${current_damage:,.2f}")
                print(f"  [Economics] Avoided Cost:            ${cost_saved:,.2f}")
                
                # Track for plotting
                economic_history.append({
                    'iteration': iteration,
                    'cost': cost,
                    'savings': cost_saved,
                    'n_tanks': len(active_candidates),
                    'flooding_reduced': flooding_reduced,
                    'flooding_remaining': flooding_vol_display,
                    'baseline_flooding': baseline_flood
                })
                
                # Calculate benefit-to-cost ratio
                if cost > 0:
                    benefit_ratio = cost_saved / cost
                else:
                    benefit_ratio = float('inf')
                
                print(f"  [Economic] Cost: ${cost:,.0f} | Savings: ${cost_saved:,.0f} | B/C Ratio: {benefit_ratio:.2f}")
                
                # === GENERATE ECONOMIC GRAPH PER ITERATION (only if config changed) ===
                current_config = tuple(sorted([c['NodeID'] for c in active_candidates]))
                if economic_history and current_config != last_tank_config:
                    self._plot_economic_curve(economic_history, iteration)
                    last_tank_config = current_config
                elif current_config == last_tank_config:
                    print(f"  [Graph] Skipped (config unchanged)")
                
                # === BREAKEVEN CHECK ===
                # Stop when cost >= savings * multiplier (e.g., multiplier=1.5 allows 50% more investment)
                threshold = cost_saved * self.breakeven_multiplier
                if self.stop_at_breakeven and cost >= threshold:
                    print(f"\n  *** BREAKEVEN REACHED ***")
                    print(f"  Construction cost (${cost:,.0f}) >= Threshold (${threshold:,.0f})")
                    print(f"  [Threshold = Savings × {self.breakeven_multiplier:.2f}]")
                    print(f"  Stopping optimization at {len(active_candidates)} tanks.")
                    break

                
                # === ITERATIVE RE-RANKING ===
                # Read residual flooding from SWMM output and update candidates
                if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:

                    metrics = self.evaluator.last_metrics
                    print(f"  [Re-Rank] Updating candidate flooding values from SWMM output...")
                    
                    updated_count = 0
                    for cand in remaining_candidates:
                        if cand['NodeID'] in used_nodes:
                            continue
                        
                        # Get new flooding value from SWMM output
                        new_flood = metrics.flooding_volumes.get(cand['NodeID'], 0.0)
                        if new_flood != cand['FloodingVolume']:
                            updated_count += 1
                        cand['FloodingFlow'] = new_flood
                        cand['FloodingVolume'] = new_flood
                        # Recalculate TankVolume (clamped to min/max)
                        cand['TankVolume'] = max(MIN_TANK_VOLUME, min(new_flood, MAX_TANK_VOLUME))
                    
                    print(f"  [Re-Rank] Updated {updated_count} candidate flooding values")
                    
                    # Show top 3 candidates after re-rank for debugging
                    remaining_candidates.sort(key=lambda x: x['FloodingFlow'], reverse=True)
                    print(f"  [Re-Rank] Top candidates after update:")
                    for i, c in enumerate(remaining_candidates[:5]):
                        if c['NodeID'] not in used_nodes:
                            print(f"    #{i+1}: {c['NodeID']} -> FloodVol: {c['FloodingVolume']:.0f} m³, TankVol: {c['TankVolume']:.0f} m³")

                    
                    # === PRUNING: REMOVE TANKS BASED ON UTILIZATION AND SYSTEMIC IMPACT ===
                    # Only prune if: (1) stored_volume < MIN, AND (2) flood reduction is minimal
                    tanks_to_remove = []
                    
                    # Get current flood reduction for systemic impact check
                    current_flood_reduction = baseline_flood - flooding_vol
                    
                    if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics and self.evaluator.last_metrics.tank_utilization:
                        util_map = self.evaluator.last_metrics.tank_utilization
                        
                        for ac in active_candidates:
                            # Tank name format: TK_{node}_{predio}
                            tk_name = f"TK_{ac['NodeID']}_{ac['PredioID']}"
                            
                            if tk_name in util_map:
                                max_depth = util_map[tk_name]['max_depth']
                                # Calculate stored volume approx: (Vol / Depth) * UsedDepth
                                used_vol = (ac['TankVolume'] / TANK_DEPTH) * max_depth
                                utilization_pct = (used_vol / ac['TankVolume']) * 100 if ac['TankVolume'] > 0 else 0
                                
                                # New logic: Only prune if utilization is very low (< 20%)
                                # Tanks with low stored_volume but decent utilization may have systemic impact
                                if used_vol < MIN_TANK_VOLUME and utilization_pct < 20:
                                    tanks_to_remove.append(ac)
                                    # Increment failure count
                                    node_id = ac['NodeID']
                                    pruning_failures[node_id] = pruning_failures.get(node_id, 0) + 1
                                    retries_left = MAX_PRUNE_RETRIES - pruning_failures[node_id]
                                    print(f"  [Prune] {tk_name}: Stored {used_vol:.0f} m³ ({utilization_pct:.1f}%) - LOW IMPACT (Retries left: {retries_left})")
                                elif used_vol < MIN_TANK_VOLUME:
                                    # Low stored but good utilization % - keep it (systemic impact likely)
                                    print(f"  [Keep] {tk_name}: Stored {used_vol:.0f} m³ but {utilization_pct:.1f}% utilization - possible systemic impact")

                    
                    for ac in tanks_to_remove:
                        node_id = ac['NodeID']
                        predio_idx = ac['PredioID']
                        # Restore area to predio
                        tank_area = (ac['TankVolume'] / TANK_DEPTH) * OCCUPATION_FACTOR
                        predio_capacity[predio_idx]['used_area'] -= tank_area
                        predio_capacity[predio_idx]['n_tanks'] -= 1
                        # Remove from used_nodes so it could be reconsidered later
                        used_nodes.discard(node_id)
                        # Remove from active
                        active_candidates.remove(ac)
            else:
                print("  Dynamic Evaluator not provided. Skipping evaluation.")

        # === GENERATE ECONOMIC GRAPH ===
        if economic_history:
            self._plot_economic_curve(economic_history)
        
        # === GENERATE SUMMARY TEXT FILE ===
        if active_candidates and self.evaluator:
            self._generate_summary_report(active_candidates, economic_history, predio_capacity)
        
        # === EXPORT SELECTED PREDIOS AS GPKG ===
        if active_candidates:
            self._export_selected_predios_gpkg(active_candidates, predio_capacity)
        
        # === GENERATE VISUAL RESULTS DASHBOARD ===
        if active_candidates and economic_history:
            self._generate_results_dashboard(active_candidates, economic_history, predio_capacity)
                
        return pd.DataFrame(results)
    
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
            f"Costo Potencial:    ${residual_cost/1e6:.2f}M\n"
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
                current_flood = self.evaluator.last_metrics.total_flooding_volume if self.evaluator.last_metrics else baseline_flood
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
                tk_name = f"TK_{ac['NodeID']}_{ac['PredioID']}"
                predio = ac['PredioID']
                design_vol = ac['TankVolume']
                
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
            pid = ac['PredioID']
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
            lambda x: sum(t['TankVolume'] for t in predio_tanks.get(x, []))
        )
        
        # Add utilization data if available
        util_map = {}
        if hasattr(self.evaluator, 'last_metrics') and self.evaluator.last_metrics:
            util_map = self.evaluator.last_metrics.tank_utilization or {}
        
        def get_stored_vol(pid):
            tanks = predio_tanks.get(pid, [])
            total = 0
            for t in tanks:
                tk_name = f"TK_{t['NodeID']}_{t['PredioID']}"
                if tk_name in util_map:
                    total += util_map[tk_name].get('stored_volume', 0)
            return total
        
        selected['total_stored_vol'] = selected.index.map(get_stored_vol)
        selected['utilization_pct'] = (selected['total_stored_vol'] / selected['total_design_vol'] * 100).fillna(0)
        
        # Add tank names
        selected['tank_names'] = selected.index.map(
            lambda x: ', '.join([f"TK_{t['NodeID']}_{t['PredioID']}" for t in predio_tanks.get(x, [])])
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
        total_design_vol = sum(ac['TankVolume'] for ac in active_candidates)
        n_predios = len(set(ac['PredioID'] for ac in active_candidates))
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
            tk_name = f"TK_{ac['NodeID']}_{ac['PredioID']}"
            design_vol = ac['TankVolume']
            stored = util_map.get(tk_name, {}).get('stored_volume', 0)
            util_pct = (stored / design_vol * 100) if design_vol > 0 else 0
            table_data.append([
                tk_name[:18], 
                str(ac['PredioID']), 
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


if __name__ == "__main__":

    
    path_proy = config.PROJECT_ROOT
    
    elev_file = r'gis\01_raster\elev.tif'
    elev_files_list = [str(path_proy / elev_file)]
    predios_path = path_proy / r'gis\00_vector\07_predios_disponibles.shp'
    flooding_path = path_proy / r"codigos\00_flooding_stats\00_flooding_nodes.gpkg"
    proj_to = "EPSG:32717"
    
    print("Loading data...")
    
    # Setup elevation getter (same as rut_10)
    source = ElevationSource(path_proy, proj_to)
    tree = source.get_elev_source(
        elev_files_list,
        check_unique_values=False,
        ellipsoidal2orthometric=False,
        m_ramales=None,
        elevation_shift=0
    )
    getter = ElevationGetter(tree=tree, m_ramales=None, threshold_distance=0.7)
    
    # Load predios (same as rut_10)
    predios_gdf = gpd.read_file(predios_path)
    predios_gdf['centroide'] = predios_gdf.geometry.centroid
    predios_gdf['z'] = getter.get_elevation_from_tree_coords(
        predios_gdf.geometry.centroid.get_coordinates().to_numpy()
    )
    predios_gdf['costo_suelo_m2'] = 50.0  # Default cost (to be updated with real data)
    predios_gdf['clasificacion_suelo'] = 2  # Default difficulty (to be updated)
    
    # Load flooding nodes (same as rut_10)
    flooding_gdf = gpd.read_file(flooding_path)
    flooding_gdf.to_crs(proj_to, inplace=True)
    flooding_gdf.index = flooding_gdf['NodeID']
    flooding_gdf['z'] = getter.get_elevation_from_tree_coords(
        flooding_gdf.geometry.centroid.get_coordinates().to_numpy()
    )
    
    # Ensure same CRS
    if predios_gdf.crs != flooding_gdf.crs:
        predios_gdf = predios_gdf.to_crs(flooding_gdf.crs)
    
    print(f"Loaded {len(predios_gdf)} predios, {len(flooding_gdf)} flooding nodes")
    print(f"Predios z range: {predios_gdf['z'].min():.2f} - {predios_gdf['z'].max():.2f}")
    print(f"Nodes z range: {flooding_gdf['z'].min():.2f} - {flooding_gdf['z'].max():.2f}")
    
    # Create optimizer
    optimizer = TankOptimizer(
        nodes_gdf=flooding_gdf,
        predios_gdf=predios_gdf,
        V_min=100.0,
        V_max=5000.0,
        max_tanks=5
    )
    
    # Run optimization
    print("\nRunning test optimization (10 generations, 30 individuals)...")
    result = optimizer.run(n_gen=10, pop_size=30, seed=42)
    
    if result is not None:
        optimizer.print_pareto_front()
        
        # Create output folder for plots
        output_folder = path_proy / "codigos" / "00_optimization_results"
        output_folder.mkdir(exist_ok=True)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        # 1. Pareto front
        optimizer.plot_pareto_front(save_path=str(output_folder / "pareto_front.png"))
        
        # 2. Dashboard with best solutions
        optimizer.plot_dashboard(save_path=str(output_folder / "dashboard.png"))
        
        # 3. Individual solution maps
        df = optimizer.get_pareto_solutions()
        for sol_id in df['solution_id'].head(3):  # Plot top 3 solutions
            optimizer.plot_solution_map(
                sol_id, 
                save_path=str(output_folder / f"solution_{sol_id}_map.png")
            )
        
        print(f"\nAll plots saved to: {output_folder}")
        
        # Show plots interactively
        import matplotlib.pyplot as plt
        plt.show()
