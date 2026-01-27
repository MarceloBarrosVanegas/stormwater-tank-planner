"""
NSGA-II Multi-Objective Tank Optimizer
======================================

Academic-standard multi-objective optimization for stormwater tank placement
using NSGA-II algorithm from pymoo library.

Reference: Standard methodology from academic papers coupling SWMM + NSGA-II
for detention tank cost-benefit optimization.

Dependencies:
    pip install pymoo
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import config
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination



import config
config.setup_sys_path()

from rut_16_dynamic_evaluator import DynamicSolutionEvaluator



@dataclass
class ParetoSolution:
    """Represents a single solution on the Pareto front."""
    tank_indices: List[int]      # Which candidate tanks are active
    tank_assignments: List[Tuple[int, float]]  # (node_idx, volume) pairs
    construction_cost: float     # Total construction cost
    damage_cost: float           # Total damage (what we want to minimize)
    bc_ratio: float              # Benefit/Cost ratio
    n_tanks: int                 # Number of tanks


@dataclass
class ParetoResult:
    """Results from NSGA-II optimization."""
    solutions: List[ParetoSolution]
    best_bc_solution: ParetoSolution  # Solution with best B/C ratio
    pareto_front: np.ndarray         # (n_solutions, 2) array of objectives
    optimization_tr_list: List[int]
    mode: str  # 'deterministic' or 'probabilistic'


class TankOptimizationProblem(Problem):
    """
    pymoo Problem definition for tank optimization.
    
    Variables: Binary array [0,1] for each candidate tank location
    Objectives:
        1. Minimize: Total construction cost
        2. Minimize: Residual flood damage (or EAD if probabilistic)
    Constraints:
        - Maximum number of tanks
    """
    
    def __init__(self, 
                 evaluator: DynamicSolutionEvaluator,
                 valid_pairs: pd.DataFrame,
                 optimization_tr_list: List[int],
                 cost_components: Dict[str, bool],
                 max_tanks: int = 20,
                 min_tanks: int = 3,
                 **kwargs):
        
        n_candidates = len(valid_pairs)
        
        super().__init__(
            n_var=n_candidates,      # Number of decision variables
            n_obj=3,                 # Three objectives: flooding, construction cost, repair cost
            n_ieq_constr=3,          # Constraints: max_tanks, min_tanks, predio capacity
            xl=0,                    # Lower bound (binary: 0)
            xu=1,                    # Upper bound (binary: 1)
            vtype=bool,              # Variable type
            **kwargs
        )
        
        self.evaluator = evaluator
        self.valid_pairs = valid_pairs.reset_index(drop=True)
        self.optimization_tr_list = optimization_tr_list
        self.cost_components = cost_components
        self.max_tanks = max_tanks
        self.min_tanks = min_tanks  # Minimum number of active tanks
        self.mode = 'probabilistic' if len(optimization_tr_list) > 1 else 'deterministic'
        
        # Tank depth for area calculation (area = volume / depth)
        self.tank_depth = 5.0  # meters
        
        # Cache for evaluated solutions
        self._cache = {}
        self._eval_count = 0
        
    def _x_to_assignments(self, x: np.ndarray) -> List[Tuple[int, float]]:
        """
        Convert binary decision vector to tank assignments.
        
        Returns format expected by rut_16._get_active_pairs:
        List indexed by node position where each element is (predio_1idx, volume).
        predio_1idx = 0 means no tank, predio_1idx = 1+ is the 1-indexed predio ID.
        """
        # Get number of nodes from evaluator
        n_nodes = len(self.evaluator.nodes_gdf)
        
        # Initialize all nodes with no tank (predio=0, volume=0)
        assignments = [(0, 0) for _ in range(n_nodes)]
        
        for i, active in enumerate(x):
            if active > 0.5:  # Binary threshold
                row = self.valid_pairs.iloc[i]
                
                # Get the original node index in nodes_gdf
                node_idx = row.get('node_idx', row.get('NodeIdx', i))
                
                # Get predio ID (1-indexed for rut_16 format)
                predio_id = row.get('PredioID', row.get('predio_idx', 0))
                predio_1idx = int(predio_id) + 1  # Convert to 1-indexed
                
                # Calculate tank volume based on flooding volume
                # FloodingVolume = actual flood volume at node
                # Apply safety factor of 1.2 to account for uncertainty
                flooding_vol = row.get('FloodingVolume', 0)
                # Tank should capture flooding volume + safety factor
                volume = flooding_vol * config.TANK_VOLUME_SAFETY_FACTOR

                
                # Clamp to config limits

                volume = max(config.TANK_MIN_VOLUME_M3, min(volume, config.TANK_MAX_VOLUME_M3))
                
                # Only assign if valid node_idx
                if 0 <= node_idx < n_nodes:
                    assignments[node_idx] = (predio_1idx, volume)
        
        return assignments
    
    def _check_predio_capacity(self, x: np.ndarray) -> float:
        """
        Check if tanks on same predio exceed available area.
        Returns constraint violation (> 0 means infeasible).
        """
        # Group active tanks by PredioID
        predio_usage = {}  # {predio_id: (total_tank_area, predio_area)}
        
        for i, active in enumerate(x):
            if active > 0.5:
                row = self.valid_pairs.iloc[i]
                predio_id = row.get('PredioID', i)
                predio_area = row.get('PredioArea', 100000)  # Default large if missing
                
                # Calculate tank volume same as in _x_to_assignments
                flooding_vol = row.get('FloodingVolume', 0)
                if flooding_vol > 0:
                    volume = flooding_vol * config.TANK_VOLUME_SAFETY_FACTOR
                else:
                    flooding_flow = row.get('FloodingFlow', 0)
                    volume = flooding_flow * 2.0 * 3600
                
                volume = max(config.TANK_MIN_VOLUME_M3, min(volume, config.TANK_MAX_VOLUME_M3))
                tank_area = volume / self.tank_depth
                
                if predio_id not in predio_usage:
                    predio_usage[predio_id] = {'tank_area': 0, 'predio_area': predio_area}
                
                predio_usage[predio_id]['tank_area'] += tank_area
        
        # Calculate max violation
        max_violation = 0.0
        for predio_id, usage in predio_usage.items():
            excess = usage['tank_area'] - usage['predio_area']
            if excess > max_violation:
                max_violation = excess
        
        return max_violation
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate a population of solutions.
        
        Objectives (all minimized):
            f1: Remaining flooding volume (m3)
            f2: Construction cost ($)
            f3: Solution repair cost ($) - minimizing this maximizes avoided cost
        
        Constraints:
            g1: n_tanks <= max_tanks
            g2: n_tanks >= min_tanks
            g3: predio capacity
        """
        pop_size = x.shape[0]
        
        # Objectives
        f1 = np.zeros(pop_size)  # Remaining flooding volume
        f2 = np.zeros(pop_size)  # Construction cost
        f3 = np.zeros(pop_size)  # Solution repair cost
        
        # Constraints
        g1 = np.zeros(pop_size)  # Max tanks constraint
        g2 = np.zeros(pop_size)  # Min tanks constraint  
        g3 = np.zeros(pop_size)  # Predio capacity constraint
        
        for i in range(pop_size):
            xi = x[i]
            n_tanks = np.sum(xi > 0.5)
            
            # Constraint 1: number of tanks <= max_tanks
            g1[i] = n_tanks - self.max_tanks  # <= 0 to be feasible
            
            # Constraint 2: number of tanks >= min_tanks
            g2[i] = self.min_tanks - n_tanks  # <= 0 to be feasible (min_tanks - n <= 0 means n >= min)
            
            # Constraint 3: predio capacity
            g3[i] = self._check_predio_capacity(xi)  # <= 0 to be feasible
            
            # Skip evaluation if constraints heavily violated or no tanks
            if n_tanks > self.max_tanks or n_tanks < self.min_tanks or n_tanks == 0 or g3[i] > 1000:
                f1[i] = 1e12
                f2[i] = 1e12
                f3[i] = 1e12
                continue
            
            # Convert to hashable key for caching
            x_key = tuple(xi > 0.5)
            if x_key in self._cache:
                f1[i], f2[i], f3[i] = self._cache[x_key]
                continue
            
            # Convert to assignments
            assignments = self._x_to_assignments(xi)
            
            try:
                self._eval_count += 1
                print(f"\n  [NSGA-II] Evaluation {self._eval_count}: {len(assignments)} tanks")
                
                # Run evaluation - returns (construction_cost, flooding_volume, repair_cost)
                constr_cost, flooding, repair_cost = self._evaluate_single(assignments)
                
                f1[i] = flooding       # Minimize remaining flooding
                f2[i] = constr_cost    # Minimize construction cost
                f3[i] = repair_cost    # Minimize repair cost (= maximize avoided cost)
                
                # Cache result
                self._cache[x_key] = (flooding, constr_cost, repair_cost)
                
            except Exception as e:
                import traceback
                print(f"\n  [FATAL ERROR] NSGA-II Evaluation failed!")
                print(f"  Error: {e}")
                traceback.print_exc()
                raise RuntimeError(f"NSGA-II Evaluation {self._eval_count} failed: {e}") from e
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2, g3])
    
    def _evaluate_single(self, assignments: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """
        Evaluate a single tank configuration.
        
        Returns:
            (construction_cost, flooding_volume, repair_cost)
        """
        if self.mode == 'deterministic':
            # Single TR evaluation
            tr = self.optimization_tr_list[0]
            cost, flooding = self.evaluator.evaluate_solution(
                assignments=assignments,
                run_swmm=True,
                solution_name=f"NSGA_eval_{self._eval_count}"
            )
            
            # Get repair cost from evaluator (set by DeferredInvestmentCost)
            repair_cost = getattr(self.evaluator, 'last_solution_infra_cost', 1e12)
            
            return cost, flooding, repair_cost
            
        else:
            # Probabilistic: multiple TRs -> EAD
            total_flooding = 0
            total_repair = 0
            
            for tr in self.optimization_tr_list:
                cost, flooding = self.evaluator.evaluate_solution(
                    assignments=assignments,
                    run_swmm=True,
                    solution_name=f"NSGA_TR{tr}_eval_{self._eval_count}"
                )
                
                repair_cost = getattr(self.evaluator, 'last_solution_infra_cost', 1e12)
                
                # Weight by probability for EAD
                prob = 1.0 / tr
                total_flooding += flooding * prob
                total_repair += repair_cost * prob
            
            return cost, total_flooding, total_repair


class NSGATankOptimizer:
    """
    Multi-objective tank optimizer using NSGA-II.
    
    Follows academic methodology: SWMM + NSGA-II → Pareto Front
    """
    
    def __init__(self,
                 evaluator: DynamicSolutionEvaluator,
                 valid_pairs: pd.DataFrame,
                 max_tanks: int = 20,
                 min_tanks: int = 3,
                 min_tank_vol: float = 1000,
                 max_tank_vol: float = 100000,
                 cost_components: Dict[str, bool] = None):
        """
        Args:
            evaluator: DynamicSolutionEvaluator from rut_16
            valid_pairs: DataFrame of candidate tank locations
            max_tanks: Maximum number of tanks to place
            min_tanks: Minimum number of tanks (constraint)
            min_tank_vol: Minimum tank volume (m³)
            max_tank_vol: Maximum tank volume (m³)
            cost_components: Dict controlling which costs to calculate
        """

        
        self.evaluator = evaluator
        self.valid_pairs = valid_pairs
        self.max_tanks = max_tanks
        self.min_tanks = min_tanks
        self.min_tank_vol = min_tank_vol
        self.max_tank_vol = max_tank_vol
        
        self.cost_components = cost_components or {
            'deferred_investment': True,
            'flood_damage': True,
            'river_damage': False
        }
        
        self.baseline_damage = getattr(evaluator, 'baseline_ead_total', None)
        
        print(f"\n{'='*60}")
        print("  NSGA-II MULTI-OBJECTIVE TANK OPTIMIZER")
        print(f"{'='*60}")
        print(f"  Candidates: {len(valid_pairs)}")
        print(f"  Tanks: {min_tanks} - {max_tanks}")
        print(f"  Cost Components: {self.cost_components}")
    
    def run(self,
            optimization_tr_list: List[int],
            validation_tr_list: List[int] = None,
            n_generations: int = 50,
            pop_size: int = 30,
            output_dir: Path = None) -> ParetoResult:
        """
        Run NSGA-II optimization.
        
        Args:
            optimization_tr_list: [25] for deterministic, [1,2,5,10,25] for probabilistic
            validation_tr_list: Optional TRs for final validation of selected solution
            n_generations: Number of NSGA-II generations
            pop_size: Population size per generation
            output_dir: Directory for outputs
            
        Returns:
            ParetoResult with all Pareto-optimal solutions
        """
        mode = 'probabilistic' if len(optimization_tr_list) > 1 else 'deterministic'
        
        print(f"\n  Mode: {mode.upper()}")
        print(f"  Optimization TRs: {optimization_tr_list}")
        print(f"  Generations: {n_generations}")
        print(f"  Population: {pop_size}")
        print(f"  Est. Evaluations: ~{n_generations * pop_size // 2}")
        
        # Create problem
        problem = TankOptimizationProblem(
            evaluator=self.evaluator,
            valid_pairs=self.valid_pairs,
            optimization_tr_list=optimization_tr_list,
            cost_components=self.cost_components,
            max_tanks=self.max_tanks,
            min_tanks=self.min_tanks
        )
        
        # Create algorithm
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )
        
        # Termination
        termination = get_termination("n_gen", n_generations)
        
        # Run optimization
        import time
        random_seed = int(time.time()) % 100000  # Random seed based on current time
        # random_seed = 42
        print(f"\n  Starting NSGA-II optimization (seed={random_seed})...")
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=random_seed,
            verbose=True
        )
        
        print(f"\n  Optimization complete!")
        print(f"  Total evaluations: {problem._eval_count}")
        print(f"  Pareto solutions: {len(result.F)}")
        
        # Convert results to ParetoSolution objects
        solutions = self._convert_results(result, problem)
        
        # Find best B/C solution
        best_bc = max(solutions, key=lambda s: s.bc_ratio if s.bc_ratio != float('inf') else 0)
        
        # Create result object
        pareto_result = ParetoResult(
            solutions=solutions,
            best_bc_solution=best_bc,
            pareto_front=result.F,
            optimization_tr_list=optimization_tr_list,
            mode=mode
        )
        
        # Generate plots
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.plot_pareto(pareto_result, output_dir / "pareto_front.png")
            self.export_results(pareto_result, output_dir / "pareto_solutions.csv")
        
        # Validation if requested
        if validation_tr_list and validation_tr_list != optimization_tr_list:
            print(f"\n  Running validation with TRs: {validation_tr_list}")
            self._validate_solution(best_bc, validation_tr_list, output_dir)
        
        return pareto_result
    
    def _convert_results(self, result, problem) -> List[ParetoSolution]:
        """Convert pymoo result to ParetoSolution objects."""
        solutions = []
        
        for i, (x, f) in enumerate(zip(result.X, result.F)):
            assignments = problem._x_to_assignments(x)
            n_tanks = len(assignments)
            
            construction_cost = f[0]
            damage_cost = f[1]
            
            # Calculate B/C ratio
            if self.baseline_damage and construction_cost > 0:
                avoided_damage = self.baseline_damage - damage_cost
                bc_ratio = avoided_damage / construction_cost
            else:
                bc_ratio = 0.0
            
            solutions.append(ParetoSolution(
                tank_indices=[a[0] for a in assignments],
                tank_assignments=assignments,
                construction_cost=construction_cost,
                damage_cost=damage_cost,
                bc_ratio=bc_ratio,
                n_tanks=n_tanks
            ))
        
        return solutions
    
    def plot_pareto(self, result: ParetoResult, save_path: Path):
        """Generate Pareto front visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot all solutions
        costs = [s.construction_cost / 1e6 for s in result.solutions]
        damages = [s.damage_cost / 1e6 for s in result.solutions]
        n_tanks = [s.n_tanks for s in result.solutions]
        
        scatter = ax.scatter(costs, damages, c=n_tanks, cmap='viridis', 
                            s=100, edgecolors='black', linewidths=1)
        
        # Highlight best B/C solution
        best = result.best_bc_solution
        ax.scatter(best.construction_cost / 1e6, best.damage_cost / 1e6,
                  c='red', s=200, marker='*', edgecolors='black', linewidths=2,
                  label=f'Best B/C = {best.bc_ratio:.2f}')
        
        # Labels
        ax.set_xlabel('Construction Cost ($M)', fontsize=12)
        ax.set_ylabel('Damage/EAD Cost ($M)', fontsize=12)
        ax.set_title(f'Pareto Front - NSGA-II Optimization ({result.mode.upper()})', 
                    fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Tanks', fontsize=10)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    
    def export_results(self, result: ParetoResult, save_path: Path):
        """Export Pareto solutions to CSV."""
        data = []
        for i, s in enumerate(result.solutions):
            data.append({
                'solution_id': i,
                'n_tanks': s.n_tanks,
                'construction_cost_usd': s.construction_cost,
                'damage_cost_usd': s.damage_cost,
                'bc_ratio': s.bc_ratio,
                'tank_indices': str(s.tank_indices)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"  Saved: {save_path}")
    
    def _validate_solution(self, solution: ParetoSolution, 
                           tr_list: List[int], output_dir: Path):
        """Run full probabilistic validation on selected solution."""
        print(f"\n  Validating solution with {len(tr_list)} TRs...")
        
        # Use AvoidedCostRunner for full EAD calculation
        # This would run the selected configuration through all TRs
        # and generate complete risk analysis
        
        # TODO: Implement full validation with AvoidedCostRunner
        print("  [TODO] Full probabilistic validation not yet implemented")


