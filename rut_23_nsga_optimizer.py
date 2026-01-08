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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# pymoo imports
try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not installed. Run: pip install pymoo")

import config
config.setup_sys_path()

from rut_16_dynamic_evaluator import DynamicSolutionEvaluator
from rut_20_avoided_costs import AvoidedCostRunner


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
                 **kwargs):
        
        n_candidates = len(valid_pairs)
        
        super().__init__(
            n_var=n_candidates,      # Number of decision variables
            n_obj=2,                 # Two objectives: cost and damage
            n_ieq_constr=2,          # Constraints: max_tanks + predio capacity
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
        self.mode = 'probabilistic' if len(optimization_tr_list) > 1 else 'deterministic'
        
        # Tank depth for area calculation (area = volume / depth)
        self.tank_depth = 5.0  # meters
        
        # Cache for evaluated solutions
        self._cache = {}
        self._eval_count = 0
        
    def _x_to_assignments(self, x: np.ndarray) -> List[Tuple[int, float]]:
        """Convert binary decision vector to tank assignments."""
        assignments = []
        for i, active in enumerate(x):
            if active > 0.5:  # Binary threshold
                row = self.valid_pairs.iloc[i]
                node_idx = i  # Use index as node_idx
                volume = row.get('tank_volume', row.get('FloodingVolume', 10000))
                assignments.append((node_idx, volume))
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
                
                # Tank area = volume / depth (approximate)
                volume = row.get('tank_volume', row.get('FloodingVolume', 10000))
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
        
        Args:
            x: (pop_size, n_var) binary matrix
            out: Dictionary to store results
        """
        pop_size = x.shape[0]
        
        # Objectives
        f1 = np.zeros(pop_size)  # Construction cost
        f2 = np.zeros(pop_size)  # Damage
        
        # Constraints
        g1 = np.zeros(pop_size)  # Max tanks constraint
        g2 = np.zeros(pop_size)  # Predio capacity constraint
        
        for i in range(pop_size):
            xi = x[i]
            n_tanks = np.sum(xi > 0.5)
            
            # Constraint 1: number of tanks <= max_tanks
            g1[i] = n_tanks - self.max_tanks  # <= 0 to be feasible
            
            # Constraint 2: predio capacity
            g2[i] = self._check_predio_capacity(xi)  # <= 0 to be feasible
            
            # Skip evaluation if constraints heavily violated
            if n_tanks > self.max_tanks or n_tanks == 0 or g2[i] > 1000:
                f1[i] = 1e12
                f2[i] = 1e12
                continue
            
            # Convert to hashable key for caching
            x_key = tuple(xi > 0.5)
            if x_key in self._cache:
                f1[i], f2[i] = self._cache[x_key]
                continue
            
            # Convert to assignments
            assignments = self._x_to_assignments(xi)
            
            try:
                self._eval_count += 1
                print(f"\n  [NSGA-II] Evaluation {self._eval_count}: {len(assignments)} tanks")
                
                # Run evaluation
                cost, damage = self._evaluate_single(assignments)
                
                f1[i] = cost
                f2[i] = damage
                
                # Cache result
                self._cache[x_key] = (cost, damage)
                
            except Exception as e:
                print(f"  [NSGA-II] Evaluation failed: {e}")
                f1[i] = 1e12
                f2[i] = 1e12
        
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])
    
    def _evaluate_single(self, assignments: List[Tuple[int, float]]) -> Tuple[float, float]:
        """
        Evaluate a single tank configuration.
        
        Returns:
            (construction_cost, damage_cost)
        """
        if self.mode == 'deterministic':
            # Single TR evaluation
            tr = self.optimization_tr_list[0]
            cost, flooding = self.evaluator.evaluate_solution(
                assignments=assignments,
                run_swmm=True,
                solution_name=f"NSGA_eval_{self._eval_count}"
            )
            
            # Get damage from evaluator
            damage = getattr(self.evaluator, 'last_flood_damage_usd', flooding * 100)
            
            # Add deferred investment if enabled
            if self.cost_components.get('deferred_investment', True):
                deferred = getattr(self.evaluator, 'last_deferred_cost', 0)
                damage += deferred
            
            return cost, damage
            
        else:
            # Probabilistic: multiple TRs -> EAD
            # This is slower but more accurate
            total_ead = 0
            
            for tr in self.optimization_tr_list:
                cost, flooding = self.evaluator.evaluate_solution(
                    assignments=assignments,
                    run_swmm=True,
                    solution_name=f"NSGA_TR{tr}_eval_{self._eval_count}"
                )
                
                damage = getattr(self.evaluator, 'last_flood_damage_usd', flooding * 100)
                if self.cost_components.get('deferred_investment', True):
                    damage += getattr(self.evaluator, 'last_deferred_cost', 0)
                
                # Weight by probability
                prob = 1.0 / tr
                total_ead += damage * prob
            
            return cost, total_ead


class NSGATankOptimizer:
    """
    Multi-objective tank optimizer using NSGA-II.
    
    Follows academic methodology: SWMM + NSGA-II → Pareto Front
    """
    
    def __init__(self,
                 evaluator: DynamicSolutionEvaluator,
                 valid_pairs: pd.DataFrame,
                 max_tanks: int = 20,
                 min_tank_vol: float = 1000,
                 max_tank_vol: float = 100000,
                 cost_components: Dict[str, bool] = None):
        """
        Args:
            evaluator: DynamicSolutionEvaluator from rut_16
            valid_pairs: DataFrame of candidate tank locations
            max_tanks: Maximum number of tanks to place
            min_tank_vol: Minimum tank volume (m³)
            max_tank_vol: Maximum tank volume (m³)
            cost_components: Dict controlling which costs to calculate
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo not installed. Run: pip install pymoo")
        
        self.evaluator = evaluator
        self.valid_pairs = valid_pairs
        self.max_tanks = max_tanks
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
        print(f"  Max Tanks:  {max_tanks}")
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
            max_tanks=self.max_tanks
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
        print(f"\n  Starting NSGA-II optimization...")
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
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


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NSGA-II Tank Optimizer - Test Run")
    print("="*60)
    
    if not PYMOO_AVAILABLE:
        print("\nError: pymoo not installed. Run:")
        print("  pip install pymoo")
        exit(1)
    
    print("\nThis module should be called from rut_10_run_tanque_tormenta.py")
    print("or used interactively after setting up the evaluator.")
