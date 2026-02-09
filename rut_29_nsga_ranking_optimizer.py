# rut_29_nsga_ranking_optimizer.py

"""
NSGA-II Ranking Optimizer for Stormwater Tank Placement
========================================================

Optimizes FLOODING_RANKING_WEIGHTS and CAPACITY_MAX_HD using NSGA-II.
Each evaluation runs the complete GreedyTankOptimizer workflow.

Author: Assistant
Date: 2024
"""

from __future__ import annotations
import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.visualization.scatter import Scatter
from pymoo.util.display.multi import MultiObjectiveOutput

# Configuración del proyecto
import config
config.setup_sys_path()

# Tus módulos (ajustar imports según tu estructura)
from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
from rut_27_model_metrics import MetricExtractor, SystemMetrics

import config
config.setup_sys_path()


# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

NSGA_CONFIG = {
    'n_generations': 100,
    'pop_size': 50,
    'seed': 42,
    'checkpoint_dir': Path("optimization_results") / "nsga_ranking_checkpoints",
    'checkpoint_freq': 1,  # cada cuántas generaciones guardar
    'results_dir': Path("optimization_results") / "nsga_ranking_results",
    'max_tanks': config.MAX_TANKS,
    'max_iterations': config.MAX_ITERATIONS,
    'capacity_max_hd_bounds': (0.0, 0.95),
}

WEIGHT_KEYS = [
    'flow_over_capacity',
    'flow_node_flooding',
    'vol_over_capacity', 
    'vol_node_flooding',
    'outfall_peak_flow',
    'failure_probability'
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SolutionResult:
    """
    Resultado de una evaluación greedy completa.
    Esta es la interfaz entre tu código y el optimizador NSGA.
    """
    # Reducciones vs baseline (positivo = mejor)
    flooding_vol_reduction: float       # m³ reducidos
    flooding_vol_reduction_pct: float   # % reducido (0-1)
    flooding_peak_flow_reduction: float     # m³/s reducidos
    flooding_peak_flow_reduction_pct: float # % reducido (0-1)
    outfall_peak_flow_reduction: float      # m³/s reducidos  
    outfall_peak_flow_reduction_pct: float  # % reducido (0-1)
    
    # Estado de la red
    network_health: float               # 0-1, mayor es mejor
    network_health_pct: float           # % mejora vs baseline
    
    # Costos
    total_cost: float                   # $ inversión
    cost_per_flooding_reduction: float  # $/(m³ reducido)
    
    # Metadata
    n_tanks: int
    n_iterations: int
    runtime_seconds: float
    
    # Objetos completos (opcional, para análisis profundo)
    baseline_metrics: Optional[SystemMetrics] = None
    solution_metrics: Optional[SystemMetrics] = None
    detailed_results: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        result = asdict(self)
        # Eliminar objetos no serializables
        result.pop('baseline_metrics', None)
        result.pop('solution_metrics', None)
        result.pop('detailed_results', None)
        return result
    
    def get_objectives_array(self) -> np.ndarray:
        """
        Devuelve array [f0, f1, f2, f3, f4] para pymoo.
        Todos negativos porque pymoo minimiza.
        """
        return np.array([
            -self.flooding_vol_reduction_pct,      # maximizar → minimizar negativo
            -self.flooding_peak_flow_reduction_pct, # maximizar → minimizar negativo
            -self.outfall_peak_flow_reduction_pct,  # maximizar → minimizar negativo
            -self.network_health,                   # maximizar → minimizar negativo
            self.total_cost,                        # minimizar directo
        ])


@dataclass
class RankingWeights:
    """Contenedor para los pesos de ranking."""
    flow_over_capacity: float = 0.0
    flow_node_flooding: float = 0.0
    vol_over_capacity: float = 0.0
    vol_node_flooding: float = 0.0
    outfall_peak_flow: float = 0.0
    failure_probability: float = 0.0
    capacity_max_hd: float = 0.0  # independiente, no suma con los demás
    
    def to_array(self) -> np.ndarray:
        """Convierte a array [w1, w2, w3, w4, w5, w6, h_d]."""
        return np.array([
            self.flow_over_capacity,
            self.flow_node_flooding,
            self.vol_over_capacity,
            self.vol_node_flooding,
            self.outfall_peak_flow,
            self.failure_probability,
            self.capacity_max_hd,
        ])
    
    @classmethod
    def from_array(cls, x: np.ndarray) -> "RankingWeights":
        """Crea desde array de NSGA."""
        # Normalizar pesos (primeros 6) a suma=1
        weights_raw = x[:6]
        weights_sum = np.sum(weights_raw)
        if weights_sum < 1e-10:
            weights = np.ones(6) / 6
        else:
            weights = weights_raw / weights_sum
            
        return cls(
            flow_over_capacity=weights[0],
            flow_node_flooding=weights[1],
            vol_over_capacity=weights[2],
            vol_node_flooding=weights[3],
            outfall_peak_flow=weights[4],
            failure_probability=weights[5],
            capacity_max_hd=np.clip(x[6], 0, 0.95),
        )
    
    def apply_to_config(self):
        """Aplica estos pesos a config.py global."""
        config.FLOODING_RANKING_WEIGHTS = {
            'flow_over_capacity': self.flow_over_capacity,
            'flow_node_flooding': self.flow_node_flooding,
            'vol_over_capacity': self.vol_over_capacity,
            'vol_node_flooding': self.vol_node_flooding,
            'outfall_peak_flow': self.outfall_peak_flow,
            'failure_probability': self.failure_probability,
        }
        config.CAPACITY_MAX_HD = self.capacity_max_hd


# =============================================================================
# CLASES PRINCIPALES
# =============================================================================

class GreedyRunner:
    """
    Wrapper para ejecutar StormwaterOptimizationRunner con configuración temporal.
    Esta clase se encarga de restaurar config después de cada corrida.
    """
    
    def __init__(self, 
                 project_root: Optional[Path] = None,
                 elev_file: Optional[Path] = None):
        self.project_root = project_root or config.PROJECT_ROOT
        self.elev_file = elev_file or config.ELEV_FILE
        self._original_config = self._save_config_state()
        
    def _save_config_state(self) -> Dict:
        """Guarda estado original de config para restaurar después."""
        return {
            'FLOODING_RANKING_WEIGHTS': config.FLOODING_RANKING_WEIGHTS.copy(),
            'CAPACITY_MAX_HD': config.CAPACITY_MAX_HD,
        }
    
    def _restore_config(self):
        """Restaura config a estado original."""
        config.FLOODING_RANKING_WEIGHTS = self._original_config['FLOODING_RANKING_WEIGHTS']
        config.CAPACITY_MAX_HD = self._original_config['CAPACITY_MAX_HD']

    def run_single_evaluation(self,
                              weights: RankingWeights,
                              max_tanks: int = 30,
                              max_iterations: int = 100,
                              min_tank_vol: float = None,
                              max_tank_vol: float = None,
                              stop_at_breakeven: bool = True,
                              breakeven_multiplier: float = 50.0,
                              optimizer_mode: str = 'greedy'):
        """Ejecuta greedy con pesos dados."""
        import time
        start_time = time.time()

        weights.apply_to_config()

        # Usar defaults de config si no se especifica
        if min_tank_vol is None:
            min_tank_vol = config.TANK_MIN_VOLUME_M3
        if max_tank_vol is None:
            max_tank_vol = config.TANK_MAX_VOLUME_M3

        try:
            runner = StormwaterOptimizationRunner(
                project_root=self.project_root,
                proj_to=config.PROJECT_CRS
            )

            result = runner.run_sequential_analysis(
                max_tanks=max_tanks,
                max_iterations=max_iterations,
                min_tank_vol=min_tank_vol,
                max_tank_vol=max_tank_vol,
                stop_at_breakeven=stop_at_breakeven,
                breakeven_multiplier=breakeven_multiplier,
                elev_file=self.elev_file,
                optimizer_mode=optimizer_mode,
                optimization_tr_list=config.TR_LIST,
                validation_tr_list=config.VALIDATION_TR_LIST,
            )

            return result

        finally:
            self._restore_config()

class RankingOptimizationProblem(Problem):
    """
    Definición del problema para pymoo.
    Variables: 7 (6 pesos normalizados + capacity_max_hd)
    Objetivos: 5 (4 maximizaciones negadas + 1 minimización)
    """
    
    def __init__(self,
                 baseline_metrics: SystemMetrics,
                 runner_factory: Callable[[], GreedyRunner],
                 max_tanks: int = 30,
                 max_iterations: int = 100):
        
        self.baseline_metrics = baseline_metrics
        self.runner_factory = runner_factory
        self.max_tanks = max_tanks
        self.max_iterations = max_iterations
        
        # Contadores para logging
        self.eval_count = 0
        self.history: List[Tuple[RankingWeights, SolutionResult]] = []
        
        # Bounds: 6 pesos [0,1], 1 h/d [0, 0.95]
        xl = np.array([0.0] * 6 + [NSGA_CONFIG['capacity_max_hd_bounds'][0]])
        xu = np.array([1.0] * 6 + [NSGA_CONFIG['capacity_max_hd_bounds'][1]])
        
        super().__init__(
            n_var=7,
            n_obj=5,
            n_constr=0,  # La normalización maneja la restricción de suma=1
            xl=xl,
            xu=xu,
            elementwise=True,
        )
    
    def _evaluate(self, x: np.ndarray, out: Dict, *args, **kwargs):
        """Evaluación llamada por pymoo."""
        self.eval_count += 1
        
        print(f"\n{'='*80}")
        print(f"NSGA EVALUATION #{self.eval_count}")
        print(f"{'='*80}")
        
        # Convertir array a pesos
        weights = RankingWeights.from_array(x)
        weights.apply_to_config()

        # PRINT PARÁMETROS AQUÍ
        print(f"\nParameters for this evaluation:")
        print(f"  Weights: {weights}")
        print(f"  Max tanks: {self.max_tanks}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Config.CAPACITY_MAX_HD: {config.CAPACITY_MAX_HD}")
        print(f"  Config.FLOODING_RANKING_WEIGHTS: {config.FLOODING_RANKING_WEIGHTS}")
        print(f"{'=' * 80}\n")
        
        # Crear runner y ejecutar
        runner = self.runner_factory()
        result = runner.run_single_evaluation(
            weights=weights,
            max_tanks=self.max_tanks,
            max_iterations=self.max_iterations,
        )
        print(result)
        
        # Guardar en historial
        self.history.append((weights, result))
        
        # Mostrar resultados
        print(f"\nResults:")
        print(f"  Flooding vol reduction: {result.flooding_vol_reduction_pct:.1%}")
        print(f"  Flooding flow reduction: {result.flooding_peak_flow_reduction_pct:.1%}")
        print(f"  Outfall flow reduction: {result.outfall_peak_flow_reduction_pct:.1%}")
        print(f"  Network health: {result.network_health:.3f}")
        print(f"  Total cost: ${result.total_cost:,.0f}")
        print(f"{'='*80}\n")
        
        # Output para pymoo
        out["F"] = result.get_objectives_array()


class NSGACheckpoint(Callback):
    """Guarda progreso cada N generaciones."""
    
    def __init__(self, checkpoint_dir: Path, frequency: int = 1):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.history: List[Dict] = []
    
    def notify(self, algorithm):
        gen = algorithm.n_gen
        
        # Guardar historial
        self.history.append({
            'generation': gen,
            'population': algorithm.pop.get("X").copy(),
            'objectives': algorithm.pop.get("F").copy(),
        })
        
        # Guardar checkpoint
        if gen % self.frequency == 0:
            path = self.checkpoint_dir / f"checkpoint_gen_{gen:04d}.pkl"
            with open(path, 'wb') as f:
                pickle.dump({
                    'algorithm': algorithm,
                    'history': self.history,
                    'config': NSGA_CONFIG,
                }, f)
            print(f"[Checkpoint] Saved generation {gen} to {path}")


class NSGARankingOptimizer:
    """
    Optimizador principal NSGA-II para pesos de ranking.
    
    Uso:
        optimizer = NSGARankingOptimizer()
        optimizer.run(n_generations=100, pop_size=50)
        optimizer.visualize_results()
        best = optimizer.get_best_solution(preference='balanced')
    """

    def __init__(self,
                 project_root: Optional[Path] = None,
                 elev_file: Optional[Path] = None,
                 results_dir: Optional[Path] = None):

        self.project_root = project_root or config.PROJECT_ROOT
        self.elev_file = elev_file or config.ELEV_FILE
        self.results_dir = results_dir if results_dir is not None else NSGA_CONFIG['results_dir']
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_metrics: Optional[SystemMetrics] = None
        self.problem: Optional[RankingOptimizationProblem] = None
        self.algorithm: Optional[NSGA2] = None
        self.result = None
        
        # Factory para crear runners
        self._runner_factory = lambda: GreedyRunner(
            project_root=self.project_root,
            elev_file=self.elev_file,
        )
    
    def _run_baseline(self) -> SystemMetrics:
        """Corre baseline sin tanques."""
        print(f"\n{'='*80}")
        print("RUNNING BASELINE SIMULATION")
        print(f"{'='*80}")
        
        extractor = MetricExtractor(
            project_root=self.project_root,
            predios_path=config.PREDIOS_FILE
        )
        extractor.run(config.SWMM_FILE)
        
        metrics = extractor.metrics
        print(f"\nBaseline:")
        print(f"  Flooding volume: {metrics.total_flooding_volume:,.2f} m³")
        print(f"  Max outfall flow: {metrics.total_max_outfall_flow:.3f} m³/s")
        print(f"  Flooded nodes: {metrics.flooded_nodes_count}")
        print(f"{'='*80}\n")
        
        return metrics
    
    def setup(self):
        """Inicializa baseline y problema."""
        print("[NSGARankingOptimizer] Initializing...")
        
        self.baseline_metrics = self._run_baseline()
        
        self.problem = RankingOptimizationProblem(
            baseline_metrics=self.baseline_metrics,
            runner_factory=self._runner_factory,
            max_tanks=NSGA_CONFIG['max_tanks'],
            max_iterations=NSGA_CONFIG['max_iterations'],
        )
        
        self.algorithm = NSGA2(
            pop_size=NSGA_CONFIG['pop_size'],
            seed=NSGA_CONFIG['seed'],
            verbose=True,
        )
        
        print(f"  Variables: {self.problem.n_var}")
        print(f"  Objectives: {self.problem.n_obj}")
        print(f"  Population: {NSGA_CONFIG['pop_size']}")
        print(f"  Generations: {NSGA_CONFIG['n_generations']}")
    
    def run(self, 
           n_generations: Optional[int] = None,
           pop_size: Optional[int] = None,
           restore_from: Optional[Path] = None):
        """
        Corre la optimización NSGA-II.
        
        Args:
            n_generations: Si no None, override de NSGA_CONFIG
            pop_size: Si no None, override de NSGA_CONFIG
            restore_from: Path a checkpoint para continuar
        """
        if self.problem is None:
            self.setup()
        
        # Override config si se especifica
        n_gen = n_generations or NSGA_CONFIG['n_generations']
        pop = pop_size or NSGA_CONFIG['pop_size']
        
        if n_generations:
            self.algorithm.pop_size = pop
        
        print(f"\n{'='*80}")
        print("STARTING NSGA-II OPTIMIZATION")
        print(f"{'='*80}\n")
        
        # Checkpoint callback
        checkpoint = NSGACheckpoint(
            NSGA_CONFIG['checkpoint_dir'],
            NSGA_CONFIG['checkpoint_freq']
        )
        
        # Restaurar si se especifica
        if restore_from and restore_from.exists():
            print(f"[Restore] Loading from {restore_from}")
            with open(restore_from, 'rb') as f:
                ckpt = pickle.load(f)
                self.algorithm = ckpt['algorithm']
        
        # Correr optimización
        self.result = minimize(
            self.problem,
            self.algorithm,
            ('n_gen', n_gen),
            seed=NSGA_CONFIG['seed'],
            callback=checkpoint,
            verbose=True,
        )
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}\n")
        
        self._save_final_results()
    
    def _save_final_results(self):
        """Guarda todos los resultados."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Pareto front completo
        if self.result and self.result.F is not None:
            F = self.result.F.copy()
            # Convertir negativos a positivos para legibilidad
            F[:, :4] = -F[:, :4]
            
            pareto_df = pd.DataFrame(F, columns=[
                'flooding_vol_reduction_pct',
                'flooding_flow_reduction_pct',
                'outfall_flow_reduction_pct',
                'network_health',
                'total_cost'
            ])
            
            # Agregar variables de decisión
            X = self.result.X
            for i, key in enumerate(WEIGHT_KEYS):
                pareto_df[f'weight_{key}'] = X[:, i] / X[:, :6].sum(axis=1)  # normalizado
            pareto_df['capacity_max_hd'] = X[:, 6]
            
            path = self.results_dir / f"pareto_front_{timestamp}.csv"
            pareto_df.to_csv(path, index=False)
            print(f"[Results] Pareto front: {path}")
        
        # 2. Historial completo de evaluaciones
        if self.problem and self.problem.history:
            history_data = []
            for weights, result in self.problem.history:
                row = {
                    'weights': weights.to_array().tolist(),
                    **result.to_dict()
                }
                history_data.append(row)
            
            history_df = pd.DataFrame(history_data)
            path = self.results_dir / f"evaluation_history_{timestamp}.csv"
            history_df.to_csv(path, index=False)
            print(f"[Results] History: {path}")
        
        # 3. Mejores soluciones por objetivo
        if self.result and self.result.F is not None:
            best_by_obj = {}
            for i, name in enumerate([
                'flooding_vol', 'flooding_flow', 'outfall_flow', 
                'network_health', 'min_cost'
            ]):
                if i < 4:  # maximización
                    idx = np.argmin(self.result.F[:, i])
                else:  # minimización
                    idx = np.argmin(self.result.F[:, i])
                
                weights = RankingWeights.from_array(self.result.X[idx])
                best_by_obj[name] = {
                    'weights': {
                        'flow_over_capacity': weights.flow_over_capacity,
                        'flow_node_flooding': weights.flow_node_flooding,
                        'vol_over_capacity': weights.vol_over_capacity,
                        'vol_node_flooding': weights.vol_node_flooding,
                        'outfall_peak_flow': weights.outfall_peak_flow,
                        'failure_probability': weights.failure_probability,
                        'capacity_max_hd': weights.capacity_max_hd,
                    },
                    'objectives': {
                        'flooding_vol_reduction_pct': float(-self.result.F[idx, 0]),
                        'flooding_flow_reduction_pct': float(-self.result.F[idx, 1]),
                        'outfall_flow_reduction_pct': float(-self.result.F[idx, 2]),
                        'network_health': float(-self.result.F[idx, 3]),
                        'total_cost': float(self.result.F[idx, 4]),
                    }
                }
            
            path = self.results_dir / f"best_solutions_{timestamp}.json"
            with open(path, 'w') as f:
                json.dump(best_by_obj, f, indent=2)
            print(f"[Results] Best solutions: {path}")
    
    def visualize_results(self, save_dir: Optional[Path] = None):
        """
        Genera visualizaciones completas del frente de Pareto.
        """
        if self.result is None or self.result.F is None:
            print("[Visualize] No results to visualize")
            return
        
        save_dir = Path(save_dir) or self.results_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        F = self.result.F.copy()
        F[:, :4] = -F[:, :4]  # positivos para plot
        
        # Figura 1: Matriz de scatter plots 2D
        self._plot_pareto_matrix(F, save_dir, timestamp)
        
        # Figura 2: Gráfico de paralelas (parallel coordinates)
        self._plot_parallel_coordinates(F, save_dir, timestamp)
        
        # Figura 3: Distribución de variables de decisión
        self._plot_variable_distribution(save_dir, timestamp)
        
        # Figura 4: Trade-off costo vs beneficio
        self._plot_cost_benefit(F, save_dir, timestamp)
        
        print(f"[Visualize] All plots saved to {save_dir}")
    
    def _plot_pareto_matrix(self, F: np.ndarray, save_dir: Path, timestamp: str):
        """Matriz de scatter plots de todos los pares de objetivos."""
        n_obj = 5
        fig, axes = plt.subplots(n_obj, n_obj, figsize=(16, 16))
        
        obj_names = [
            'Flooding Vol\\nReduction',
            'Flooding Flow\\nReduction',
            'Outfall Flow\\nReduction',
            'Network\\nHealth',
            'Total Cost'
        ]
        
        for i in range(n_obj):
            for j in range(n_obj):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histograma
                    ax.hist(F[:, i], bins=20, color='steelblue', alpha=0.7)
                    ax.set_title(obj_names[i], fontsize=9)
                else:
                    # Off-diagonal: scatter
                    scatter = ax.scatter(F[:, j], F[:, i], c=F[:, 4], 
                                       cmap='viridis', alpha=0.6, s=20)
                    ax.set_xlabel(obj_names[j], fontsize=8)
                    ax.set_ylabel(obj_names[i], fontsize=8)
                
                ax.tick_params(labelsize=7)
        
        plt.tight_layout()
        path = save_dir / f"pareto_matrix_{timestamp}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Pareto matrix: {path}")
    
    def _plot_parallel_coordinates(self, F: np.ndarray, save_dir: Path, timestamp: str):
        """Gráfico de coordenadas paralelas."""
        from pandas.plotting import parallel_coordinates
        
        # Crear DataFrame para parallel_coordinates
        df_plot = pd.DataFrame(F, columns=[
            'flooding_vol', 'flooding_flow', 'outfall_flow',
            'network_health', 'total_cost'
        ])
        
        # Normalizar para visualización
        df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
        df_norm['solution_id'] = range(len(df_norm))
        
        # Samplear si hay muchas soluciones
        if len(df_norm) > 100:
            df_norm = df_norm.sample(100, random_state=42)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        parallel_coordinates(df_norm, 'solution_id', cols=df_norm.columns[:-1], 
                           ax=ax, alpha=0.3, linewidth=1)
        ax.set_title('Pareto Front - Parallel Coordinates (Normalized)')
        ax.legend_.remove()
        
        path = save_dir / f"parallel_coords_{timestamp}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Parallel coordinates: {path}")
    
    def _plot_variable_distribution(self, save_dir: Path, timestamp: str):
        """Distribución de las variables de decisión en el frente de Pareto."""
        if self.result is None:
            return
            
        X = self.result.X
        # Normalizar pesos
        weights_norm = X[:, :6] / X[:, :6].sum(axis=1, keepdims=True)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Pesos
        for i, key in enumerate(WEIGHT_KEYS):
            ax = axes[i]
            ax.hist(weights_norm[:, i], bins=20, color='coral', alpha=0.7)
            ax.set_title(f'Weight: {key}', fontsize=10)
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Frequency')
        
        # Capacity max h/d
        ax = axes[6]
        ax.hist(X[:, 6], bins=20, color='seagreen', alpha=0.7)
        ax.set_title('Capacity Max H/D', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Ocultar último eje
        axes[7].axis('off')
        
        plt.tight_layout()
        path = save_dir / f"variable_distribution_{timestamp}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Variable distribution: {path}")
    
    def _plot_cost_benefit(self, F: np.ndarray, save_dir: Path, timestamp: str):
        """Trade-off costo vs beneficio total."""
        # Beneficio agregado (promedio de las 4 primeras métricas)
        benefit = np.mean(F[:, :4], axis=1)
        cost = F[:, 4]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(cost, benefit, c=F[:, 0], cmap='RdYlGn', 
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Total Cost ($)', fontsize=12)
        ax.set_ylabel('Average Benefit Score', fontsize=12)
        ax.set_title('Cost-Benefit Trade-off\\n(Color = Flooding Vol Reduction)', fontsize=14)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Flooding Volume Reduction', rotation=270, labelpad=20)
        
        # Anotar soluciones extremas
        best_benefit_idx = np.argmax(benefit)
        best_cost_idx = np.argmin(cost)
        
        ax.annotate('Best Benefit', 
                   xy=(cost[best_benefit_idx], benefit[best_benefit_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate('Lowest Cost',
                   xy=(cost[best_cost_idx], benefit[best_cost_idx]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        path = save_dir / f"cost_benefit_{timestamp}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [Plot] Cost-benefit: {path}")
    
    def get_best_solution(self, 
                         preference: str = 'balanced',
                         cost_weight: float = 0.5) -> Optional[Dict]:
        """
        Retorna la mejor solución según preferencia.
        
        Args:
            preference: 'balanced', 'min_cost', 'max_flooding', 'max_health'
            cost_weight: para 'balanced', peso del costo vs beneficio (0-1)
        """
        if self.result is None or self.result.F is None:
            return None
        
        F = self.result.F.copy()
        F[:, :4] = -F[:, :4]  # positivos
        
        if preference == 'min_cost':
            idx = np.argmin(F[:, 4])
        elif preference == 'max_flooding':
            idx = np.argmax(F[:, 0])  # flooding_vol_reduction
        elif preference == 'max_health':
            idx = np.argmax(F[:, 3])  # network_health
        else:  # balanced
            benefit = np.mean(F[:, :4], axis=1)
            # Normalizar costo y beneficio
            cost_norm = (F[:, 4] - F[:, 4].min()) / (F[:, 4].max() - F[:, 4].min() + 1e-10)
            benefit_norm = (benefit - benefit.min()) / (benefit.max() - benefit.min() + 1e-10)
            score = benefit_norm - cost_weight * cost_norm
            idx = np.argmax(score)
        
        weights = RankingWeights.from_array(self.result.X[idx])
        
        return {
            'preference': preference,
            'weights': {
                'flow_over_capacity': weights.flow_over_capacity,
                'flow_node_flooding': weights.flow_node_flooding,
                'vol_over_capacity': weights.vol_over_capacity,
                'vol_node_flooding': weights.vol_node_flooding,
                'outfall_peak_flow': weights.outfall_peak_flow,
                'failure_probability': weights.failure_probability,
                'capacity_max_hd': weights.capacity_max_hd,
            },
            'performance': {
                'flooding_vol_reduction_pct': F[idx, 0],
                'flooding_flow_reduction_pct': F[idx, 1],
                'outfall_flow_reduction_pct': F[idx, 2],
                'network_health': F[idx, 3],
                'total_cost': F[idx, 4],
            }
        }


if __name__ == "__main__":

    # ================================================================
    # CONFIGURACIÓN - AJUSTAR ESTOS VALORES
    # ================================================================

    N_GENERATIONS = 100  # Número de generaciones
    POP_SIZE = 50  # Tamaño de población
    RESTORE_FROM = None  # Path a checkpoint para continuar, o None

    # Ejemplo: RESTORE_FROM = Path("optimization_results/nsga_ranking_checkpoints/checkpoint_gen_0025.pkl")

    # ================================================================
    # EJECUCIÓN
    # ================================================================

    print("=" * 80)
    print("NSGA-II RANKING OPTIMIZER")
    print("=" * 80)
    print(f"Generations: {N_GENERATIONS}")
    print(f"Population: {POP_SIZE}")
    print(f"Restore from: {RESTORE_FROM}")
    print("=" * 80)

    optimizer = NSGARankingOptimizer()

    optimizer.run(
        n_generations=N_GENERATIONS,
        pop_size=POP_SIZE,
        restore_from=RESTORE_FROM,
    )

    optimizer.visualize_results()

    print("\n" + "=" * 80)
    print("BEST SOLUTIONS SUMMARY")
    print("=" * 80)

    for pref in ['balanced', 'min_cost', 'max_flooding', 'max_health']:
        sol = optimizer.get_best_solution(preference=pref)
        if sol:
            print(f"\n{pref.upper()}:")
            print(f"  Weights:")
            for k, v in sol['weights'].items():
                print(f"    {k}: {v:.4f}")
            print(f"  Performance:")
            for k, v in sol['performance'].items():
                if 'cost' in k:
                    print(f"    {k}: ${v:,.0f}")
                else:
                    print(f"    {k}: {v:.4f}")

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)