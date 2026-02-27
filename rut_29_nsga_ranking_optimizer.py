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
import csv
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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
    'n_generations': config.N_GENERATIONS,
    'pop_size': config.POP_SIZE,
    'seed': 42,
    'checkpoint_dir': Path("optimization_results") / "nsga_ranking_checkpoints",
    'checkpoint_freq': 1,  # cada cuántas generaciones guardar
    'results_dir': Path("optimization_results") / "nsga_ranking_results",
    'max_tanks': config.MAX_TANKS,
    'max_iterations': config.MAX_ITERATIONS,
    'capacity_max_hd_bounds': (0.0, 0.95),
    # Early Stopping Configuration
    'early_stopping': {
        'enabled': True,
        'patience': 10,  # generaciones sin mejora antes de parar
        'min_delta': 0.001,  # mejora mínima significativa (1%)
        'metric': 'hypervolume',  # 'hypervolume', 'best_score', o 'igd'
        'window_size': 5,  # ventana móvil para suavizar
    },
    # Scoring Configuration
    'scoring': {
        'benefit_weights': {
            'flooding_vol_reduction': 0.35,
            'flooding_flow_reduction': 0.25,
            'outfall_flow_reduction': 0.20,
            'network_health': 0.20,
        },
        'cost_weight': 0.3,  # peso del costo en el score agregado
    }
}

# =============================================================================
# DETECCIÓN AUTOMÁTICA DE PESOS ACTIVOS
# =============================================================================

def get_active_weights() -> List[str]:
    """
    Detecta automáticamente qué pesos en FLOODING_RANKING_WEIGHTS son != 0.
    Solo esos pesos serán optimizados por NSGA-II.
    Los pesos = 0 se mantienen fijos.
    
    Returns:
        Lista de keys con valor != 0 en config.FLOODING_RANKING_WEIGHTS
    """
    import config
    active = []
    for key, value in config.FLOODING_RANKING_WEIGHTS.items():
        if abs(value) > 1e-10 or key in ['flow_over_capacity', 'flow_node_flooding', 
                                          'vol_node_flooding', 'outfall_peak_flow', 
                                          'failure_probability']:
            # Solo incluir si el valor inicial no es exactamente 0
            # o si es uno de los pesos conocidos
            if abs(value) > 1e-10:
                active.append(key)
    
    # Si no hay ninguno activo, usar todos por defecto
    if not active:
        active = ['flow_over_capacity', 'flow_node_flooding', 'vol_node_flooding',
                  'outfall_peak_flow', 'failure_probability']
    
    return active

# Lista dinámica de pesos activos (se actualiza al importar)
WEIGHT_KEYS = get_active_weights()
ALL_WEIGHT_KEYS = ['flow_over_capacity', 'flow_node_flooding', 'vol_node_flooding',
                   'outfall_peak_flow', 'failure_probability']  # 5 pesos posibles (vol_over_capacity eliminado)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass 
class SolutionRecord:
    """
    Registro completo de una solución evaluada.
    Se guarda en el CSV de evolución.
    """
    # Identificación
    generation: int
    individual_id: int
    timestamp: str
    eval_id: str
    
    # Variables de decisión (6 variables: 5 pesos + capacity_max_hd)
    weight_flow_over_capacity: float
    weight_flow_node_flooding: float
    weight_vol_node_flooding: float
    weight_outfall_peak_flow: float
    weight_failure_probability: float
    capacity_max_hd: float
    
    # Objetivos (5 objetivos - convertidos a positivos para legibilidad)
    flooding_vol_reduction_pct: float
    flooding_flow_reduction_pct: float
    outfall_flow_reduction_pct: float
    network_health: float
    total_cost: float
    
    # Scores calculados
    score_weighted: float          # Score ponderado por usuario
    score_benefit_cost: float      # Beneficio/Costo
    score_dominance: float         # Nivel de dominancia
    rank: int                      # Ranking en población
    crowding_distance: float       # Distancia de crowding NSGA-II
    is_pareto: bool                # Está en el frente de Pareto?
    
    # Métricas de evolución (diferencias)
    delta_vs_prev_gen: float       # Diferencia vs mejor de gen anterior
    delta_vs_best: float           # Diferencia vs mejor global
    delta_vs_baseline: float       # Diferencia vs baseline (si aplica)
    
    # Métricas de diversidad
    population_std: float          # Desviación estándar de objetivos en población
    hypervolume: float             # Hypervolume del frente (solo se guarda 1x por gen)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'individual_id': self.individual_id,
            'timestamp': self.timestamp,
            'eval_id': self.eval_id,
            'weight_flow_over_capacity': self.weight_flow_over_capacity,
            'weight_flow_node_flooding': self.weight_flow_node_flooding,
            'weight_vol_over_capacity': self.weight_vol_over_capacity,
            'weight_vol_node_flooding': self.weight_vol_node_flooding,
            'weight_outfall_peak_flow': self.weight_outfall_peak_flow,
            'weight_failure_probability': self.weight_failure_probability,
            'capacity_max_hd': self.capacity_max_hd,
            'flooding_vol_reduction_pct': self.flooding_vol_reduction_pct,
            'flooding_flow_reduction_pct': self.flooding_flow_reduction_pct,
            'outfall_flow_reduction_pct': self.outfall_flow_reduction_pct,
            'network_health': self.network_health,
            'total_cost': self.total_cost,
            'score_weighted': self.score_weighted,
            'score_benefit_cost': self.score_benefit_cost,
            'score_dominance': self.score_dominance,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'is_pareto': self.is_pareto,
            'delta_vs_prev_gen': self.delta_vs_prev_gen,
            'delta_vs_best': self.delta_vs_best,
            'delta_vs_baseline': self.delta_vs_baseline,
            'population_std': self.population_std,
            'hypervolume': self.hypervolume,
        }
    
    @staticmethod
    def get_csv_header() -> List[str]:
        return [
            'generation', 'individual_id', 'timestamp', 'eval_id',
            'weight_flow_over_capacity', 'weight_flow_node_flooding', 'weight_vol_node_flooding',
            'weight_outfall_peak_flow', 'weight_failure_probability',
            'capacity_max_hd', 'flooding_vol_reduction_pct', 'flooding_flow_reduction_pct',
            'outfall_flow_reduction_pct', 'network_health', 'total_cost',
            'score_weighted', 'score_benefit_cost', 'score_dominance', 'rank',
            'crowding_distance', 'is_pareto', 'delta_vs_prev_gen', 'delta_vs_best',
            'delta_vs_baseline', 'population_std', 'hypervolume'
        ]


@dataclass
class GenerationSummary:
    """Resumen de una generación para el JSON de evolución."""
    generation: int
    timestamp: str
    n_evaluations: int
    
    # Mejor solución de esta generación
    best_score: float
    best_individual_id: int
    best_flooding_reduction: float
    best_cost: float
    
    # Estadísticas de población
    mean_score: float
    std_score: float
    min_cost: float
    max_cost: float
    mean_cost: float
    
    # Métricas de convergencia
    hypervolume: float
    hypervolume_delta: float  # Cambio vs generación anterior
    diversity_index: float    # Índice de diversidad poblacional
    
    # Early stopping
    generations_without_improvement: int
    should_stop: bool


class OptimizationLogger:
    """
    Logger completo para la optimización NSGA-II.
    Guarda cada solución evaluada y resúmenes por generación.
    """
    
    def __init__(self, output_dir: Path, base_metrics_extractor: Optional[SystemMetrics] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_metrics = base_metrics_extractor.metrics
        
        # Archivos de salida
        self.detailed_csv = self.output_dir / "optimization_log_detailed.csv"
        self.summary_json = self.output_dir / "optimization_summary.json"
        
        # Estado
        self.records: List[SolutionRecord] = []
        self.generation_summaries: List[GenerationSummary] = []
        self.best_score_global = -float('inf')
        self.best_solution_global = None
        self.generation_best_scores: Dict[int, float] = {}
        self.reference_point = np.array([0, 0, 0, 0, 1e10])  # Para hypervolume
        
        # Crear CSV con header
        self._init_csv()
    
    def _init_csv(self):
        """Inicializa el CSV con el header (solo si no existe o está vacío)."""
        # Verificar si el archivo ya existe y tiene contenido
        if self.detailed_csv.exists() and self.detailed_csv.stat().st_size > 100:
            print(f"[Logger] CSV already exists with data, appending new generations...")
            return
        
        with open(self.detailed_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SolutionRecord.get_csv_header())
    
    def calculate_hypervolume(self, F: np.ndarray) -> float:
        """
        Calcula el hypervolume del frente de Pareto.
        F: matriz de objetivos (ya convertidos a positivos para maximización)
        """
        try:
            from pymoo.indicators.hv import Hypervolume
            
            # pymoo minimiza, así que negamos los objetivos de maximización
            F_minimize = F.copy()
            F_minimize[:, :4] = -F_minimize[:, :4]  # Negar los 4 primeros (maximización)
            
            # Punto de referencia: peor valor posible en cada objetivo
            ref_point = np.array([0, 0, 0, 0, 1e10])  # [max_flood, max_flow, max_outfall, max_health, max_cost]
            
            hv = Hypervolume(ref_point=ref_point)
            return hv.do(F_minimize)
        except Exception as e:
            print(f"[Logger] Warning: Could not calculate hypervolume: {e}")
            return 0.0
    
    def calculate_scores(self, objectives: np.ndarray, weights_norm: np.ndarray) -> Dict[str, float]:
        """
        Calcula múltiples scores para una solución.
        objectives: [flooding_vol, flooding_flow, outfall_flow, health, cost]
        """
        flooding_vol, flooding_flow, outfall_flow, health, cost = objectives
        
        # 1. Score ponderado (configurable por usuario)
        cfg = NSGA_CONFIG['scoring']
        w = cfg['benefit_weights']
        benefit = (
            flooding_vol * w['flooding_vol_reduction'] +
            flooding_flow * w['flooding_flow_reduction'] +
            outfall_flow * w['outfall_flow_reduction'] +
            health * w['network_health']
        )
        cost_normalized = cost / 1e6  # Normalizar costo a millones
        score_weighted = benefit - cfg['cost_weight'] * cost_normalized
        
        # 2. Score Beneficio/Costo
        if cost > 0:
            total_benefit = flooding_vol + flooding_flow + outfall_flow + health
            score_bc = total_benefit / (cost / 1e6)  # Beneficio por millón de dólares
        else:
            score_bc = 0.0
        
        return {
            'score_weighted': score_weighted,
            'score_benefit_cost': score_bc,
        }
    
    def log_generation(self, 
                       generation: int,
                       X: np.ndarray,  # Variables de decisión
                       F: np.ndarray,  # Objetivos (negativos para maximización)
                       algorithm,
                       should_stop: bool = False,
                       generations_without_improvement: int = 0,
                       active_weights: List[str] = None) -> GenerationSummary:
        """
        Loguea una generación completa.
        X: [n_individuals, n_var] - pesos activos + h_d
        F: [n_individuals, 5] - objetivos (negativos para maximización)
        active_weights: Lista de pesos que están siendo optimizados
        """
        timestamp = datetime.now().isoformat()
        n_individuals = X.shape[0]
        
        # Detectar pesos activos si no se proporcionan
        if active_weights is None:
            active_weights = get_active_weights()
        n_active = len(active_weights)
        
        # Convertir F a positivos para legibilidad
        F_pos = F.copy()
        F_pos[:, :4] = -F_pos[:, :4]  # Negar los 4 primeros (maximización)
        F_pos[:, 4] = F_pos[:, 4]      # Costo ya es minimización
        
        # Calcular hypervolume
        hypervolume = self.calculate_hypervolume(F_pos)
        hypervolume_delta = 0.0
        if self.generation_summaries:
            hypervolume_delta = hypervolume - self.generation_summaries[-1].hypervolume
        
        # Calcular diversidad poblacional (std promedio de objetivos)
        diversity = np.mean(np.std(F_pos, axis=0))
        
        # Encontrar rangos y crowding distance del algoritmo
        pop = algorithm.pop
        ranks = pop.get("rank") if pop.has("rank") else np.zeros(n_individuals)
        crowding = pop.get("crowding") if pop.has("crowding") else np.zeros(n_individuals)
        
        # Mejor solución de esta generación (por score ponderado)
        scores = []
        for i in range(n_individuals):
            scores_dict = self.calculate_scores(F_pos[i], X[i])
            scores.append(scores_dict['score_weighted'])
        scores = np.array(scores)
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Actualizar mejor global
        if best_score > self.best_score_global:
            self.best_score_global = best_score
            self.best_solution_global = {
                'generation': generation,
                'individual_id': best_idx,
                'objectives': F_pos[best_idx].tolist(),
                'weights': X[best_idx].tolist(),
                'score': best_score,
            }
        
        self.generation_best_scores[generation] = best_score
        
        # Calcular delta vs generación anterior
        delta_vs_prev = 0.0
        if generation > 1 and (generation - 1) in self.generation_best_scores:
            delta_vs_prev = best_score - self.generation_best_scores[generation - 1]
        
        # Guardar cada solución
        with open(self.detailed_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for i in range(n_individuals):
                scores_dict = self.calculate_scores(F_pos[i], X[i])
                
                # Normalizar pesos activos para guardar
                weights_raw = X[i, :n_active]
                weights_sum = np.sum(weights_raw)
                if weights_sum < 1e-10:
                    weights_norm = np.ones(n_active) / n_active
                else:
                    weights_norm = weights_raw / weights_sum
                
                # Construir diccionario de todos los pesos (activos + inactivos=0)
                weight_values = {key: 0.0 for key in ALL_WEIGHT_KEYS}
                for idx, key in enumerate(active_weights):
                    weight_values[key] = weights_norm[idx]
                
                # Diferencias
                delta_vs_best = scores[i] - self.best_score_global if self.best_score_global != -float('inf') else 0.0
                
                record = SolutionRecord(
                    generation=generation,
                    individual_id=i,
                    timestamp=timestamp,
                    eval_id=f"gen{generation:04d}_ind{i:03d}",
                    weight_flow_over_capacity=weight_values['flow_over_capacity'],
                    weight_flow_node_flooding=weight_values['flow_node_flooding'],
                    weight_vol_node_flooding=weight_values['vol_node_flooding'],
                    weight_outfall_peak_flow=weight_values['outfall_peak_flow'],
                    weight_failure_probability=weight_values['failure_probability'],
                    capacity_max_hd=np.clip(X[i, n_active], 0, 0.95),
                    flooding_vol_reduction_pct=F_pos[i, 0],
                    flooding_flow_reduction_pct=F_pos[i, 1],
                    outfall_flow_reduction_pct=F_pos[i, 2],
                    network_health=F_pos[i, 3],
                    total_cost=F_pos[i, 4],
                    score_weighted=scores_dict['score_weighted'],
                    score_benefit_cost=scores_dict['score_benefit_cost'],
                    score_dominance=-int(ranks[i]) if ranks is not None else 0,  # Negativo para que mayor sea mejor
                    rank=int(ranks[i]) if ranks is not None else 0,
                    crowding_distance=float(crowding[i]) if crowding is not None else 0.0,
                    is_pareto=(ranks[i] == 0) if ranks is not None else False,
                    delta_vs_prev_gen=delta_vs_prev if i == best_idx else 0.0,
                    delta_vs_best=delta_vs_best,
                    delta_vs_baseline=0.0,  # Se puede calcular si se pasa baseline
                    population_std=diversity,
                    hypervolume=hypervolume if i == 0 else 0.0,  # Solo guardar 1x por gen
                )
                
                writer.writerow([
                    record.generation, record.individual_id, record.timestamp, record.eval_id,
                    record.weight_flow_over_capacity, record.weight_flow_node_flooding,
                    record.weight_vol_node_flooding,
                    record.weight_outfall_peak_flow, record.weight_failure_probability,
                    record.capacity_max_hd, record.flooding_vol_reduction_pct,
                    record.flooding_flow_reduction_pct, record.outfall_flow_reduction_pct,
                    record.network_health, record.total_cost, record.score_weighted,
                    record.score_benefit_cost, record.score_dominance, record.rank,
                    record.crowding_distance, record.is_pareto, record.delta_vs_prev_gen,
                    record.delta_vs_best, record.delta_vs_baseline, record.population_std,
                    record.hypervolume
                ])
                
                self.records.append(record)
        
        # Crear resumen de generación
        summary = GenerationSummary(
            generation=generation,
            timestamp=timestamp,
            n_evaluations=n_individuals,
            best_score=best_score,
            best_individual_id=best_idx,
            best_flooding_reduction=F_pos[best_idx, 0],
            best_cost=F_pos[best_idx, 4],
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            min_cost=np.min(F_pos[:, 4]),
            max_cost=np.max(F_pos[:, 4]),
            mean_cost=np.mean(F_pos[:, 4]),
            hypervolume=hypervolume,
            hypervolume_delta=hypervolume_delta,
            diversity_index=diversity,
            generations_without_improvement=generations_without_improvement,
            should_stop=should_stop,
        )
        
        self.generation_summaries.append(summary)
        
        # Guardar JSON actualizado
        self._save_summary_json()
        
        return summary
    
    def _save_summary_json(self):
        """Guarda el resumen en JSON."""
        data = {
            'metadata': {
                'total_generations': len(self.generation_summaries),
                'total_evaluations': len(self.records),
                'best_score_global': self.best_score_global,
                'best_solution': self.best_solution_global,
            },
            'generations': [
                {
                    'generation': s.generation,
                    'timestamp': s.timestamp,
                    'n_evaluations': s.n_evaluations,
                    'best_score': s.best_score,
                    'best_individual_id': s.best_individual_id,
                    'best_flooding_reduction': s.best_flooding_reduction,
                    'best_cost': s.best_cost,
                    'mean_score': s.mean_score,
                    'std_score': s.std_score,
                    'min_cost': s.min_cost,
                    'max_cost': s.max_cost,
                    'mean_cost': s.mean_cost,
                    'hypervolume': s.hypervolume,
                    'hypervolume_delta': s.hypervolume_delta,
                    'diversity_index': s.diversity_index,
                    'generations_without_improvement': s.generations_without_improvement,
                    'should_stop': s.should_stop,
                }
                for s in self.generation_summaries
            ]
        }
        
        with open(self.summary_json, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def print_generation_summary(self, summary: GenerationSummary):
        """Imprime un resumen formateado de la generación."""
        print(f"\n{'='*80}")
        print(f"GENERATION {summary.generation} SUMMARY")
        print(f"{'='*80}")
        print(f"  Best Score:        {summary.best_score:.4f} (Ind #{summary.best_individual_id})")
        print(f"  Mean ± Std:        {summary.mean_score:.4f} ± {summary.std_score:.4f}")
        print(f"  Flooding Reduc:    {summary.best_flooding_reduction:.2%}")
        print(f"  Cost Range:        ${summary.min_cost:,.0f} - ${summary.max_cost:,.0f}")
        print(f"  Hypervolume:       {summary.hypervolume:.4e} (Δ {summary.hypervolume_delta:+.4e})")
        print(f"  Diversity:         {summary.diversity_index:.4f}")
        if summary.generations_without_improvement > 0:
            print(f"  ⚠ No Improvement:  {summary.generations_without_improvement} generations")
        if summary.should_stop:
            print(f"  ⛔ STOPPING: Early stopping triggered!")
        print(f"{'='*80}")


class EarlyStoppingMonitor:
    """
    Monitorea la evolución y detecta estancamiento.
    Soporta múltiples criterios: hypervolume, best_score, diversity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.enabled = config.get('enabled', True)
        self.patience = config.get('patience', 10)
        self.min_delta = config.get('min_delta', 0.001)
        self.metric = config.get('metric', 'hypervolume')
        self.window_size = config.get('window_size', 5)
        
        # Estado
        self.best_value = -float('inf')
        self.generations_without_improvement = 0
        self.history: List[float] = []
        self.should_stop = False
        self.stop_reason = None
        
        print(f"[EarlyStopping] Configured: metric={self.metric}, patience={self.patience}, "
              f"min_delta={self.min_delta}")
    
    def update(self, generation: int, summary: GenerationSummary) -> bool:
        """
        Actualiza el monitoreo con datos de la nueva generación.
        Retorna True si se debe detener.
        """
        if not self.enabled:
            return False
        
        # Obtener métrica actual
        if self.metric == 'hypervolume':
            current = summary.hypervolume
        elif self.metric == 'best_score':
            current = summary.best_score
        elif self.metric == 'diversity':
            current = summary.diversity_index
        else:
            current = summary.hypervolume
        
        self.history.append(current)
        
        # Verificar mejora
        improvement = current - self.best_value
        
        if improvement > self.min_delta:
            # Hubo mejora significativa
            self.best_value = current
            self.generations_without_improvement = 0
            print(f"[EarlyStopping] ✓ Improvement: {improvement:+.4e} (new best: {current:.4e})")
        else:
            # No hubo mejora significativa
            self.generations_without_improvement += 1
            print(f"[EarlyStopping] ⚠ No improvement: {self.generations_without_improvement}/{self.patience}")
        
        # Verificar condiciones de parada
        if self.generations_without_improvement >= self.patience:
            self.should_stop = True
            self.stop_reason = f"No improvement for {self.patience} generations"
            print(f"[EarlyStopping] ⛔ STOPPING: {self.stop_reason}")
            return True
        
        # Verificar convergencia por ventana móvil (std pequeña)
        if len(self.history) >= self.window_size:
            window = self.history[-self.window_size:]
            window_std = np.std(window)
            if window_std < self.min_delta * 0.1:  # Criterio más estricto
                self.should_stop = True
                self.stop_reason = f"Converged (window std={window_std:.4e})"
                print(f"[EarlyStopping] ⛔ STOPPING: {self.stop_reason}")
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado actual del monitoreo."""
        return {
            'enabled': self.enabled,
            'best_value': self.best_value,
            'generations_without_improvement': self.generations_without_improvement,
            'patience': self.patience,
            'should_stop': self.should_stop,
            'stop_reason': self.stop_reason,
            'metric': self.metric,
        }


# Añadir import csv al inicio del archivo si no está
import csv


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


class RankingWeights:
    """
    Contenedor DINÁMICO para los pesos de ranking.
    
    Detecta automáticamente qué pesos están activos en config.FLOODING_RANKING_WEIGHTS
    y solo optimiza esos. Los pesos = 0 se mantienen fijos.
    
    Variables siempre incluidas:
    - Los pesos con valor != 0 en FLOODING_RANKING_WEIGHTS
    - capacity_max_hd (siempre optimizable)
    """
    
    def __init__(self, **kwargs):
        """
        Inicializa pesos. Puede recibir:
        - Pesos individuales: RankingWeights(flow_over_capacity=0.5, ...)
        - O nada: RankingWeights() (usa valores de config)
        """
        self.active_weights = get_active_weights()  # Detectar dinámicamente
        self.n_active = len(self.active_weights)
        
        # Inicializar todos los pesos posibles
        self.weights = {key: 0.0 for key in ALL_WEIGHT_KEYS}
        self.weights['capacity_max_hd'] = kwargs.get('capacity_max_hd', 0.0)
        
        # Actualizar con valores pasados o de config
        for key in self.active_weights:
            if key in kwargs:
                self.weights[key] = kwargs[key]
            elif hasattr(config, 'FLOODING_RANKING_WEIGHTS') and key in config.FLOODING_RANKING_WEIGHTS:
                self.weights[key] = config.FLOODING_RANKING_WEIGHTS[key]
        
        # Valores fijos (no optimizables) - SIEMPRE 0
        for key in ALL_WEIGHT_KEYS:
            if key not in self.active_weights:
                self.weights[key] = 0.0  # Fijar en 0, ignorar config original
    
    def __getattr__(self, name: str) -> float:
        """Permite acceder como objeto.atributo"""
        if name in self.weights:
            return self.weights[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: float):
        """Permite asignar como objeto.atributo = valor"""
        if name in ['active_weights', 'n_active', 'weights']:
            super().__setattr__(name, value)
        elif hasattr(self, 'weights') and name in self.weights:
            self.weights[name] = value
        else:
            super().__setattr__(name, value)
    
    @property
    def capacity_max_hd(self) -> float:
        return self.weights.get('capacity_max_hd', 0.0)
    
    @capacity_max_hd.setter
    def capacity_max_hd(self, value: float):
        self.weights['capacity_max_hd'] = value
    
    def to_array(self) -> np.ndarray:
        """Convierte a array [w1, w2, ..., wn, capacity_max_hd]."""
        # Solo los pesos activos + capacity_max_hd
        active_values = [self.weights[k] for k in self.active_weights]
        return np.array(active_values + [self.capacity_max_hd])
    
    @classmethod
    def from_array(cls, x: np.ndarray, active_weights: List[str] = None) -> "RankingWeights":
        """
        Crea desde array de NSGA.
        
        Args:
            x: Array con [w1, w2, ..., wn, capacity_max_hd]
            active_weights: Lista de keys activos (si None, usa get_active_weights())
        """
        if active_weights is None:
            active_weights = get_active_weights()
        
        n_active = len(active_weights)
        
        # Normalizar pesos activos a suma=1
        weights_raw = x[:n_active]
        weights_sum = np.sum(weights_raw)
        if weights_sum < 1e-10:
            weights_norm = np.ones(n_active) / n_active
        else:
            weights_norm = weights_raw / weights_sum
        
        # Crear diccionario de pesos activos
        kwargs = {active_weights[i]: weights_norm[i] for i in range(n_active)}
        kwargs['capacity_max_hd'] = np.clip(x[n_active], 0, 0.95)
        
        return cls(**kwargs)
    
    def apply_to_config(self):
        """Aplica estos pesos a config.py global."""
        # Construir diccionario completo (activos + fijos)
        full_weights = {}
        for key in ALL_WEIGHT_KEYS:
            full_weights[key] = self.weights.get(key, 0.0)
        
        config.FLOODING_RANKING_WEIGHTS = full_weights
        config.CAPACITY_MAX_HD = self.capacity_max_hd
    
    def to_dict(self) -> Dict[str, float]:
        """Convierte a diccionario para serialización."""
        return {
            'weights': {k: self.weights[k] for k in ALL_WEIGHT_KEYS},
            'capacity_max_hd': self.capacity_max_hd,
            'active_weights': self.active_weights,
        }


# =============================================================================
# FUNCIÓN DE EVALUACIÓN PARA PARALELISMO
# =============================================================================

def _evaluate_individual_worker(args) -> np.ndarray:
    """
    Función standalone para evaluar un individuo en un proceso worker.
    DINÁMICA: Detecta automáticamente qué pesos están activos.
    
    Args:
        args: Tuple con (x_array, project_root, elev_file, 
                         max_tanks, max_iterations, eval_id, active_weights, baseline_extractor_path)
    
    Returns:
        Array de objetivos [f0, f1, f2, f3, f4]
    """
    x, project_root_str, elev_file_str, max_tanks, max_iterations, eval_id, active_weights, baseline_extractor_path = args
    
    try:
        # Importar dentro del worker
        import config
        config.setup_sys_path()
        
        from rut_10_run_tanque_tormenta import StormwaterOptimizationRunner
        
        n_active = len(active_weights)
        
        # Crear pesos desde el array (n_active pesos + capacity_max_hd)
        weights_raw = x[:n_active]
        weights_sum = np.sum(weights_raw)
        if weights_sum < 1e-10:
            weights_norm = np.ones(n_active) / n_active
        else:
            weights_norm = weights_raw / weights_sum
        
        # Construir diccionario de pesos (activos + fijos en CERO)
        # Los activos toman el valor normalizado del NSGA-II
        # Los NO activos SIEMPRE son 0 (ignorar config.py original)
        full_weights = {}
        for key in ALL_WEIGHT_KEYS:
            if key in active_weights:
                idx = active_weights.index(key)
                full_weights[key] = weights_norm[idx]
            else:
                full_weights[key] = 0.0  # FIJAR EN 0 - no usar config original
        
        config.FLOODING_RANKING_WEIGHTS = full_weights
        config.CAPACITY_MAX_HD = np.clip(x[n_active], 0, 0.95)
        
        # Ejecutar evaluación
        runner = StormwaterOptimizationRunner(
            project_root=Path(project_root_str),
            proj_to=config.PROJECT_CRS,
            eval_id=eval_id
        )
        
        # Preparar parametros EXPLICITOS para pasar al runner
        explicit_weights = full_weights.copy()
        explicit_capacity_hd = float(config.CAPACITY_MAX_HD)
        
        result = runner.run_sequential_analysis(
            max_tanks=max_tanks,
            max_iterations=max_iterations,
            min_tank_vol=config.TANK_MIN_VOLUME_M3,
            max_tank_vol=config.TANK_MAX_VOLUME_M3,
            stop_at_breakeven=True,
            breakeven_multiplier=50.0,
            elev_file=Path(elev_file_str) if elev_file_str else None,
            optimizer_mode='greedy',
            optimization_tr_list=config.TR_LIST,
            validation_tr_list=config.VALIDATION_TR_LIST,
            ranking_weights=explicit_weights,  # EXPLICITO
            capacity_max_hd=explicit_capacity_hd,  # EXPLICITO
            baseline_extractor_path=baseline_extractor_path,  # EXPLICITO: Ruta al extractor
        )
        
        # Devolver objetivos (negativos para maximización)
        return np.array([
            -result.flooding_vol_reduction_pct,
            -result.flooding_peak_flow_reduction_pct,
            -result.outfall_peak_flow_reduction_pct,
            -result.network_health,
            result.total_cost,
        ])
            
    except Exception as e:
        import traceback
        print(f"[Worker Error] {eval_id}: {e}")
        traceback.print_exc()
        return np.array([0, 0, 0, 0, 1e10])  # Valores malos en caso de error


# =============================================================================
# CLASES PRINCIPALES
# =============================================================================

class GreedyRunner:
    """
    Wrapper para ejecutar StormwaterOptimizationRunner con configuración temporal.
    Esta clase se encarga de restaurar config después de cada corrida.
    """
    
    # Contador global de evaluaciones (clase-level)
    _eval_counter = 0
    
    def __init__(self, 
                 project_root: Optional[Path] = None,
                 elev_file: Optional[Path] = None):
        self.project_root = project_root or config.PROJECT_ROOT
        self.elev_file = elev_file or config.ELEV_FILE
        self._original_config = self._save_config_state()
        
        # Incrementar contador y generar ID para esta evaluación
        GreedyRunner._eval_counter += 1
        self.eval_id = f"eval_{GreedyRunner._eval_counter:05d}"
        
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
                proj_to=config.PROJECT_CRS,
                eval_id=self.eval_id  # Pasar el ID de evaluación
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
    Definición del problema para pymoo con evaluación paralela.
    DINÁMICO: Detecta automáticamente qué pesos están activos en config.
    
    Variables: n_active (pesos != 0) + 1 (capacity_max_hd)
    Objetivos: 5 (4 maximizaciones negadas + 1 minimización)
    
    Paralelismo: Usa ProcessPoolExecutor para evaluar múltiples individuos
    en paralelo según config.NSGA_PARALLEL_WORKERS.
    """
    
    def __init__(self,
                 baseline_extractor_path: Path,  # Ruta al pickle del extractor
                 runner_factory: Callable[[], GreedyRunner],
                 max_tanks: int = 30,
                 max_iterations: int = 100,
                 n_workers: int = None):
        
        self.baseline_extractor_path = baseline_extractor_path
        self.runner_factory = runner_factory
        self.max_tanks = max_tanks
        self.max_iterations = max_iterations
        self.n_workers = n_workers or config.NSGA_PARALLEL_WORKERS
        self.project_root = config.PROJECT_ROOT
        self.elev_file = config.ELEV_FILE
        
        # Detectar pesos activos dinámicamente
        self.active_weights = get_active_weights()
        self.n_active = len(self.active_weights)
        self.n_var = self.n_active + 1  # pesos activos + capacity_max_hd
        
        print(f"[RankingOptimizationProblem] Pesos activos ({self.n_active}): {self.active_weights}")
        print(f"[RankingOptimizationProblem] Variables totales: {self.n_var}")
        
        # Contadores para logging
        self.eval_count = 0
        self.history: List[Tuple[RankingWeights, SolutionResult]] = []
        
        # Bounds: n_active pesos [0,1] + 1 h/d [0, 0.95]
        xl = np.array([0.0] * self.n_active + [NSGA_CONFIG['capacity_max_hd_bounds'][0]])
        xu = np.array([1.0] * self.n_active + [NSGA_CONFIG['capacity_max_hd_bounds'][1]])
        
        # elementwise=False porque procesamos batch completo con ProcessPool
        super().__init__(
            n_var=self.n_var,  # Dinámico: n_active pesos + capacity_max_hd
            n_obj=5,
            n_constr=0,
            xl=xl,
            xu=xu,
            elementwise=False,
        )
    
    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluación batch llamada por pymoo.
        X: Matriz de tamaño (n_individuals, n_var)
        """
        n_individuals = X.shape[0]
        F = np.zeros((n_individuals, 5))
        
        # Obtener número de generación actual (actualizado por el callback)
        gen = getattr(self, '_current_gen', 0)
        if gen == 0 and hasattr(self, '_current_gen_override'):
            gen = self._current_gen_override
        
        print(f"\n{'='*80}")
        print(f"NSGA PARALLEL EVALUATION: Gen {gen}, {n_individuals} individuals, {self.n_workers} workers")
        print(f"{'='*80}")
        
        # Preparar argumentos para workers (incluyendo active_weights)
        args_list = [
            (
                X[i], 
                str(self.project_root),
                str(self.elev_file) if self.elev_file else None,
                self.max_tanks,
                self.max_iterations,
                f"eval_gen{gen:03d}_ind{i:03d}",
                self.active_weights,  # Pasar para que el worker sepa qué pesos usar
                str(self.baseline_extractor_path)  # Ruta al extractor completo
            )
            for i in range(n_individuals)
        ]
        
        # Ejecutar en paralelo o secuencial
        if self.n_workers > 1 and n_individuals > 1:
            print(f"[Parallel] Starting {self.n_workers} workers...")
            
            # Usar spawn para evitar problemas en Windows
            ctx = multiprocessing.get_context('spawn')
            
            with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as executor:
                results = list(executor.map(_evaluate_individual_worker, args_list))
            
            for i, result in enumerate(results):
                F[i] = result
                self.eval_count += 1
                print(f"  ✓ Individual {i+1}/{n_individuals} completed")
        else:
            # Modo secuencial
            print(f"[Sequential] Running {n_individuals} evaluations...")
            for i, args in enumerate(args_list):
                self.eval_count += 1
                print(f"\n--- Individual {i+1}/{n_individuals} (Eval #{self.eval_count}) ---")
                F[i] = _evaluate_individual_worker(args)
                print(f"  ✓ Completed")
        
        print(f"\n{'='*80}")
        print(f"BATCH COMPLETE: {n_individuals} evaluations")
        print(f"{'='*80}\n")
        
        out["F"] = F


class NSGACheckpoint(Callback):
    """
    Callback mejorado con logging detallado y early stopping.
    Guarda checkpoints, loguea todas las soluciones, y detecta estancamiento.
    """
    
    def __init__(self, 
                 checkpoint_dir: Path, 
                 frequency: int = 1,
                 logger: Optional[OptimizationLogger] = None,
                 early_stopping: Optional[EarlyStoppingMonitor] = None,
                 problem: Optional[RankingOptimizationProblem] = None):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.history: List[Dict] = []
        
        # Logger y early stopping
        self.logger = logger
        self.early_stopping = early_stopping or EarlyStoppingMonitor(NSGA_CONFIG['early_stopping'])
        
        # Referencia al problema para actualizar la generación
        self.problem = problem
        
        # Control de terminación
        self.has_stopped = False
        self.stop_generation = None
        self._last_logged_gen = 0  # Track para evitar duplicados en CSV
    
    def notify(self, algorithm):
        gen = algorithm.n_gen
        
        # Actualizar la generación en el problema para que los workers la usen
        if self.problem is not None:
            self.problem._current_gen = gen
            self.problem._current_gen_override = gen
        
        # Obtener datos de la población
        X = algorithm.pop.get("X").copy()
        F = algorithm.pop.get("F").copy()
        
        # Guardar historial básico
        self.history.append({
            'generation': gen,
            'population': X,
            'objectives': F,
        })
        
        # Loguear generación con el logger detallado (solo si no hemos logueado esta gen)
        if self.logger is not None and gen > self._last_logged_gen:
            self._last_logged_gen = gen
            should_stop = self.has_stopped
            
            # Obtener pesos activos del problema si está disponible
            active_weights = None
            if self.problem is not None:
                active_weights = getattr(self.problem, 'active_weights', None)
            
            summary = self.logger.log_generation(
                generation=gen,
                X=X,
                F=F,
                algorithm=algorithm,
                should_stop=should_stop,
                generations_without_improvement=getattr(self.early_stopping, 'generations_without_improvement', 0),
                active_weights=active_weights
            )
            
            # Actualizar early stopping con el summary
            if self.early_stopping is not None:
                stop_triggered = self.early_stopping.update(gen, summary)
                
                if stop_triggered and not self.has_stopped:
                    self.has_stopped = True
                    self.stop_generation = gen
                    summary.should_stop = True
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED AT GENERATION {gen}")
                    print(f"{'='*80}\n")
                    
                    # Guardar checkpoint final
                    self._save_checkpoint(algorithm, gen, final=True)
                    
                    # Terminar el algoritmo
                    algorithm.termination.force_termination = True
                    return
                
                # Re-loguear si cambió el estado
                if summary.generations_without_improvement != self.early_stopping.generations_without_improvement:
                    summary.generations_without_improvement = self.early_stopping.generations_without_improvement
            
            # Imprimir resumen
            self.logger.print_generation_summary(summary)
        
        # Guardar checkpoint periódico
        if gen % self.frequency == 0:
            self._save_checkpoint(algorithm, gen)
    
    def _save_checkpoint(self, algorithm, gen: int, final: bool = False):
        """Guarda un checkpoint del algoritmo."""
        suffix = "_FINAL" if final else ""
        path = self.checkpoint_dir / f"checkpoint_gen_{gen:04d}{suffix}.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump({
                'algorithm': algorithm,
                'history': self.history,
                'config': NSGA_CONFIG,
                'early_stopping': self.early_stopping.get_status() if self.early_stopping else None,
                'stopped_early': final,
            }, f)
        
        print(f"[Checkpoint] Saved generation {gen} to {path}")
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Retorna información de convergencia."""
        if self.early_stopping is None:
            return {}
        
        status = self.early_stopping.get_status()
        return {
            'stopped_early': self.has_stopped,
            'stop_generation': self.stop_generation,
            'stop_reason': status.get('stop_reason'),
            'best_value': status.get('best_value'),
            'generations_without_improvement': status.get('generations_without_improvement'),
        }


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

        self.base_metrics_extractor: Optional[SystemMetrics] = None
        self.problem: Optional[RankingOptimizationProblem] = None
        self.algorithm: Optional[NSGA2] = None
        self.result = None
        
    def _create_runner(self):
        """Factory method para crear GreedyRunner instances."""
        return GreedyRunner(
            project_root=self.project_root,
            elev_file=self.elev_file,
        )
    
    def _run_baseline(self) -> 'MetricExtractor':
        """Corre baseline sin tanques y guarda el extractor completo."""
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
        
        # Guardar extractor completo para workers
        import pickle
        baseline_extractor_path = Path(config.CODIGOS_DIR) / "optimization_results" / "baseline_extractor.pkl"
        baseline_extractor_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_extractor_path, 'wb') as f:
            pickle.dump(extractor, f)
        print(f"[Baseline Extractor] Guardado en: {baseline_extractor_path}")
        
        return extractor
    
    def setup(self):
        """Inicializa baseline y problema."""
        print("[NSGARankingOptimizer] Initializing...")
        print(f"\n[Parallel Config]")
        print(f"  NSGA PARALLEL WORKERS: {config.NSGA_PARALLEL_WORKERS}\n")
        
        self.base_metrics_extractor = self._run_baseline()
        
        # Guardar ruta del pickle para los workers
        self.baseline_extractor_path = Path(config.CODIGOS_DIR) / "optimization_results" / "baseline_extractor.pkl"
        
        self.problem = RankingOptimizationProblem(
            baseline_extractor_path=self.baseline_extractor_path,
            runner_factory=self._create_runner,
            max_tanks=NSGA_CONFIG['max_tanks'],
            max_iterations=NSGA_CONFIG['max_iterations'],
            n_workers=config.NSGA_PARALLEL_WORKERS,
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
           restore_from: Optional[Path] = None,
           enable_logging: bool = True,
           enable_early_stopping: bool = True):
        """
        Corre la optimización NSGA-II con logging y early stopping.
        
        Args:
            n_generations: Si no None, override de NSGA_CONFIG
            pop_size: Si no None, override de NSGA_CONFIG
            restore_from: Path a checkpoint para continuar
            enable_logging: Guardar log detallado de todas las soluciones
            enable_early_stopping: Detener si no hay mejora
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
        print(f"{'='*80}")
        print(f"Generations: {n_gen}")
        print(f"Population: {pop}")
        print(f"Logging: {'enabled' if enable_logging else 'disabled'}")
        print(f"Early Stopping: {'enabled' if enable_early_stopping else 'disabled'}")
        print(f"{'='*80}\n")
        
        # Crear logger si está habilitado
        logger = None
        if enable_logging:
            log_dir = self.results_dir / "logs"
            logger = OptimizationLogger(
                output_dir=log_dir,
                base_metrics_extractor=self.base_metrics_extractor
            )
            print(f"[Logger] Detailed logs will be saved to: {log_dir}")
            print(f"  - CSV: {logger.detailed_csv}")
            print(f"  - JSON: {logger.summary_json}")
        
        # Crear early stopping monitor
        early_stopping = None
        if enable_early_stopping:
            early_stopping = EarlyStoppingMonitor(NSGA_CONFIG['early_stopping'])
        
        # Checkpoint callback integrado
        checkpoint = NSGACheckpoint(
            checkpoint_dir=NSGA_CONFIG['checkpoint_dir'],
            frequency=NSGA_CONFIG['checkpoint_freq'],
            logger=logger,
            early_stopping=early_stopping,
            problem=self.problem,  # Pasar referencia al problema para actualizar generación
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
        
        # Obtener información de convergencia
        convergence_info = checkpoint.get_convergence_info()
        
        print(f"\n{'='*80}")
        if convergence_info.get('stopped_early'):
            print("OPTIMIZATION STOPPED EARLY (Early Stopping)")
            print(f"Stop Reason: {convergence_info.get('stop_reason')}")
            print(f"Stopped at generation: {convergence_info.get('stop_generation')}")
        else:
            print("OPTIMIZATION COMPLETE (All generations)")
        print(f"{'='*80}\n")
        
        self._save_final_results()
        
        # Imprimir instrucciones para análisis
        if enable_logging and logger:
            print(f"\n{'='*80}")
            print("ANALYSIS INSTRUCTIONS")
            print(f"{'='*80}")
            print(f"\nPara analizar los resultados:")
            print(f"  1. Ver log detallado: {logger.detailed_csv}")
            print(f"  2. Ver resumen JSON: {logger.summary_json}")
            print(f"\n  En Python puedes cargar el CSV:")
            print(f"     import pandas as pd")
            print(f"     df = pd.read_csv('{logger.detailed_csv}')")
            print(f"     df_sorted = df.sort_values('score_weighted', ascending=False)")
            print(f"     print(df_sorted.head(10))")
            print(f"\n  O filtrar por generación:")
            print(f"     gen_best = df.loc[df.groupby('generation')['score_weighted'].idxmax()]")
            print(f"     print(gen_best[['generation', 'score_weighted', 'total_cost']])")
            print(f"{'='*80}")
    
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
            
            # Agregar variables de decisión (dinámico según pesos activos)
            X = self.result.X
            n_active = len(WEIGHT_KEYS)
            for i, key in enumerate(WEIGHT_KEYS):
                pareto_df[f'weight_{key}'] = X[:, i] / X[:, :n_active].sum(axis=1)  # normalizado
            pareto_df['capacity_max_hd'] = X[:, n_active]
            
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
        
        save_dir = Path(save_dir) if save_dir is not None else self.results_dir
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
        n_active = len(WEIGHT_KEYS)
        
        # Normalizar pesos (dinámico según pesos activos)
        weights_norm = X[:, :n_active] / X[:, :n_active].sum(axis=1, keepdims=True)
        
        # Calcular layout de subplots según número de variables
        n_total = n_active + 1  # pesos + capacity_max_hd
        n_cols = 3
        n_rows = (n_total + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        # Pesos
        for i, key in enumerate(WEIGHT_KEYS):
            ax = axes[i]
            ax.hist(weights_norm[:, i], bins=20, color='coral', alpha=0.7)
            ax.set_title(f'Weight: {key}', fontsize=10)
            ax.set_xlabel('Normalized Value')
            ax.set_ylabel('Frequency')
        
        # Capacity max h/d
        ax = axes[n_active]
        ax.hist(X[:, n_active], bins=20, color='seagreen', alpha=0.7)
        ax.set_title('Capacity Max H/D', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Ocultar ejes sobrantes
        for i in range(n_active + 1, len(axes)):
            axes[i].axis('off')
        
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


# =============================================================================
# BENCHMARK DE NÚCLEOS PARA SWMM (v2 - Modificando THREADS en .inp)
# =============================================================================

class SwmmCoreBenchmark:
    """
    Benchmark para determinar la configuración óptima de núcleos para SWMM.
    
    SWMM EPA controla paralelismo mediante la keyword THREADS en el archivo .inp
    Este benchmark modifica ese valor y mide el tiempo real de ejecución.
    
    Uso:
        benchmark = SwmmCoreBenchmark()
        benchmark.run(max_cores_percent=90, n_runs_per_config=3)
        benchmark.plot_results()
    """
    
    def __init__(self, swmm_file: Path = None, output_dir: Path = None):
        self.swmm_file = Path(swmm_file) if swmm_file else Path(config.SWMM_FILE)
        self.output_dir = Path(output_dir) if output_dir else Path("optimization_results") / "core_benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detectar núcleos disponibles
        self.total_cores = os.cpu_count()
        print(f"[Benchmark] Total cores detectados: {self.total_cores}")
        print(f"[Benchmark] Archivo INP: {self.swmm_file}")
        
        self.results = []
        
    def _set_threads_in_inp(self, n_threads: int) -> Path:
        """
        Crea una copia temporal del .inp con THREADS = n_threads.
        Retorna el path al archivo modificado.
        """
        # Leer archivo original
        with open(self.swmm_file, 'r') as f:
            content = f.read()
        
        # Buscar y reemplazar línea THREADS
        import re
        
        # Patrón: THREADS seguido de número (al inicio de línea o después de espacios)
        pattern = r'^(THREADS\s+)\d+'
        
        if re.search(pattern, content, re.MULTILINE):
            # Reemplazar valor existente
            modified = re.sub(pattern, rf'\g<1>{n_threads}', content, flags=re.MULTILINE)
        else:
            # Si no existe, agregar al final de la sección [OPTIONS]
            # Buscar donde termina [OPTIONS]
            options_pattern = r'(\[OPTIONS\].*?)(?=\[|\Z)'
            match = re.search(options_pattern, content, re.DOTALL)
            if match:
                # Insertar THREADS al final de [OPTIONS]
                end_pos = match.end()
                modified = content[:end_pos] + f"THREADS\t{n_threads}\n" + content[end_pos:]
            else:
                # No hay sección [OPTIONS], agregar al inicio
                modified = f"[OPTIONS]\nTHREADS\t{n_threads}\n\n" + content
        
        # Guardar archivo temporal
        temp_inp = self.output_dir / f"benchmark_{n_threads}_threads.inp"
        with open(temp_inp, 'w') as f:
            f.write(modified)
        
        return temp_inp
    
    def _run_swmm(self, inp_file: Path) -> float:
        """
        Ejecuta SWMM con el archivo .inp dado usando pyswmm.
        Retorna tiempo en segundos.
        """
        import time
        from pyswmm import Simulation
        
        out_file = self.output_dir / f"{inp_file.stem}.out"
        
        start_time = time.time()
        
        try:
            with Simulation(str(inp_file)) as sim:
                for _ in sim:
                    pass
                    
        except Exception as e:
            print(f"  [Error] SWMM simulation failed: {e}")
            return -1
        
        elapsed = time.time() - start_time
        
        # Limpiar archivos temporales
        if out_file.exists():
            out_file.unlink()
        
        return elapsed
        
    def run(self, max_cores_percent: float = 90, n_runs_per_config: int = 3):
        """
        Ejecuta benchmark desde 1 thread hasta max_cores_percent.
        
        Args:
            max_cores_percent: Porcentaje máximo de cores a usar (ej: 90)
            n_runs_per_config: Cuántas veces repetir cada configuración para promedio
        """
        max_cores = int(self.total_cores * max_cores_percent / 100)
        
        # Configuraciones a probar: 1, 2, 4, 6, 8, 12, 16... hasta max_cores
        configs = [1, 2, 4]
        c = 6
        while c <= max_cores:
            configs.append(c)
            c += 2
        
        if configs[-1] != max_cores and max_cores not in configs:
            configs.append(max_cores)
        
        print(f"[Benchmark] Configuraciones a probar (THREADS): {configs}")
        print(f"[Benchmark] Runs por configuración: {n_runs_per_config}")
        print("=" * 80)
        
        for n_cores in configs:
            print(f"\n[Benchmark] Testing THREADS = {n_cores} ...")
            
            # Crear archivo con THREADS modificado
            temp_inp = self._set_threads_in_inp(n_cores)
            
            times = []
            for run in range(n_runs_per_config):
                elapsed = self._run_swmm(temp_inp)
                if elapsed > 0:
                    times.append(elapsed)
                    print(f"  Run {run+1}/{n_runs_per_config}: {elapsed:.2f}s")
                else:
                    print(f"  Run {run+1}/{n_runs_per_config}: FALLIDO")
            
            if len(times) >= 2:  # Necesitamos al menos 2 runs válidos
                avg_time = np.mean(times)
                std_time = np.std(times)
                speedup = self.results[0]['avg_time'] / avg_time if self.results else 1.0
                
                self.results.append({
                    'n_cores': n_cores,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'times': times,
                    'speedup': speedup
                })
                
                print(f"  → Promedio: {avg_time:.2f}s ± {std_time:.2f}s | Speedup: {speedup:.2f}x")
            else:
                print(f"  → INSUFICIENTES RUNS VÁLIDOS, saltando...")
            
            # Limpiar archivo temporal inp
            if temp_inp.exists():
                temp_inp.unlink()
        
        self._save_results()
        
    def _save_results(self):
        """Guarda resultados en CSV."""
        if not self.results:
            print("[Benchmark] No hay resultados para guardar")
            return
            
        df = pd.DataFrame([
            {
                'threads': r['n_cores'],
                'avg_time_seconds': r['avg_time'],
                'std_time_seconds': r['std_time'],
                'speedup': r['speedup']
            }
            for r in self.results
        ])
        
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Benchmark] Resultados guardados en: {csv_path}")
        
    def plot_results(self):
        """Genera gráficos de análisis."""
        if not self.results or len(self.results) < 2:
            print("[Benchmark] No hay suficientes resultados para graficar")
            return None, None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SWMM Thread Scaling Analysis', fontsize=14, fontweight='bold')
        
        cores = [r['n_cores'] for r in self.results]
        times = [r['avg_time'] for r in self.results]
        speedups = [r['speedup'] for r in self.results]
        stds = [r['std_time'] for r in self.results]
        
        # 1. Tiempo vs THREADS
        ax = axes[0, 0]
        ax.errorbar(cores, times, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('THREADS en INP')
        ax.set_ylabel('Tiempo de Ejecución (s)')
        ax.set_title('Tiempo vs THREADS')
        ax.grid(True, alpha=0.3)
        
        # 2. Speedup vs THREADS
        ax = axes[0, 1]
        ideal = cores  # Línea ideal
        ax.plot(cores, ideal, 'k--', label='Speedup Ideal (lineal)', alpha=0.5, linewidth=2)
        ax.plot(cores, speedups, 'o-', linewidth=2, markersize=8, label='Speedup Real', color='coral')
        ax.set_xlabel('THREADS en INP')
        ax.set_ylabel('Speedup (T1/Tn)')
        ax.set_title('Speedup vs THREADS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Eficiencia (speedup / threads)
        ax = axes[1, 0]
        efficiencies = [s / c for s, c in zip(speedups, cores)]
        ax.plot(cores, efficiencies, 'o-', linewidth=2, markersize=8, color='green')
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Eficiencia 80%')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Eficiencia 60%')
        ax.axhline(y=0.5, color='darkred', linestyle='--', alpha=0.5, label='Eficiencia 50%')
        ax.set_xlabel('THREADS en INP')
        ax.set_ylabel('Eficiencia')
        ax.set_title('Eficiencia Paralela')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 4. Tabla de recomendaciones
        ax = axes[1, 1]
        ax.axis('off')
        
        # Encontrar punto óptimo: máximo speedup donde eficiencia > 60%
        best_idx = -1
        best_speedup = 0
        for i, (s, e) in enumerate(zip(speedups, efficiencies)):
            if e >= 0.5 and s > best_speedup:
                best_speedup = s
                best_idx = i
        
        if best_idx == -1:
            best_idx = len(cores) // 2  # Fallback al medio
        
        best_cores = cores[best_idx]
        pymoo_workers = max(1, self.total_cores - best_cores)
        
        table_data = [
            ['Configuración', 'THREADS SWMM', 'Workers pymoo', 'Total', 'Speedup', 'Efic.'],
            ['Mínima', 1, self.total_cores - 1, self.total_cores, f"{speedups[0]:.2f}x", "100%"],
            ['Óptima', best_cores, pymoo_workers, self.total_cores, 
             f"{speedups[best_idx]:.2f}x", f"{efficiencies[best_idx]:.0%}"],
            ['Máxima', max(cores), self.total_cores - max(cores), self.total_cores, 
             f"{speedups[-1]:.2f}x", f"{efficiencies[-1]:.0%}"],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.18, 0.18, 0.18, 0.12, 0.15, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Colorear header y óptima
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(6):
            table[(2, i)].set_facecolor('#E8F5E9')  # Resaltar fila óptima
        
        ax.set_title('Recomendaciones de Configuración\n(Resaltada = Óptima)', pad=20)
        
        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Benchmark] Gráfico guardado en: {plot_path}")
        
        # Resumen en consola
        print("\n" + "=" * 80)
        print("RESUMEN DEL BENCHMARK")
        print("=" * 80)
        print(f"Total cores disponibles: {self.total_cores}")
        print(f"\nConfiguración ÓPTIMA:")
        print(f"  THREADS en SWMM:  {best_cores}")
        print(f"  Workers pymoo:    {pymoo_workers}")
        print(f"  Speedup logrado:  {speedups[best_idx]:.2f}x")
        print(f"  Eficiencia:       {efficiencies[best_idx]:.1%}")
        print("\nPara aplicar esta configuración:")
        print(f"  1. Editar tu archivo INP: THREADS {best_cores}")
        print(f"  2. En NSGA_CONFIG usar: n_workers={pymoo_workers}")
        print("=" * 80)
        
        return best_cores, pymoo_workers

# =============================================================================
# MAIN - BENCHMARK MODE
# =============================================================================

if __name__ == "__main__":
    # CRÍTICO para Windows: freeze_support evita que procesos workers reejecuten el main
    multiprocessing.freeze_support()
    
    import sys
    
    # # MODO BENCHMARK - Ejecutar primero para determinar configuración óptima
    # print("=" * 80)
    # print("MODO BENCHMARK - Detectando configuración óptima de núcleos")
    # print("=" * 80)
    #
    # benchmark = SwmmCoreBenchmark()
    # benchmark.run(max_cores_percent=90, n_runs_per_config=3)
    # optimal_swmm_cores, optimal_pymoo_workers = benchmark.plot_results()
    #
    # print(f"\n[Resultado] Usar {optimal_swmm_cores} cores para cada SWMM")
    # print(f"[Resultado] Esto permite {optimal_pymoo_workers} workers en paralelo para pymoo")
    # print("\nActualiza NSGA_CONFIG con estos valores y vuelve a correr.")
    #
    # sys.exit(0)
    
    # NOTA: Una vez tengas los resultados del benchmark, comenta las líneas anteriores
    # y descomenta el código de optimización normal de abajo:
    #
    # ================================================================
    # CONFIGURACIÓN - AJUSTAR ESTOS VALORES
    # ================================================================

    N_GENERATIONS = config.N_GENERATIONS  # Número de generaciones
    POP_SIZE = config.POP_SIZE  # Tamaño de población
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